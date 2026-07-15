"""
ip_adapter_pipeline.py — SD1.5 + IP-Adapter-FaceID pipeline for identity unlearning.

IP-Adapter-FaceID injects ArcFace identity embeddings into SD1.5 cross-attention
via a lightweight adapter on top of the base UNet's cross-attention K/V projections.

Identity conditioning interface:
  - Input:  512-dim L2-normalized ArcFace embedding from InsightFace
  - Adapter: projects 512 → 768 (cross-attention dim), then injected via K/V decoupled CA
  - Output:  Standard SD1.5 generated image (512×512)

This module provides:
  1. `load_ip_adapter_faceid_pipeline()` — load SD1.5 + IP-Adapter-FaceID 
  2. `IPAdapterFaceIDForUnlearning` — wrapper that supports LoRA unlearning
  3. `generate_with_face_id()` — generate images given a face embedding

Architecture of IP-Adapter-FaceID injection:
  For each SD1.5 cross-attention block, a parallel IP cross-attention block is added:
    - to_k_ip, to_v_ip: project face embedding to KV space
    - The final attention = sum of text attention + face attention (weighted by scale)

For unlearning: we target to_k_ip and to_v_ip projections with LoRA adapters.
These are the exact parameters that control face identity. Patching them surgically
achieves identity forgetting while leaving the text-conditioned pathway intact.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# IP-Adapter cross-attention injection
# ---------------------------------------------------------------------------

class IPAttnProcessor(nn.Module):
    """
    IP-Adapter attention processor that adds a face-identity-conditioned
    parallel cross-attention path to an existing SD1.5 attention block.

    This is the standard IP-Adapter mechanism:
        attn_out = text_attn(Q, K_text, V_text) + scale * face_attn(Q, K_ip, V_ip)

    For unlearning, we inject LoRA into K_ip and V_ip (to_k_ip, to_v_ip).
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        num_tokens: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or hidden_size
        self.scale = scale
        self.num_tokens = num_tokens

        # IP-specific K and V projections (from face embedding space)
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, ip_hidden_states=None, *args, **kwargs):
        """
        Forward pass merging text cross-attention with face identity attention.
        """
        if ip_hidden_states is None and hasattr(self, '_ip_hidden_states'):
            ip_hidden_states = self._ip_hidden_states

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        B, seq_len, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, B)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # Standard text cross-attention
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # IP face identity cross-attention (if face embedding provided)
        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)
            ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
            ip_hidden = torch.bmm(ip_attention_probs, ip_value)
            ip_hidden = attn.batch_to_head_dim(ip_hidden)
            hidden_states = hidden_states + self.scale * ip_hidden

        # Linear projection + reshape
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Load IP-Adapter-FaceID weights into SD1.5
# ---------------------------------------------------------------------------


def load_ip_adapter_faceid_pipeline(
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    ip_adapter_path: str = None,
    ip_adapter_scale: float = 0.8,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    Load SD1.5 base model + IP-Adapter-FaceID weights.

    The IP-Adapter-FaceID adapter injects ArcFace embeddings into cross-attention
    by adding parallel K/V projection heads (to_k_ip, to_v_ip).

    Args:
        base_model_id:    HuggingFace ID for SD 1.5 base.
        ip_adapter_path:  Path to ip-adapter-faceid .bin file.
        ip_adapter_scale: Scale factor for face conditioning (0=ignore, 1=full).
        device:           Compute device.
        dtype:            Model dtype.

    Returns:
        (pipeline, ip_adapter_weights_dict) — pipeline ready for generation,
        and the raw IP adapter weights for LoRA injection.
    """
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler
    from diffusers.models.attention_processor import AttnProcessor2_0

    print(f"[IPAdapter] Loading SD1.5 base from '{base_model_id}'...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Load IP-Adapter weights
    ip_weights = {}
    if ip_adapter_path and os.path.exists(ip_adapter_path):
        print(f"[IPAdapter] Loading IP-Adapter-FaceID from '{ip_adapter_path}'...")
        state_dict = torch.load(ip_adapter_path, map_location="cpu")
        # IP-Adapter .bin has "image_proj_model" and "ip_adapter" keys
        ip_weights = state_dict.get("ip_adapter", state_dict)

        # Inject IP attention processors
        attn_procs = {}
        unet_sd = pipe.unet.state_dict()
        for name in pipe.unet.attn_processors.keys():
            cross_attn_dim = None if name.endswith("attn1.processor") else pipe.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = pipe.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipe.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipe.unet.config.block_out_channels[block_id]
            else:
                hidden_size = pipe.unet.config.block_out_channels[0]

            if cross_attn_dim is None:
                # Self-attention: use default processor
                attn_procs[name] = AttnProcessor2_0()
            else:
                # Cross-attention: inject IP adapter
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attn_dim,
                    scale=ip_adapter_scale,
                    num_tokens=4,
                ).to(device=device, dtype=dtype)
        pipe.unet.set_attn_processor(attn_procs)

        # Load IP-Adapter K/V weights per-processor with shape checking
        # ip_weights is keyed by the integer index in the FULL attn_processors dict (e.g. 1, 3, 5, ...)
        loaded = 0
        for i, (name, attn_proc) in enumerate(pipe.unet.attn_processors.items()):
            if isinstance(attn_proc, IPAttnProcessor):
                for proj_name in ("to_k_ip", "to_v_ip"):
                    wkey = f"{i}.{proj_name}.weight"
                    if wkey in ip_weights:
                        w = ip_weights[wkey]
                        layer = getattr(attn_proc, proj_name)
                        if w.shape == layer.weight.shape:
                            layer.weight.data.copy_(w.float())
                            loaded += 1
                        # else: shape mismatch — keep random init for that layer
        print(f"[IPAdapter] Loaded {loaded}/32 IP K/V weight tensors")
        
        # Inject the FaceID UNet LoRAs
        pipe = _inject_faceid_loras(pipe, ip_weights, dtype=dtype)

        
    else:
        print(f"[IPAdapter] No IP-Adapter weights found — using base SD1.5 only")

    # Image proj model for face embedding → token sequence
    # IP-Adapter-FaceID uses a simple linear projection (no image encoder needed)
    # The face embedding goes directly into a proj layer: 512 → num_tokens * cross_attn_dim
    image_proj_model = _build_face_proj_model(
        face_dim=512,
        cross_attention_dim=pipe.unet.config.cross_attention_dim,
        num_tokens=4,
        dtype=dtype,
        device=device,
    )
    if ip_adapter_path and os.path.exists(ip_adapter_path):
        proj_state = state_dict.get("image_proj", {})
        if proj_state:
            image_proj_model.load_state_dict(proj_state, strict=False)

    pipe.face_proj_model = image_proj_model
    print(f"[IPAdapter] Pipeline ready. Face proj: 512 → (4, {pipe.unet.config.cross_attention_dim})")

    return pipe


class FaceProjModel(nn.Module):
    """
    Projects ArcFace 512-dim embedding to (num_tokens, cross_attention_dim)
    token sequence for injection into SD1.5 cross-attention.
    """

    def __init__(self, face_dim: int = 512, cross_attention_dim: int = 768, num_tokens: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(face_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, cross_attention_dim * num_tokens)
        )
        self.norm = nn.LayerNorm(cross_attention_dim)
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim

    def forward(self, face_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            face_emb: (B, 512) ArcFace embeddings.
        Returns:
            (B, num_tokens, cross_attention_dim) token sequence.
        """
        tokens = self.proj(face_emb)  # (B, num_tokens * cross_attn_dim)
        tokens = tokens.reshape(-1, self.num_tokens, self.cross_attention_dim)
        tokens = self.norm(tokens)
        return tokens


def _build_face_proj_model(
    face_dim: int,
    cross_attention_dim: int,
    num_tokens: int,
    dtype: torch.dtype,
    device: str,
) -> FaceProjModel:
    model = FaceProjModel(face_dim, cross_attention_dim, num_tokens)
    model = model.to(dtype=dtype, device=device)
    return model


def _inject_faceid_loras(pipe, ip_weights: Dict[str, torch.Tensor], dtype: torch.dtype):
    """
    FaceID requires LoRAs on the base UNet's attention layers (to_q, to_k, to_v, to_out)
    to render faces correctly. This function loads them from the .bin weights.
    """
    from lora_utils import LoRALinear
    
    loaded = 0
    # Map index 0..31 to the actual attention modules
    for i, (name, _) in enumerate(pipe.unet.attn_processors.items()):
        # name is e.g. down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor
        attn_name = name.replace(".processor", "")
        try:
            attn_module = pipe.unet.get_submodule(attn_name)
        except AttributeError:
            continue
            
        for proj_name in ("to_q", "to_k", "to_v", "to_out"):
            lora_down_key = f"{i}.{proj_name}_lora.down.weight"
            lora_up_key = f"{i}.{proj_name}_lora.up.weight"
            
            if lora_down_key in ip_weights and lora_up_key in ip_weights:
                original = getattr(attn_module, proj_name)
                rank = ip_weights[lora_down_key].shape[0]
                
                # In SD1.5, to_out is sometimes a ModuleList [Linear, Dropout]
                is_module_list = isinstance(original, nn.ModuleList)
                original_layer = original[0] if is_module_list else original
                
                if isinstance(original_layer, LoRALinear):
                    lora_layer = original_layer
                else:
                    # alpha = rank means scale = 1.0 (which is standard for FaceID LoRAs)
                    lora_layer = LoRALinear(original_layer, rank=rank, alpha=rank)
                    
                    if is_module_list:
                        attn_module.to_out[0] = lora_layer
                    else:
                        setattr(attn_module, proj_name, lora_layer)
                
                # Load weights
                lora_layer.lora_A.data.copy_(ip_weights[lora_down_key].to(dtype))
                lora_layer.lora_B.data.copy_(ip_weights[lora_up_key].to(dtype))
                
                # Freeze FaceID LoRAs (we don't want to unlearn them, we only unlearn IP adapter layers)
                lora_layer.lora_A.requires_grad = False
                lora_layer.lora_B.requires_grad = False
                
                loaded += 2
                
    print(f"[IPAdapter] Loaded {loaded} FaceID UNet LoRA weight tensors")
    return pipe


# ---------------------------------------------------------------------------
# Generation with face identity conditioning
# ---------------------------------------------------------------------------

def generate_with_face_id(
    pipe,
    face_embedding: Union[np.ndarray, torch.Tensor],
    prompt: str = "a high quality photo of a person",
    negative_prompt: str = "blurry, bad quality, cartoon, low resolution, ugly",
    num_images: int = 1,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    ip_adapter_scale: float = 0.8,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    device: str = "cuda",
) -> List[Image.Image]:
    """
    Generate face images conditioned on an ArcFace embedding.

    Args:
        pipe:           SD1.5 + IP-Adapter-FaceID pipeline.
        face_embedding: (512,) or (N, 512) ArcFace embedding(s).
        prompt:         Text prompt.
        ...

    Returns:
        List of PIL Images.
    """
    if isinstance(face_embedding, np.ndarray):
        face_embedding = torch.tensor(face_embedding, dtype=torch.float32)
    if face_embedding.ndim == 1:
        face_embedding = face_embedding.unsqueeze(0)  # (1, 512)

    face_embedding = face_embedding.to(device=device, dtype=next(pipe.unet.parameters()).dtype)

    # Project face embedding to token sequence
    with torch.no_grad():
        ip_tokens = pipe.face_proj_model(face_embedding)  # (1, num_tokens, cross_attn_dim)
        ip_tokens = ip_tokens.repeat(num_images, 1, 1)
        if guidance_scale > 1.0:
            uncond_ip_tokens = pipe.face_proj_model(torch.zeros_like(face_embedding))
            uncond_ip_tokens = uncond_ip_tokens.repeat(num_images, 1, 1)
            ip_tokens = torch.cat([uncond_ip_tokens, ip_tokens], dim=0)

    # Set scale on all IP processors
    def set_attn_processor_scale(unet, scale):
        for name, attn_proc in unet.attn_processors.items():
            if isinstance(attn_proc, IPAttnProcessor):
                attn_proc.scale = scale

        # Update FaceID UNet LoRAs scale dynamically
        from lora_utils import LoRALinear
        for module in unet.modules():
            if isinstance(module, LoRALinear) and module.rank == 128:
                module.scale = scale

    set_attn_processor_scale(pipe.unet, ip_adapter_scale)

    # Monkey-patch the UNet's forward to inject IP hidden states
    # This hooks the IP face tokens into each cross-attention processor call
    _set_ip_hidden_states(pipe.unet, ip_tokens)

    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        output = pipe(
            prompt=[prompt] * num_images,
            negative_prompt=[negative_prompt] * num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )

    return output.images


def _set_ip_hidden_states(unet: nn.Module, ip_tokens: torch.Tensor):
    """
    Hook IP face tokens into all IPAttnProcessor instances.
    This replaces the need to modify the UNet's forward() signature.
    """
    for attn_proc in unet.attn_processors.values():
        if isinstance(attn_proc, IPAttnProcessor):
            # We'll store ip_hidden_states on the processor and retrieve in __call__
            attn_proc._ip_hidden_states = ip_tokens


# ---------------------------------------------------------------------------
# LoRA injection for IP-Adapter K/V projections (for unlearning)
# ---------------------------------------------------------------------------

def inject_lora_into_ip_adapter(
    pipe,
    rank: int = 16,
    alpha: float = 16.0,
) -> object:
    """
    Inject LoRA adapters into IP-Adapter's to_k_ip and to_v_ip projections.

    These are the ONLY parameters that control face identity conditioning.
    By training LoRA on them, we can selectively forget one identity while
    preserving all others and the base text-conditioned generation.

    Args:
        pipe:   SD1.5 + IP-Adapter-FaceID pipeline.
        rank:   LoRA rank.
        alpha:  LoRA scaling alpha.

    Returns:
        Modified pipe with LoRA-adapted IP projections.
    """
    from lora_utils import LoRALinear

    n_injected = 0
    for name, attn_proc in pipe.unet.attn_processors.items():
        if not isinstance(attn_proc, IPAttnProcessor):
            continue
        for proj_name in ("to_k_ip", "to_v_ip"):
            original = getattr(attn_proc, proj_name)
            if isinstance(original, nn.Linear):
                lora_layer = LoRALinear(original, rank=rank, alpha=alpha)
                setattr(attn_proc, proj_name, lora_layer)
                n_injected += 1

    print(f"[LoRA-IP] Injected LoRA into {n_injected} IP cross-attention projections")

    # Freeze everything except LoRA parameters on the IP projections
    for name, param in pipe.unet.named_parameters():
        param.requires_grad = False

    for name, param in pipe.unet.named_parameters():
        # ONLY unfreeze the unlearning LoRAs on to_k_ip and to_v_ip.
        # Do not unfreeze the base FaceID LoRAs (to_q, to_k, etc.)
        if ("to_k_ip.lora" in name or "to_v_ip.lora" in name) and ("lora_A" in name or "lora_B" in name):
            param.requires_grad = True

    # Unfreeze face_proj_model for fine-tuning to prevent identity collapse
    pipe.face_proj_model.requires_grad_(True)
    pipe.face_proj_model.to(torch.float32)

    trainable = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad) + sum(p.numel() for p in pipe.face_proj_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pipe.unet.parameters()) + sum(p.numel() for p in pipe.face_proj_model.parameters())
    print(f"[LoRA-IP] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    return pipe


def get_ip_lora_state_dict(pipe) -> Dict[str, torch.Tensor]:
    """Extract LoRA parameters from IP-Adapter processors."""
    from lora_utils import LoRALinear
    lora_sd = {}
    for proc_name, attn_proc in pipe.unet.attn_processors.items():
        if not isinstance(attn_proc, IPAttnProcessor):
            continue
        for proj_name in ("to_k_ip", "to_v_ip"):
            layer = getattr(attn_proc, proj_name, None)
            if isinstance(layer, LoRALinear):
                lora_sd[f"{proc_name}.{proj_name}.lora_A"] = layer.lora_A.data.clone()
                lora_sd[f"{proc_name}.{proj_name}.lora_B"] = layer.lora_B.data.clone()
    return lora_sd


def save_ip_lora_adapter(pipe, save_dir: str, metadata: Optional[Dict] = None):
    """Save IP-Adapter LoRA weights to directory."""
    import json
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lora_sd = get_ip_lora_state_dict(pipe)
    torch.save(lora_sd, save_dir / "ip_lora_weights.pt")

    # Also save face_proj_model if it was trained
    torch.save(pipe.face_proj_model.state_dict(), save_dir / "face_proj_model.pt")

    config = {
        "model_type": "ip_adapter_faceid_lora",
        "base_model": "runwayml/stable-diffusion-v1-5",
        "n_lora_params": len(lora_sd),
        "identity_conditioning": "arcface_512dim",
        **(metadata or {}),
    }
    with open(save_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[LoRA-IP] Saved {len(lora_sd)} LoRA params → {save_dir}")


def load_ip_lora_adapter(pipe, load_dir: str, device: str = "cuda") -> object:
    """Load saved IP-Adapter LoRA weights onto a pipeline."""
    load_dir = Path(load_dir)
    lora_path = load_dir / "ip_lora_weights.pt"
    proj_path = load_dir / "face_proj_model.pt"

    if lora_path.exists():
        lora_sd = torch.load(lora_path, map_location=device)
        # Apply to IP processors
        from lora_utils import LoRALinear
        for proc_name, attn_proc in pipe.unet.attn_processors.items():
            if not isinstance(attn_proc, IPAttnProcessor):
                continue
            for proj_name in ("to_k_ip", "to_v_ip"):
                layer = getattr(attn_proc, proj_name, None)
                if isinstance(layer, LoRALinear):
                    ka_key = f"{proc_name}.{proj_name}.lora_A"
                    kb_key = f"{proc_name}.{proj_name}.lora_B"
                    if ka_key in lora_sd:
                        layer.lora_A.data = lora_sd[ka_key].to(device)
                    if kb_key in lora_sd:
                        layer.lora_B.data = lora_sd[kb_key].to(device)
        print(f"[LoRA-IP] Loaded {len(lora_sd)} LoRA weights from {load_dir}")

    # Load face_proj_model weights if they exist in the load_dir
    if proj_path.exists():
        pipe.face_proj_model.load_state_dict(torch.load(proj_path, map_location=device))
        pipe.face_proj_model.to(dtype=pipe.unet.dtype)
        print("[IPAdapter] Loaded face_proj_model weights from adapter directory")
    else:
        print("[IPAdapter] Warning: no face_proj_model.pt found in load_dir")

    return pipe


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing IP-Adapter pipeline module...")
    print("  FaceProjModel...")
    proj = FaceProjModel(face_dim=512, cross_attention_dim=768, num_tokens=4)
    x = torch.randn(2, 512)
    out = proj(x)
    assert out.shape == (2, 4, 768), f"Expected (2,4,768), got {out.shape}"
    print(f"  FaceProjModel output shape: {out.shape}  ✓")

    print("  IPAttnProcessor initialization...")
    attn_proc = IPAttnProcessor(hidden_size=320, cross_attention_dim=768, scale=0.8)
    print(f"  to_k_ip: {attn_proc.to_k_ip}")
    print(f"  to_v_ip: {attn_proc.to_v_ip}")
    print("All tests passed!")
