"""
lora_utils.py — LoRA adapter utilities for Arc2Face cross-attention unlearning.

Arc2Face injects identity via cross-attention in the SD 1.5 U-Net.
We inject LoRA adapters ONLY into the cross-attention key/value projections
(to_k, to_v) in the down/mid/up blocks of the U-Net.

Why only K/V?
  - The query Q comes from spatial features (not identity-specific).
  - Keys K and Values V are computed from the identity conditioning embedding.
  - Patching K/V projections precisely targets the identity pathway.
  - This leaves all spatial priors (pose, expression, lighting) intact → high GP + AR.

Usage:
    pipe = StableDiffusionPipeline.from_pretrained("FoivosPar/Arc2Face")
    pipe = inject_lora_to_crossattention(pipe, rank=16, alpha=16)
    # ... train ...
    save_lora_adapter(pipe, "./checkpoints/lora_set1/")
    # Later:
    pipe = load_lora_adapter(pipe, "./checkpoints/lora_set1/")
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA layer implementation
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    A linear layer with a low-rank adaptation.

    Replaces:  y = W·x
    With:      y = W·x + (B·A)·x · (alpha/rank)

    Where A ∈ R^{rank×in_features}, B ∈ R^{out_features×rank}.
    W is frozen; A and B are trainable.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scale = alpha / rank

        # Frozen original weight
        self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None
        # LoRA matrices are ALWAYS float32 for stable training with GradScaler
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32, device=original_linear.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32, device=original_linear.weight.device))

        # Initialize: A with kaiming, B with zeros (so LoRA starts as identity delta)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        # Cast LoRA parameters to the input dtype (e.g. float16) during forward pass
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        lora_out = F.linear(self.dropout(x), lora_A)  # (*, rank)
        lora_out = F.linear(lora_out, lora_B)         # (*, out_features)
        return base_out + lora_out * self.scale

    @property
    def merged_weight(self) -> torch.Tensor:
        """Return W + B·A (the fully merged weight matrix)."""
        return self.weight + (self.lora_B @ self.lora_A) * self.scale

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}"


# ---------------------------------------------------------------------------
# Injection and removal helpers
# ---------------------------------------------------------------------------

def _get_cross_attention_modules(unet: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Walk the U-Net and collect all cross-attention modules
    (those with both to_k and to_v projections from conditioning).
    Returns list of (name, module).
    """
    cross_attn_modules = []
    for name, module in unet.named_modules():
        # In diffusers: cross-attention has to_k, to_v, to_q, to_out
        if (
            hasattr(module, "to_k")
            and hasattr(module, "to_v")
            and hasattr(module, "to_q")
            and isinstance(module.to_k, nn.Linear)
        ):
            cross_attn_modules.append((name, module))
    return cross_attn_modules


def inject_lora_to_crossattention(
    pipe,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_projections: Tuple[str, ...] = ("to_k", "to_v"),
) -> object:
    """
    Inject LoRA adapters into cross-attention K/V projections of the U-Net.

    Args:
        pipe:               A diffusers StableDiffusionPipeline (with pipe.unet).
        rank:               LoRA rank (16 is standard, 4 for lightweight).
        alpha:              LoRA scaling factor (often equal to rank).
        dropout:            LoRA dropout probability.
        target_projections: Which projections to adapt ('to_k', 'to_v', optionally 'to_q').

    Returns:
        Modified pipe with LoRA-adapted cross-attention layers.
    """
    unet = pipe.unet
    cross_attn_modules = _get_cross_attention_modules(unet)

    n_injected = 0
    for name, module in cross_attn_modules:
        for proj_name in target_projections:
            original = getattr(module, proj_name)
            if not isinstance(original, nn.Linear):
                continue
            lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, proj_name, lora_layer)
            n_injected += 1

    print(f"[LoRA] Injected LoRA into {n_injected} cross-attention projections "
          f"(rank={rank}, alpha={alpha})")

    # Freeze everything except LoRA parameters
    for name, param in unet.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"[LoRA] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return pipe


def get_lora_state_dict(unet: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only the LoRA parameters from the U-Net as a state dict."""
    lora_sd = {}
    for name, param in unet.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_sd[name] = param.data.clone()
    return lora_sd


def save_lora_adapter(
    pipe,
    save_dir: str,
    metadata: Optional[Dict] = None,
):
    """
    Save LoRA adapter weights + metadata to a directory.

    Saved files:
        lora_weights.pt     — LoRA state dict {param_name: tensor}
        adapter_config.json — LoRA config (rank, alpha, target projections)

    Args:
        pipe:       Diffusers pipeline with LoRA-injected U-Net.
        save_dir:   Directory path to save the adapter.
        metadata:   Optional extra metadata to include in config.
    """
    import json
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lora_sd = get_lora_state_dict(pipe.unet)
    torch.save(lora_sd, save_dir / "lora_weights.pt")
    print(f"[LoRA] Saved {len(lora_sd)} LoRA tensors → {save_dir / 'lora_weights.pt'}")

    # Infer config from the saved layers
    cross_attn_modules = _get_cross_attention_modules(pipe.unet)
    rank = None
    alpha = None
    for _, module in cross_attn_modules:
        if isinstance(module.to_k, LoRALinear):
            rank = module.to_k.rank
            alpha = module.to_k.scale * rank
            break

    config = {
        "lora_rank": rank,
        "lora_alpha": alpha,
        "target_projections": ["to_k", "to_v"],
        "base_model": "FoivosPar/Arc2Face",
        "n_lora_params": len(lora_sd),
        **(metadata or {}),
    }
    with open(save_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[LoRA] Saved config → {save_dir / 'adapter_config.json'}")


def load_lora_adapter(
    pipe,
    load_dir: str,
    device: Optional[str] = None,
) -> object:
    """
    Load and apply a saved LoRA adapter onto a fresh pipeline.

    If LoRA is not yet injected, this function injects it first using
    the saved config, then loads the weights.

    Args:
        pipe:       Diffusers pipeline (with or without LoRA already injected).
        load_dir:   Directory containing lora_weights.pt + adapter_config.json.
        device:     Target device string (None = auto from pipe).

    Returns:
        Modified pipe with LoRA weights applied.
    """
    import json
    load_dir = Path(load_dir)

    with open(load_dir / "adapter_config.json") as f:
        config = json.load(f)

    rank = config.get("lora_rank", 16)
    alpha = config.get("lora_alpha", 16.0)

    # Check if LoRA is already injected
    cross_attn_modules = _get_cross_attention_modules(pipe.unet)
    already_injected = any(
        isinstance(getattr(m, "to_k", None), LoRALinear)
        for _, m in cross_attn_modules
    )

    if not already_injected:
        pipe = inject_lora_to_crossattention(pipe, rank=rank, alpha=alpha)

    # Load weights
    dev = device or next(pipe.unet.parameters()).device
    lora_sd = torch.load(load_dir / "lora_weights.pt", map_location=dev)

    missing, unexpected = pipe.unet.load_state_dict(lora_sd, strict=False)
    if unexpected:
        print(f"[LoRA] WARNING: {len(unexpected)} unexpected keys in state dict")
    print(f"[LoRA] Loaded {len(lora_sd)} LoRA weights from {load_dir}")

    return pipe


def merge_lora_into_base(pipe) -> object:
    """
    Permanently merge LoRA weights into the base weights:
        W_new = W_orig + B·A · scale
    Then remove the LoRA adapter, returning a standard linear layer.
    Useful for producing a merged checkpoint for submission.
    """
    unet = pipe.unet
    cross_attn_modules = _get_cross_attention_modules(unet)
    n_merged = 0

    for _, module in cross_attn_modules:
        for proj_name in ("to_k", "to_v", "to_q"):
            layer = getattr(module, proj_name, None)
            if not isinstance(layer, LoRALinear):
                continue
            merged_w = layer.merged_weight.detach()
            new_linear = nn.Linear(layer.in_features, layer.out_features,
                                   bias=(layer.bias is not None))
            new_linear.weight = nn.Parameter(merged_w)
            if layer.bias is not None:
                new_linear.bias = nn.Parameter(layer.bias.detach())
            setattr(module, proj_name, new_linear)
            n_merged += 1

    print(f"[LoRA] Merged {n_merged} LoRA layers into base weights")
    return pipe


if __name__ == "__main__":
    # Smoke test (no model download needed)
    print("Testing LoRALinear...")
    linear = nn.Linear(512, 512)
    lora = LoRALinear(linear, rank=16, alpha=16.0)

    x = torch.randn(4, 512)
    y = lora(x)
    assert y.shape == (4, 512), f"Expected (4, 512), got {y.shape}"

    # At init, LoRA delta = 0 (B initialized to zeros)
    delta = torch.abs(y - linear(x)).max().item()
    assert delta < 1e-5, f"LoRA delta at init should be ~0, got {delta}"
    print(f"  Output shape: {y.shape}, init delta: {delta:.2e}  ✓")
    print("LoRALinear test passed!")
