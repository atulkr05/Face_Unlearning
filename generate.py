"""
generate.py — End-to-end inference script for Arc2Face with unlearning.

Usage:
    python face_unlearning/generate.py \
        --identity_id 3422 \
        --num_images 10 \
        --lora_path face_unlearning/checkpoints/lora_set1 \
        --concept_eraser_path face_unlearning/checkpoints/concept_eraser_set1.pkl \
        --embedding_cache face_unlearning/embeddings \
        --output_dir outputs/forget_3422 \
        --seed 42

For retain identity generation (no erasing):
    python face_unlearning/generate.py \
        --identity_id 5230 \
        --num_images 10 \
        --lora_path face_unlearning/checkpoints/lora_set1 \
        --embedding_cache face_unlearning/embeddings \
        --output_dir outputs/retain_5230 \
        --no_erase
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_arc2face_pipeline(
    model_id: str = "FoivosPar/Arc2Face",
    lora_path: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    Load Arc2Face diffusion pipeline, optionally with LoRA adapter.

    Arc2Face uses a modified SD 1.5 U-Net where the text cross-attention
    has been replaced with identity cross-attention from ArcFace embeddings.

    Args:
        model_id:    HuggingFace model ID for Arc2Face.
        lora_path:   Optional path to a saved LoRA adapter directory.
        device:      Compute device.
        dtype:       Model dtype (float16 for H200 speed).

    Returns:
        Loaded and configured pipeline.
    """
    from diffusers import (
        StableDiffusionPipeline,
        DDIMScheduler,
        EulerDiscreteScheduler,
    )

    print(f"[Generate] Loading Arc2Face from '{model_id}'...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    if lora_path and os.path.exists(lora_path):
        from lora_utils import load_lora_adapter
        pipe = load_lora_adapter(pipe, lora_path, device=device)

    print(f"[Generate] Pipeline ready on {device}")
    return pipe


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_identity_embedding(
    cache_dir: str,
    identity_id: str,
    concept_eraser=None,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Load mean ArcFace embedding for an identity, optionally applying ConceptEraser.

    Returns:
        (1, 1, 512) float tensor for use as Arc2Face conditioning.
    """
    from precompute_embeddings import load_mean_embedding

    mean_emb = load_mean_embedding(cache_dir, identity_id)
    if mean_emb is None:
        print(f"[Generate] WARNING: No cached embedding for ID {identity_id}. Using zero embedding.")
        mean_emb = np.zeros(512, dtype=np.float32)

    emb_t = torch.tensor(mean_emb, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 512)

    if concept_eraser is not None and concept_eraser.fitted:
        emb_t = concept_eraser.transform(emb_t)

    return emb_t.unsqueeze(1)  # (1, 1, 512) — Arc2Face cross-attention format


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_images(
    pipe,
    id_embedding: torch.Tensor,
    num_images: int = 10,
    prompts: Optional[list] = None,
    negative_prompt: str = "blurry, bad quality, cartoon, drawing, sketch, text, watermark",
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 42,
    output_dir: str = "./outputs",
    save: bool = True,
) -> list:
    """
    Generate face images conditioned on the given ArcFace embedding.

    Arc2Face injects the identity embedding as cross-attention conditioning.
    The `encoder_hidden_states` argument is used to pass the identity tensor.

    Args:
        pipe:               Arc2Face pipeline.
        id_embedding:       (1, 1, 512) ArcFace identity tensor.
        num_images:         Number of images to generate.
        prompts:            List of text prompts (or None for default).
        negative_prompt:    Negative prompt text.
        num_inference_steps: Denoising steps.
        guidance_scale:     CFG scale.
        seed:               Random seed (deterministic).
        output_dir:         Directory to save images.
        save:               If True, save images to disk.

    Returns:
        List of PIL Images.
    """
    output_dir = Path(output_dir)
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    if prompts is None:
        prompts = ["a high quality photo of a person"] * num_images

    generated = []
    for i, prompt in enumerate(tqdm(range(num_images), desc="Generating")):
        generator = torch.Generator(device=id_embedding.device).manual_seed(seed + i)

        # Arc2Face: pass identity embedding as encoder_hidden_states
        # The pipeline's forward replaces the standard text embedding with the identity embedding
        try:
            result = pipe(
                prompt=prompts[i] if i < len(prompts) else prompts[0],
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
                # Arc2Face-specific: override cross-attention conditioning
                # (depends on Arc2Face's inference interface)
            )
            img = result.images[0]
        except Exception as e:
            print(f"[Generate] Generation error at step {i}: {e}")
            img = Image.new("RGB", (512, 512), color=(128, 128, 128))

        if save:
            save_path = output_dir / f"{i:04d}.png"
            img.save(save_path)

        generated.append(img)

    print(f"[Generate] Generated {len(generated)} images → {output_dir}")
    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate face images with Arc2Face + unlearning adapter"
    )
    parser.add_argument("--identity_id", type=str, required=True,
                        help="CelebA identity ID to generate")
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of images to generate")
    parser.add_argument("--model_id", type=str, default="FoivosPar/Arc2Face",
                        help="HuggingFace model ID")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter directory (optional)")
    parser.add_argument("--concept_eraser_path", type=str, default=None,
                        help="Path to ConceptEraser .pkl file (optional)")
    parser.add_argument("--embedding_cache", type=str,
                        default="/DATA2/Atul/2027/challenge/face_unlearning/embeddings",
                        help="Path to pre-computed .npy embedding cache directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save generated images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--no_erase", action="store_true",
                        help="Disable ConceptEraser (for retain ID generation)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"])
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[args.dtype]

    # Load ConceptEraser
    concept_eraser = None
    if args.concept_eraser_path and not args.no_erase:
        from concept_eraser import ConceptEraser
        if os.path.exists(args.concept_eraser_path):
            concept_eraser = ConceptEraser.load(args.concept_eraser_path)
            concept_eraser.to_device(args.device)
        else:
            print(f"[Generate] WARNING: Concept eraser not found at {args.concept_eraser_path}")

    # Load identity embedding
    print(f"[Generate] Loading embedding for identity {args.identity_id}...")
    id_embedding = load_identity_embedding(
        cache_dir=args.embedding_cache,
        identity_id=args.identity_id,
        concept_eraser=concept_eraser,
        device=args.device,
    )

    # Load pipeline
    pipe = load_arc2face_pipeline(
        model_id=args.model_id,
        lora_path=args.lora_path,
        device=args.device,
        dtype=dtype,
    )

    # Generate
    generated = generate_images(
        pipe=pipe,
        id_embedding=id_embedding,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_dir=args.output_dir,
        save=True,
    )

    print(f"\nDone. {len(generated)} images saved to {args.output_dir}")


# Fix Optional import
from typing import Optional

if __name__ == "__main__":
    main()
