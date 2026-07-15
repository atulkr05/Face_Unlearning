"""
main.py — Orchestrator for the Face Identity Unlearning Pipeline

Supports two modes:
  1. arc2face (production): Full Arc2Face + LoRA + ConceptEraser pipeline
  2. legacy (testing):      Original dummy-model pipeline for quick sanity checks

Arc2Face Mode Usage:
    python face_unlearning/main.py \
        --mode arc2face \
        --data_dir CelebAHQ/Img/img_celeba \
        --identity_file CelebAHQ/Anno/identity_CelebA.txt \
        --splits_file face_unlearning/validation-splits.json \
        --split_name "Face Set 1" \
        --embedding_cache face_unlearning/embeddings \
        --checkpoint_dir face_unlearning/checkpoints \
        --epochs 20 \
        --gpu_id 0

Legacy Mode Usage:
    python face_unlearning/main.py \
        --mode legacy \
        --data_dir /path/to/data \
        --splits_file face_unlearning/validation-splits.json \
        --split_name "Face Set 1"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Dummy models (kept for legacy mode)
# ---------------------------------------------------------------------------

class DummyFaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, images, conditions):
        return self.conv(images)


class DummyArcFaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 256 * 256, 512)

    def forward(self, images):
        batch_size = images.size(0)
        out = self.fc(images.view(batch_size, -1))
        return torch.nn.functional.normalize(out, p=2, dim=1)


# ---------------------------------------------------------------------------
# Arc2Face pipeline runner
# ---------------------------------------------------------------------------

def run_arc2face_pipeline(args, device: str):
    """Full Arc2Face unlearning pipeline."""
    from data_loader import get_celeba_dataloaders, DIFFUSION_TRANSFORM
    from precompute_embeddings import precompute_embeddings_for_ids
    from concept_eraser import ConceptEraser
    from lora_utils import inject_lora_to_crossattention, save_lora_adapter
    from unlearn import train_arc2face_unlearning
    from evaluate import run_full_evaluation, load_arcface_model

    # --- Parse split info ---
    with open(args.splits_file) as f:
        splits_data = json.load(f)

    split_info = None
    for s in splits_data["splits"]:
        if s["set"] == args.split_name and s["track"] == "face":
            split_info = s
            break

    if split_info is None:
        raise ValueError(f"Split '{args.split_name}' not found in {args.splits_file}")

    forget_id = split_info["forget_id"]
    retain_ids = split_info["retain_ids"]
    all_ids = [forget_id] + retain_ids

    print(f"\n{'='*60}")
    print(f"  Face Identity Unlearning — {args.split_name}")
    print(f"  Forget ID: {forget_id}")
    print(f"  Retain IDs: {retain_ids}")
    print(f"{'='*60}\n")

    checkpoint_dir = Path(args.checkpoint_dir) / args.split_name.replace(" ", "_")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eraser_path = checkpoint_dir / "concept_eraser.pkl"
    lora_path = checkpoint_dir / "lora_adapter"

    # --- Step 1: Precompute embeddings ---
    print("[Step 1] Precomputing ArcFace embeddings...")
    if not args.identity_file or not os.path.exists(args.identity_file):
        print(f"  ERROR: identity_CelebA.txt not found at: {args.identity_file}")
        print("  Download from: https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS")
        print("  Place at: CelebAHQ/Anno/identity_CelebA.txt")
        sys.exit(1)

    embeddings = precompute_embeddings_for_ids(
        img_dir=args.data_dir,
        identity_file=args.identity_file,
        identity_ids=all_ids,
        output_dir=args.embedding_cache,
        model_name=args.arcface_model,
        gpu_id=args.gpu_id,
        force_recompute=args.force_recompute,
    )

    import numpy as np
    forget_embs = embeddings.get(forget_id, np.zeros((0, 512)))
    retain_embs_list = [embeddings.get(r, np.zeros((0, 512))) for r in retain_ids]
    retain_embs_all = np.concatenate(
        [e for e in retain_embs_list if len(e) > 0], axis=0
    ) if any(len(e) > 0 for e in retain_embs_list) else np.zeros((1, 512))

    print(f"  Forget ID {forget_id}: {len(forget_embs)} embeddings")
    for rid, embs in zip(retain_ids, retain_embs_list):
        print(f"  Retain ID {rid}: {len(embs)} embeddings")

    # --- Step 2: Fit ConceptEraser ---
    print("\n[Step 2] Fitting ConceptEraser (Stage 1 — embedding-space nullification)...")
    if not eraser_path.exists() or args.force_recompute:
        concept_eraser = ConceptEraser(dim=512)
        if len(forget_embs) > 0 and len(retain_embs_all) > 0:
            concept_eraser.fit(forget_embs, retain_embs_all, n_directions=3)
        else:
            print("  WARNING: insufficient embeddings, using identity eraser")
            concept_eraser.fitted = True
        concept_eraser.save(eraser_path)
    else:
        concept_eraser = ConceptEraser.load(eraser_path)

    # --- Step 3: Load Arc2Face + Inject LoRA ---
    print("\n[Step 3] Loading Arc2Face pipeline + injecting LoRA adapters...")
    try:
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    except Exception as e:
        print(f"  ERROR loading Arc2Face: {e}")
        print("  Tip: run `huggingface-cli login` and ensure HF_TOKEN is set.")
        sys.exit(1)

    pipe = inject_lora_to_crossattention(pipe, rank=args.lora_rank, alpha=args.lora_rank)

    # --- Step 4: Build DataLoaders ---
    print("\n[Step 4] Building data loaders...")
    t_loader, r_loader = get_celeba_dataloaders(
        img_dir=args.data_dir,
        identity_file=args.identity_file,
        forget_id=forget_id,
        retain_ids=retain_ids,
        batch_size=args.batch_size,
        max_per_id=args.max_images_per_id,
    )
    print(f"  Target loader: {len(t_loader)} batches")
    print(f"  Retain loader: {len(r_loader)} batches")

    # --- Step 5: Train Unlearning ---
    print(f"\n[Step 5] Training unlearning ({args.epochs} epochs)...")
    pipe = train_arc2face_unlearning(
        pipe=pipe,
        target_loader=t_loader,
        retain_loader=r_loader,
        concept_eraser=concept_eraser,
        forget_id=forget_id,
        retain_ids=retain_ids,
        embedding_cache_dir=args.embedding_cache,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        checkpoint_dir=str(lora_path),
        log_every=args.log_every,
    )

    # --- Step 6: Evaluate ---
    print("\n[Step 6] Running full evaluation...")
    arcface_app = load_arcface_model(model_name=args.arcface_model, gpu_id=args.gpu_id)
    output_dir = checkpoint_dir / "evaluation"

    results = run_full_evaluation(
        pipe=pipe,
        concept_eraser=concept_eraser,
        arcface_app=arcface_app,
        forget_id=forget_id,
        retain_ids=retain_ids,
        embedding_cache_dir=args.embedding_cache,
        output_dir=str(output_dir),
        n_samples_per_id=args.eval_samples,
        device=device,
    )

    # Save final results summary
    results_summary = checkpoint_dir / "results_summary.json"
    with open(results_summary, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Results saved to {results_summary}")
    return results


# ---------------------------------------------------------------------------
# Legacy mode runner
# ---------------------------------------------------------------------------

def run_legacy_pipeline(args, device: str):
    """Original dummy-model pipeline for quick sanity testing."""
    from data_loader import get_dataloaders
    from unlearn import train_unlearning
    from evaluate import evaluate_unlearned_model

    print(f"Loading data from {args.data_dir} for split {args.split_name}...")
    try:
        t_loader, r_loader = get_dataloaders(args.data_dir, args.splits_file, args.split_name)
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    print("Initializing dummy models...")
    base_model = DummyFaceModel().to(device)
    arcface = DummyArcFaceModel().to(device)

    print("\n--- Starting Legacy Unlearning ---")
    unlearned_model = train_unlearning(
        model=base_model,
        target_loader=t_loader,
        retain_loader=r_loader,
        num_epochs=args.epochs,
        device=device,
    )

    print("\n--- Evaluating Unlearned Model ---")
    evaluate_unlearned_model(
        model=unlearned_model,
        arcface_model=arcface,
        target_loader=t_loader,
        retain_loader=r_loader,
        device=device,
    )

    print("\nLegacy pipeline complete.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Face Identity Unlearning — Main Runner")

    # Mode
    parser.add_argument("--mode", type=str, default="arc2face",
                        choices=["arc2face", "legacy"],
                        help="Pipeline mode: 'arc2face' (production) or 'legacy' (dummy model test)")

    # Data
    parser.add_argument("--data_dir", type=str,
                        default="/DATA2/Atul/2027/challenge/CelebAHQ/Img/img_celeba",
                        help="Path to CelebA images directory")
    parser.add_argument("--identity_file", type=str,
                        default="/DATA2/Atul/2027/challenge/CelebAHQ/Anno/identity_CelebA.txt",
                        help="Path to identity_CelebA.txt")
    parser.add_argument("--splits_file", type=str,
                        default="/DATA2/Atul/2027/challenge/face_unlearning/validation-splits.json")
    parser.add_argument("--split_name", type=str, default="Face Set 1",
                        help="Which validation split to use: 'Face Set 1' or 'Face Set 2'")
    parser.add_argument("--embedding_cache", type=str,
                        default="/DATA2/Atul/2027/challenge/face_unlearning/embeddings")

    # Model
    parser.add_argument("--model_id", type=str, default="FoivosPar/Arc2Face")
    parser.add_argument("--arcface_model", type=str, default="buffalo_l",
                        choices=["buffalo_l", "antelopev2"])
    parser.add_argument("--lora_rank", type=int, default=16)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_images_per_id", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0,   help="Forget loss weight")
    parser.add_argument("--beta", type=float, default=2.0,    help="Retain loss weight")
    parser.add_argument("--gamma", type=float, default=0.5,   help="ID retain loss weight")
    parser.add_argument("--delta", type=float, default=0.01,  help="EWC loss weight")
    parser.add_argument("--log_every", type=int, default=10)

    # Checkpoints
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/DATA2/Atul/2027/challenge/face_unlearning/checkpoints")
    parser.add_argument("--force_recompute", action="store_true")

    # Eval
    parser.add_argument("--eval_samples", type=int, default=30)

    # Hardware
    parser.add_argument("--gpu_id", type=int, default=0)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"[Main] Device: {device}")

    # Add face_unlearning to path for module imports
    sys.path.insert(0, str(Path(__file__).parent))

    if args.mode == "arc2face":
        run_arc2face_pipeline(args, device)
    else:
        run_legacy_pipeline(args, device)


if __name__ == "__main__":
    main()
