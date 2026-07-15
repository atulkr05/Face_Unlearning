"""
run_unlearning.py — Complete IP-Adapter-FaceID unlearning pipeline.

This is the production script that runs the full 3-stage unlearning:
  Stage 1: ConceptEraser (embedding-space nullification)
  Stage 2: LoRA fine-tuning on IP cross-attention K/V projections
  Stage 3: ConceptEraser applied at inference (hard guardrail)

Usage:
    # Face Set 1 (forget 3422)
    python face_unlearning/run_unlearning.py --split_name "Face Set 1" --gpu 0

    # Face Set 2 (forget 3376)
    python face_unlearning/run_unlearning.py --split_name "Face Set 2" --gpu 1

    # Both in parallel:
    python face_unlearning/run_unlearning.py --split_name "Face Set 1" --gpu 0 &
    python face_unlearning/run_unlearning.py --split_name "Face Set 2" --gpu 1 &
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Loss functions for IP-Adapter unlearning
# ---------------------------------------------------------------------------

def compute_ip_attribute_preservation_loss(
    unet: nn.Module,
    orig_unet: nn.Module,
    vae,
    noise_scheduler,
    images: torch.Tensor,
    ip_tokens_forget: torch.Tensor,
    ip_tokens_forget_erased: torch.Tensor,
    text_emb: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Compute behavioral matching loss for forget identity with attribute preservation.
    The current model (with forget identity conditioning) should produce the same
    noise predictions as the original frozen model with erased identity conditioning.
    """
    from ip_adapter_pipeline import IPAttnProcessor

    B = images.shape[0]

    with torch.no_grad():
        latents = vae.encode(images.to(dtype=vae.dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(100, 900, (B,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Original model prediction (frozen) with erased concept
    with torch.no_grad():
        for proc in orig_unet.attn_processors.values():
            if isinstance(proc, IPAttnProcessor):
                proc._ip_hidden_states = ip_tokens_forget_erased.expand(B, -1, -1)
        noise_pred_orig = orig_unet(
            noisy_latents, timesteps,
            encoder_hidden_states=text_emb.expand(B, -1, -1),
        ).sample

    # Current model prediction with forget concept
    for proc in unet.attn_processors.values():
        if isinstance(proc, IPAttnProcessor):
            proc._ip_hidden_states = ip_tokens_forget.expand(B, -1, -1)
    noise_pred_curr = unet(
        noisy_latents, timesteps,
        encoder_hidden_states=text_emb.expand(B, -1, -1),
    ).sample

    return F.mse_loss(noise_pred_curr.float(), noise_pred_orig.detach().float())


def compute_ip_retain_loss(
    unet: nn.Module,
    orig_unet: nn.Module,
    vae,
    noise_scheduler,
    images: torch.Tensor,
    ip_tokens_retain: torch.Tensor,
    text_emb: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Compute behavioral matching loss: current model should produce the same
    noise predictions as the original frozen model for retain identities.
    """
    from ip_adapter_pipeline import IPAttnProcessor

    B = images.shape[0]

    with torch.no_grad():
        latents = vae.encode(images.to(dtype=vae.dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(100, 900, (B,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Original model prediction (frozen)
    with torch.no_grad():
        for proc in orig_unet.attn_processors.values():
            if isinstance(proc, IPAttnProcessor):
                proc._ip_hidden_states = ip_tokens_retain.expand(B, -1, -1)
        noise_pred_orig = orig_unet(
            noisy_latents, timesteps,
            encoder_hidden_states=text_emb.expand(B, -1, -1),
        ).sample

    # Current model prediction
    for proc in unet.attn_processors.values():
        if isinstance(proc, IPAttnProcessor):
            proc._ip_hidden_states = ip_tokens_retain.expand(B, -1, -1)
    noise_pred_curr = unet(
        noisy_latents, timesteps,
        encoder_hidden_states=text_emb.expand(B, -1, -1),
    ).sample

    return F.mse_loss(noise_pred_curr.float(), noise_pred_orig.detach().float())


def compute_ip_ewc_loss(
    unet: nn.Module,
    ewc_reference: Dict[str, torch.Tensor],
    device: str,
) -> torch.Tensor:
    """
    EWC regularization: penalize deviation of LoRA parameters from their
    initial values (which correspond to the original IP-Adapter behavior).
    """
    ewc_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    for name, param in unet.named_parameters():
        if param.requires_grad and name in ewc_reference:
            ref = ewc_reference[name].to(device)
            ewc_loss += F.mse_loss(param.float(), ref.float())
    return ewc_loss


# ---------------------------------------------------------------------------
# Main unlearning loop for IP-Adapter-FaceID
# ---------------------------------------------------------------------------

def train_ip_adapter_unlearning(
    pipe,
    target_loader,
    retain_loader,
    concept_eraser,
    forget_id: str,
    retain_ids: List[str],
    embedding_cache_dir: str,
    device: str = "cuda",
    num_epochs: int = 20,
    lr: float = 5e-5,
    alpha: float = 1.0,
    beta: float = 2.0,
    delta: float = 0.01,
    checkpoint_dir: Optional[str] = None,
    log_every: int = 5,
) -> object:
    """
    Fine-tune IP-Adapter LoRA adapters to unlearn a specific face identity.

    Three loss terms:
      L = α·MSE(forget vs. erased_original) + β·MSE(retain vs. original) + δ·EWC(lora_weights)

    Args:
        pipe:               SD1.5 + IP-Adapter-FaceID pipeline with LoRA injected.
        target_loader:      DataLoader for forget-identity images.
        retain_loader:      DataLoader for retain-identity images.
        concept_eraser:     Pre-fitted ConceptEraser (applied to forget embedding).
        forget_id:          CelebA forget identity ID.
        retain_ids:         List of retain identity IDs.
        embedding_cache_dir: Path to pre-computed .npy embedding caches.
        device:             Compute device.
        num_epochs:         Training epochs.
        lr:                 Learning rate.
        alpha/beta/delta:   Loss weights.
        checkpoint_dir:     Save directory.
        log_every:          Logging frequency.
    """
    from precompute_embeddings import load_mean_embedding
    from ip_adapter_pipeline import IPAttnProcessor, save_ip_lora_adapter

    unet = pipe.unet.to(device)
    vae = pipe.vae.to(device)
    noise_scheduler = pipe.scheduler
    face_proj = pipe.face_proj_model.to(device)

    vae.requires_grad_(False)
    vae.eval()
    face_proj.eval()

    # Frozen reference model
    orig_unet = deepcopy(unet)
    orig_unet.requires_grad_(False)
    orig_unet.eval()

    # Load pre-computed mean embeddings
    forget_mean = load_mean_embedding(embedding_cache_dir, forget_id)
    retain_means = {}
    for rid in retain_ids:
        m = load_mean_embedding(embedding_cache_dir, rid)
        if m is not None:
            retain_means[rid] = torch.tensor(m, dtype=torch.float32).to(device)

    if forget_mean is None:
        raise RuntimeError(f"No embedding cache for forget ID {forget_id}")

    forget_mean_t = torch.tensor(forget_mean, dtype=torch.float32).to(device)

    # Apply ConceptEraser to forget embedding (Stage 1 → Stage 2 handoff)
    if concept_eraser is not None and concept_eraser.fitted:
        forget_erased = concept_eraser.transform(forget_mean_t.unsqueeze(0)).squeeze(0)
    else:
        forget_erased = forget_mean_t

    # We will compute IP tokens dynamically during training so face_proj receives gradients.
    with torch.no_grad():
        # Precompute the erased embedding target so it stays fixed
        ip_tokens_forget_erased_target = face_proj(forget_erased.unsqueeze(0).to(face_proj.proj[0].weight.dtype))

    # Get null text embedding (unconditional)
    with torch.no_grad():
        null_text_emb = pipe.encode_prompt(
            "",
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )[0]  # (1, 77, 768)

    # EWC reference: snapshot of initial LoRA weights and face_proj weights
    ewc_reference = {
        n: p.data.clone() for n, p in unet.named_parameters() if p.requires_grad
    }
    ewc_reference.update({
        f"face_proj.{n}": p.data.clone() for n, p in face_proj.named_parameters() if p.requires_grad
    })

    # Optimizer
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters())) + \
                       list(filter(lambda p: p.requires_grad, face_proj.parameters()))
    optimizer = AdamW(
        trainable_params,
        lr=lr, betas=(0.9, 0.999), weight_decay=1e-4,
    )
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n[Unlearning] {'='*50}")
    print(f"  Forget ID: {forget_id} | Retain IDs: {retain_ids}")
    print(f"  Forget emb cosine after erasure: "
          f"{float(F.cosine_similarity(forget_mean_t.unsqueeze(0), forget_erased.unsqueeze(0))):.4f}")
    print(f"  Loss weights: α={alpha} (forget attribute preservation), β={beta} (retain), δ={delta} (EWC)")
    print(f"  Epochs: {num_epochs}, LR: {lr}")
    print(f"{'='*54}\n")

    unet.train()
    best_total = float("inf")

    for epoch in range(num_epochs):
        retain_iter = iter(retain_loader)
        totals = {"f": 0.0, "r": 0.0, "ewc": 0.0, "total": 0.0}
        n = 0

        for step, target_batch in enumerate(target_loader):
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)

            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            target_imgs = target_batch["image"].to(device)
            retain_imgs = retain_batch["image"].to(device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                # Dynamically project embeddings so face_proj receives gradients
                # Autocast automatically handles dtype, but we explicitly cast the input to match face_proj
                ip_tokens_forget_curr = face_proj(forget_mean_t.unsqueeze(0).to(face_proj.proj[0].weight.dtype))
                
                # ── FORGET LOSS (attribute preservation with erased identity) ──
                loss_f = compute_ip_attribute_preservation_loss(
                    unet, orig_unet, vae, noise_scheduler,
                    target_imgs,
                    ip_tokens_forget_curr,
                    ip_tokens_forget_erased_target,
                    null_text_emb,
                    device,
                )
                total_loss = total_loss + alpha * loss_f   # Gradient descent (minimize MSE)

                # ── RETAIN LOSS (MSE vs original model on retain images) ─────
                # Get per-identity retain embedding
                rid = retain_batch["identity"][0]
                if rid in retain_means:
                    r_emb = retain_means[rid].unsqueeze(0)
                    ip_tok_r = face_proj(r_emb.to(face_proj.proj[0].weight.dtype))
                else:
                    if retain_means:
                        retain_mean_t = torch.stack(list(retain_means.values())).mean(0)
                        ip_tok_r = face_proj(retain_mean_t.unsqueeze(0).to(face_proj.proj[0].weight.dtype))
                    else:
                        ip_tok_r = torch.zeros(1, 4, 768, device=device, dtype=torch.float16)

                loss_r = compute_ip_retain_loss(
                    unet, orig_unet, vae, noise_scheduler,
                    retain_imgs,
                    ip_tok_r,
                    null_text_emb,
                    device,
                )
                total_loss = total_loss + beta * loss_r

                # ── EWC REGULARIZATION ───────────────────────────────────────
                loss_ewc = compute_ip_ewc_loss(unet, ewc_reference, device)
                # Add EWC for face_proj_model
                for name, param in face_proj.named_parameters():
                    if param.requires_grad and f"face_proj.{name}" in ewc_reference:
                        ref = ewc_reference[f"face_proj.{name}"].to(device)
                        loss_ewc += F.mse_loss(param.float(), ref.float())
                total_loss = total_loss + delta * loss_ewc

            # ── BACKWARD ─────────────────────────────────────────────────
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=0.5,
            )
            scaler.step(optimizer)
            scaler.update()

            totals["f"] += loss_f.item()
            totals["r"] += loss_r.item()
            totals["ewc"] += loss_ewc.item()
            totals["total"] += total_loss.item()
            n += 1

            if step % log_every == 0:
                print(
                    f"  [{epoch+1}/{num_epochs}][{step}] "
                    f"Forget: {loss_f.item():.4f} | "
                    f"Retain: {loss_r.item():.4f} | "
                    f"EWC: {loss_ewc.item():.4f} | "
                    f"Total: {total_loss.item():.4f}"
                )

        scheduler_lr.step()

        avg_total = totals["total"] / max(n, 1)
        print(f"\nEpoch {epoch+1}/{num_epochs} ─ "
              f"Forget: {totals['f']/max(n,1):.4f} | "
              f"Retain: {totals['r']/max(n,1):.4f} | "
              f"EWC: {totals['ewc']/max(n,1):.4f} | "
              f"Total: {avg_total:.4f}")

        if checkpoint_dir and avg_total < best_total:
            best_total = avg_total
            save_ip_lora_adapter(
                pipe, checkpoint_dir,
                metadata={"epoch": epoch+1, "forget_id": forget_id, "retain_ids": retain_ids}
            )
            print(f"  → Best checkpoint saved (total={best_total:.4f})")

    print("\n[Unlearning] Training complete!")
    if checkpoint_dir:
        save_ip_lora_adapter(
            pipe, checkpoint_dir,
            metadata={"epoch": num_epochs, "forget_id": forget_id, "final": True}
        )
    return pipe


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="IP-Adapter-FaceID Identity Unlearning")
    p.add_argument("--split_name", type=str, default="Face Set 1")
    p.add_argument("--splits_file", type=str,
                   default="validation-splits.json")
    p.add_argument("--img_dir", type=str,
                   default="../CelebAHQ/Img/img_celeba")
    p.add_argument("--identity_file", type=str,
                   default="../CelebAHQ/Anno/identity_CelebA.txt")
    p.add_argument("--embedding_cache", type=str,
                   default="embeddings")
    p.add_argument("--ip_adapter_path", type=str,
                   default="checkpoints/ip_adapter_faceid/ip-adapter-faceid_sd15.bin")
    p.add_argument("--checkpoint_dir", type=str,
                   default="checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=5.0)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--n_eraser_directions", type=int, default=3)
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    # Load split info
    with open(args.splits_file) as f:
        splits = json.load(f)
    split_info = next((s for s in splits["splits"]
                       if s["set"] == args.split_name and s["track"] == "face"), None)
    if not split_info:
        raise ValueError(f"Split '{args.split_name}' not found")

    forget_id = split_info["forget_id"]
    retain_ids = split_info["retain_ids"]
    all_ids = [forget_id] + retain_ids
    ckpt_dir = Path(args.checkpoint_dir) / args.split_name.replace(" ", "_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eraser_path = ckpt_dir / "concept_eraser.pkl"
    lora_ckpt = ckpt_dir / "lora_adapter"

    print(f"\n{'='*60}")
    print(f"  IP-Adapter-FaceID Unlearning — {args.split_name}")
    print(f"  Forget: {forget_id}  |  Retain: {retain_ids}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    from precompute_embeddings import precompute_embeddings_for_ids, load_mean_embedding
    from concept_eraser import ConceptEraser
    from ip_adapter_pipeline import (
        load_ip_adapter_faceid_pipeline, inject_lora_into_ip_adapter
    )
    from data_loader import get_celeba_dataloaders, DIFFUSION_TRANSFORM

    # Step 1: Precompute embeddings
    print("[Step 1] Loading embeddings...")
    embeddings = precompute_embeddings_for_ids(
        img_dir=args.img_dir,
        identity_file=args.identity_file,
        identity_ids=all_ids,
        output_dir=args.embedding_cache,
        gpu_id=args.gpu,
    )

    forget_embs = embeddings.get(forget_id, np.zeros((0, 512)))
    retain_embs_all = np.concatenate(
        [embeddings[r] for r in retain_ids if len(embeddings.get(r, [])) > 0],
        axis=0
    ) if any(len(embeddings.get(r, [])) > 0 for r in retain_ids) else np.ones((1, 512)) / 512**0.5

    print(f"  Forget ID {forget_id}: {len(forget_embs)} embeddings")
    for rid in retain_ids:
        print(f"  Retain ID {rid}: {len(embeddings.get(rid, []))} embeddings")

    # Step 2: Fit ConceptEraser
    print("\n[Step 2] Fitting ConceptEraser...")
    if not eraser_path.exists():
        eraser = ConceptEraser(dim=512)
        if len(forget_embs) > 0 and len(retain_embs_all) > 0:
            eraser.fit(forget_embs, retain_embs_all, n_directions=args.n_eraser_directions)
        eraser.save(eraser_path)
    else:
        eraser = ConceptEraser.load(eraser_path)

    if args.eval_only:
        print("[eval_only] Skipping training.")
    else:
        # Step 3: Load pipeline
        print("\n[Step 3] Loading IP-Adapter-FaceID pipeline...")
        pipe = load_ip_adapter_faceid_pipeline(
            ip_adapter_path=args.ip_adapter_path,
            device=device,
            dtype=torch.float16,
        )
        pipe = inject_lora_into_ip_adapter(pipe, rank=args.lora_rank, alpha=args.lora_rank)

        # Step 4: Build data loaders
        print("\n[Step 4] Building data loaders...")
        t_loader, r_loader = get_celeba_dataloaders(
            img_dir=args.img_dir,
            identity_file=args.identity_file,
            forget_id=forget_id,
            retain_ids=retain_ids,
            batch_size=args.batch_size,
            transform=DIFFUSION_TRANSFORM,
        )

        # Step 5: Train unlearning
        print(f"\n[Step 5] Training ({args.epochs} epochs)...")
        pipe = train_ip_adapter_unlearning(
            pipe=pipe,
            target_loader=t_loader,
            retain_loader=r_loader,
            concept_eraser=eraser,
            forget_id=forget_id,
            retain_ids=retain_ids,
            embedding_cache_dir=args.embedding_cache,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            alpha=args.alpha,
            beta=args.beta,
            delta=args.delta,
            checkpoint_dir=str(lora_ckpt),
            log_every=args.log_every,
        )

    # ── 4. Final Evaluation ──────────────────────────────────────────────────
    print("\n[Unlearning] Running final ArcFace evaluation...")
    
    # Cast face_proj_model back to pipeline's dtype for inference
    pipe.face_proj_model.to(dtype=pipe.unet.dtype)
    
    from evaluate import run_full_evaluation, load_arcface_model
    import os
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.8/targets/x86_64-linux/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    arcface_app = load_arcface_model(model_name="buffalo_l", gpu_id=args.gpu)
    output_dir = ckpt_dir / "evaluation"
    
    # If we skipped training but want to evaluate, load the pipeline and LoRA weights
    if args.eval_only:
        print("\nLoading pipeline and LoRA for evaluation...")
        pipe = load_ip_adapter_faceid_pipeline(
            ip_adapter_path=args.ip_adapter_path,
            device=device,
            dtype=torch.float16,
        )
        from ip_adapter_pipeline import load_ip_lora_adapter
        pipe = load_ip_lora_adapter(pipe, str(lora_ckpt))

    results = run_full_evaluation(
        pipe=pipe,
        concept_eraser=eraser,
        arcface_app=arcface_app,
        forget_id=forget_id,
        retain_ids=retain_ids,
        embedding_cache_dir=args.embedding_cache,
        output_dir=str(output_dir),
        n_samples_per_id=30,
        device=device,
    )
    
    # Save results
    with open(ckpt_dir / "results_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[Done] Checkpoints at: {ckpt_dir}")
    print(f"  ConceptEraser: {eraser_path}")
    print(f"  LoRA adapter:  {lora_ckpt}/")
    print("\nLoading instructions:")
    print("  from ip_adapter_pipeline import load_ip_adapter_faceid_pipeline, load_ip_lora_adapter")
    print("  from concept_eraser import ConceptEraser")
    print(f"  pipe = load_ip_adapter_faceid_pipeline(ip_adapter_path='...')")
    print(f"  pipe = load_ip_lora_adapter(pipe, '{lora_ckpt}/')")
    print(f"  eraser = ConceptEraser.load('{eraser_path}')")


if __name__ == "__main__":
    main()
