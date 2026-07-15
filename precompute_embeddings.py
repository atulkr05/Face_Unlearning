"""
precompute_embeddings.py — Extract and cache ArcFace embeddings for each identity.

Usage:
    python face_unlearning/precompute_embeddings.py \
        --img_dir CelebAHQ/Img/img_celeba \
        --identity_file CelebAHQ/Anno/identity_CelebA.txt \
        --identity_ids 3422 5230 5239 1539 3376 3602 608 7405 \
        --output_dir face_unlearning/embeddings/ \
        --model_name buffalo_l

Outputs:
    embeddings/{identity_id}.npy  — (N, 512) float32 ArcFace embeddings
    embeddings/{identity_id}_mean.npy — (512,) mean embedding
    embeddings/metadata.json  — per-ID stats (count, mean_norm, etc.)
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# InsightFace ArcFace extractor
# ---------------------------------------------------------------------------

class ArcFaceExtractor:
    """
    Wraps InsightFace's face analysis pipeline to extract ArcFace embeddings.
    Handles: face detection → alignment → 512-dim L2-normalized embedding.
    """

    def __init__(self, model_name: str = "buffalo_l", gpu_id: int = 0):
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface not installed. Run: pip install insightface onnxruntime-gpu"
            )

        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=[f"CUDAExecutionProvider"] if gpu_id >= 0 else ["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=gpu_id, det_size=(640, 640))
        self.model_name = model_name
        print(f"[ArcFace] Loaded model '{model_name}' on GPU {gpu_id}")

    def extract(self, img_path: str) -> Optional[np.ndarray]:
        """
        Extract ArcFace embedding from an image file.

        Returns:
            512-dim L2-normalized embedding (float32), or None if no face detected.
        """
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            # InsightFace expects BGR
            img_bgr = img[:, :, ::-1].copy()
            faces = self.app.get(img_bgr)
            if not faces:
                return None
            # Take the largest face by bounding box area
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = face.normed_embedding  # already L2-normalized, shape (512,)
            return emb.astype(np.float32)
        except Exception as e:
            print(f"[WARNING] Failed to process {img_path}: {e}")
            return None

    def extract_batch(
        self, img_paths: List[str], show_progress: bool = True
    ) -> List[Optional[np.ndarray]]:
        """Extract embeddings for a list of image paths."""
        results = []
        iterator = tqdm(img_paths, desc="Extracting embeddings") if show_progress else img_paths
        for p in iterator:
            results.append(self.extract(p))
        return results


# ---------------------------------------------------------------------------
# Main embedding precomputation function
# ---------------------------------------------------------------------------

def precompute_embeddings_for_ids(
    img_dir: str,
    identity_file: str,
    identity_ids: List[str],
    output_dir: str,
    model_name: str = "buffalo_l",
    gpu_id: int = 0,
    force_recompute: bool = False,
    max_per_id: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    For each identity ID, extract ArcFace embeddings from all its CelebA images.
    Saves .npy cache files and returns dict {identity_id: embeddings_array}.

    Args:
        img_dir:        Path to img_celeba/ (flat JPEGs)
        identity_file:  Path to identity_CelebA.txt
        identity_ids:   List of CelebA identity IDs to process
        output_dir:     Directory to store .npy cache files
        model_name:     InsightFace model ('buffalo_l' or 'antelopev2')
        gpu_id:         GPU device ID (-1 = CPU)
        force_recompute: Overwrite existing cache files
        max_per_id:     Process at most this many images per identity
    """
    from data_loader import load_celeba_identity_map, build_identity_to_files

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    identity_ids = [str(i) for i in identity_ids]

    # Load identity mapping
    print(f"[Embeddings] Loading identity map from {identity_file}...")
    id_map = load_celeba_identity_map(identity_file)
    id_to_files = build_identity_to_files(id_map, img_dir, identity_ids)

    # Initialize extractor lazily (only if any IDs need processing)
    extractor = None
    results = {}
    metadata = {}

    for identity in identity_ids:
        npy_path = output_dir / f"{identity}.npy"
        mean_path = output_dir / f"{identity}_mean.npy"

        if npy_path.exists() and not force_recompute:
            print(f"[Embeddings] Cache hit for identity {identity} at {npy_path}")
            embs = np.load(str(npy_path))
            results[identity] = embs
            metadata[identity] = {
                "count": len(embs),
                "mean_norm": float(np.linalg.norm(embs.mean(axis=0))),
                "cached": True,
            }
            continue

        # Lazy-init extractor
        if extractor is None:
            extractor = ArcFaceExtractor(model_name=model_name, gpu_id=gpu_id)

        files = id_to_files.get(identity, [])
        if not files:
            print(f"[WARNING] No images found for identity {identity}")
            results[identity] = np.zeros((0, 512), dtype=np.float32)
            continue

        if max_per_id:
            files = files[:max_per_id]

        print(f"[Embeddings] Processing identity {identity}: {len(files)} images...")
        img_paths = [str(f) for f in files]
        raw_embs = extractor.extract_batch(img_paths)

        # Filter out None (no face detected)
        valid_embs = [e for e in raw_embs if e is not None]
        failed = len(raw_embs) - len(valid_embs)
        if failed > 0:
            print(f"  [INFO] {failed}/{len(raw_embs)} images had no detectable face (skipped)")

        if len(valid_embs) == 0:
            print(f"[WARNING] No valid embeddings for identity {identity}!")
            results[identity] = np.zeros((0, 512), dtype=np.float32)
            continue

        embs = np.stack(valid_embs, axis=0).astype(np.float32)  # (N, 512)
        mean_emb = embs.mean(axis=0)
        mean_emb /= np.linalg.norm(mean_emb) + 1e-8  # re-normalize

        np.save(str(npy_path), embs)
        np.save(str(mean_path), mean_emb)

        results[identity] = embs
        metadata[identity] = {
            "count": len(embs),
            "failed": failed,
            "mean_norm": float(np.linalg.norm(mean_emb)),
            "cached": False,
        }
        print(f"  Saved {len(embs)} embeddings → {npy_path}")

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[Embeddings] Done. Metadata saved to {meta_path}")

    return results


def load_mean_embedding(cache_dir: str, identity_id: str) -> Optional[np.ndarray]:
    """Load the pre-computed mean ArcFace embedding for an identity."""
    mean_path = Path(cache_dir) / f"{identity_id}_mean.npy"
    if mean_path.exists():
        return np.load(str(mean_path)).astype(np.float32)
    # Fall back: compute from per-image cache
    npy_path = Path(cache_dir) / f"{identity_id}.npy"
    if npy_path.exists():
        embs = np.load(str(npy_path))
        if len(embs) == 0:
            return None
        mean_emb = embs.mean(axis=0)
        mean_emb /= np.linalg.norm(mean_emb) + 1e-8
        return mean_emb.astype(np.float32)
    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute ArcFace embeddings for CelebA identities")
    parser.add_argument("--img_dir", type=str,
                        default="../CelebAHQ/Img/img_celeba",
                        help="Path to flat CelebA image directory")
    parser.add_argument("--identity_file", type=str,
                        default="../CelebAHQ/Anno/identity_CelebA.txt",
                        help="Path to identity_CelebA.txt")
    parser.add_argument("--identity_ids", nargs="+", type=str,
                        default=["3422", "5230", "5239", "1539", "3376", "3602", "608", "7405"],
                        help="CelebA identity IDs to precompute embeddings for")
    parser.add_argument("--output_dir", type=str,
                        default="embeddings",
                        help="Directory to save .npy embedding caches")
    parser.add_argument("--model_name", type=str, default="buffalo_l",
                        choices=["buffalo_l", "antelopev2"],
                        help="InsightFace recognition model")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID (-1 for CPU)")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Overwrite existing cache files")
    parser.add_argument("--max_per_id", type=int, default=None,
                        help="Max images per identity (None = all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.identity_file):
        print(f"\n[ERROR] identity_CelebA.txt not found at: {args.identity_file}")
        print("Download it from: https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS")
        print("Place it at: CelebAHQ/Anno/identity_CelebA.txt")
        exit(1)

    results = precompute_embeddings_for_ids(
        img_dir=args.img_dir,
        identity_file=args.identity_file,
        identity_ids=args.identity_ids,
        output_dir=args.output_dir,
        model_name=args.model_name,
        gpu_id=args.gpu_id,
        force_recompute=args.force_recompute,
        max_per_id=args.max_per_id,
    )

    print("\n=== Summary ===")
    for iid, embs in results.items():
        print(f"  Identity {iid}: {len(embs)} embeddings")
