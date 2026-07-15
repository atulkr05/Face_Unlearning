"""
data_loader.py — Face Identity Unlearning Data Pipeline

Supports:
  1. CelebA flat-file layout (img_celeba/XXXXXX.jpg) with identity_CelebA.txt mapping
  2. CelebA-HQ (data256x256/XXXXX.jpg) with attribute annotations
  3. Pre-computed ArcFace embedding cache for fast training
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Standard transforms
# ---------------------------------------------------------------------------
ARCFACE_TRANSFORM = T.Compose([
    T.Resize((112, 112)),          # ArcFace / InsightFace native resolution
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])

DIFFUSION_TRANSFORM = T.Compose([
    T.Resize((512, 512)),          # Arc2Face / SD 1.5 native resolution
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])

EVAL_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])


# ---------------------------------------------------------------------------
# Identity mapping from identity_CelebA.txt
# ---------------------------------------------------------------------------

def load_celeba_identity_map(identity_file: str) -> Dict[str, str]:
    """
    Parse identity_CelebA.txt → {filename: identity_id_str}
    Format: "000001.jpg 2880\n000002.jpg 2937\n..."
    """
    id_map = {}
    with open(identity_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                fname, identity = parts[0], parts[1]
                id_map[fname] = identity
    return id_map


def build_identity_to_files(
    identity_map: Dict[str, str],
    img_dir: str,
    target_ids: List[str],
) -> Dict[str, List[Path]]:
    """Map identity IDs → list of image Paths that exist on disk."""
    target_set = set(str(i) for i in target_ids)
    id_to_files: Dict[str, List[Path]] = {i: [] for i in target_set}

    for fname, identity in identity_map.items():
        if identity in target_set:
            fpath = Path(img_dir) / fname
            if fpath.exists():
                id_to_files[identity].append(fpath)

    return id_to_files


# ---------------------------------------------------------------------------
# Dataset: CelebA flat layout with pre-loaded identity mapping
# ---------------------------------------------------------------------------

class CelebAFlatIdentityDataset(Dataset):
    """
    Load images for specific CelebA identity IDs from the flat img_celeba/
    directory using identity_CelebA.txt.

    Args:
        img_dir:        Path to img_celeba/ (flat JPEGs: 000001.jpg …)
        identity_file:  Path to identity_CelebA.txt
        identity_ids:   List of CelebA identity IDs (strings/ints) to include
        transform:      torchvision transform to apply
        max_per_id:     Maximum images per identity (None = all)
    """

    def __init__(
        self,
        img_dir: str,
        identity_file: str,
        identity_ids: List[str],
        transform=None,
        max_per_id: Optional[int] = None,
    ):
        self.img_dir = Path(img_dir)
        self.transform = transform or DIFFUSION_TRANSFORM
        self.identity_ids = [str(i) for i in identity_ids]

        id_map = load_celeba_identity_map(identity_file)
        id_to_files = build_identity_to_files(id_map, img_dir, self.identity_ids)

        self.samples: List[Tuple[Path, str]] = []
        for identity, paths in id_to_files.items():
            if max_per_id is not None:
                paths = paths[:max_per_id]
            for p in sorted(paths):
                self.samples.append((p, identity))

        if len(self.samples) == 0:
            print(f"[WARNING] No images found for IDs: {identity_ids} in {img_dir}")
        else:
            print(f"[DataLoader] Loaded {len(self.samples)} images for IDs: {identity_ids}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, identity = self.samples[idx]
        try:
            img = Image.open(fpath).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"[WARNING] Failed to load {fpath}: {e}")
            img = torch.zeros(3, 512, 512)
        return {"image": img, "identity": identity, "path": str(fpath)}


# ---------------------------------------------------------------------------
# Dataset: CelebA-HQ 256×256 layout
# ---------------------------------------------------------------------------

class CelebAHQDataset(Dataset):
    """
    Load CelebA-HQ images from data256x256/ (or other HQ resolution dirs).
    Uses the same identity_CelebA.txt mapping (HQ images have the same
    1-based index as CelebA: HQ image 00001.jpg corresponds to CelebA 000001.jpg).

    Args:
        hq_dir:         Path to data256x256/ (images: 00001.jpg …)
        identity_file:  Path to identity_CelebA.txt
        identity_ids:   Identity IDs to include
        transform:      Transform to apply
        hq_index_file:  Optional path to CelebA-HQ index file (maps HQ idx → CelebA idx)
    """

    def __init__(
        self,
        hq_dir: str,
        identity_file: str,
        identity_ids: List[str],
        transform=None,
        hq_index_file: Optional[str] = None,
    ):
        self.hq_dir = Path(hq_dir)
        self.transform = transform or EVAL_TRANSFORM
        self.identity_ids = [str(i) for i in identity_ids]

        # CelebA-HQ: 00001.jpg … 30000.jpg corresponds to celeba indices 1…30000
        # Map HQ idx (1-based) → celeba filename (000001.jpg format)
        id_map = load_celeba_identity_map(identity_file)  # {000001.jpg: "2880", …}

        self.samples: List[Tuple[Path, str]] = []
        target_set = set(self.identity_ids)

        # Iterate over HQ directory
        for hq_path in sorted(self.hq_dir.glob("*.jpg")):
            hq_idx = int(hq_path.stem)  # e.g., 00001 → 1
            celeba_fname = f"{hq_idx:06d}.jpg"
            identity = id_map.get(celeba_fname)
            if identity and identity in target_set:
                self.samples.append((hq_path, identity))

        print(f"[CelebAHQ] Loaded {len(self.samples)} HQ images for IDs: {identity_ids}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, identity = self.samples[idx]
        try:
            img = Image.open(fpath).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"[WARNING] Failed to load {fpath}: {e}")
            img = torch.zeros(3, 256, 256)
        return {"image": img, "identity": identity, "path": str(fpath)}


# ---------------------------------------------------------------------------
# Dataset backed by pre-computed ArcFace embeddings
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """
    Dataset backed by pre-computed ArcFace embeddings (numpy .npy files).
    Used for fast embedding-space operations during unlearning.

    Expected cache structure:
        cache_dir/{identity_id}.npy  → (N, 512) float32 array
    """

    def __init__(self, cache_dir: str, identity_ids: List[str]):
        self.cache_dir = Path(cache_dir)
        self.identity_ids = [str(i) for i in identity_ids]
        self.samples: List[Tuple[np.ndarray, str]] = []

        for identity in self.identity_ids:
            npy_path = self.cache_dir / f"{identity}.npy"
            if npy_path.exists():
                embs = np.load(str(npy_path))  # (N, 512)
                for emb in embs:
                    self.samples.append((emb, identity))
            else:
                print(f"[WARNING] No embedding cache for identity {identity} at {npy_path}")

        print(f"[EmbeddingDataset] {len(self.samples)} embeddings for IDs: {identity_ids}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emb, identity = self.samples[idx]
        return {
            "embedding": torch.tensor(emb, dtype=torch.float32),
            "identity": identity,
        }


# ---------------------------------------------------------------------------
# Convenience factory: get DataLoaders for a validation split
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_dir: str,
    splits_file: str,
    split_name: str,
    batch_size: int = 4,
):
    """
    Backward-compatible factory (used by legacy main.py).
    Returns (target_loader, retain_loader) using FaceIdentityDataset.
    """
    target_dataset = FaceIdentityDataset(
        data_dir, splits_file, split_name, is_target=True
    )
    retain_dataset = FaceIdentityDataset(
        data_dir, splits_file, split_name, is_target=False
    )
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
    return target_loader, retain_loader


def get_celeba_dataloaders(
    img_dir: str,
    identity_file: str,
    forget_id: str,
    retain_ids: List[str],
    batch_size: int = 4,
    transform=None,
    max_per_id: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build target and retain DataLoaders from flat CelebA img_celeba directory.

    Args:
        img_dir:        Path to img_celeba/
        identity_file:  Path to identity_CelebA.txt
        forget_id:      CelebA identity ID to forget (string)
        retain_ids:     List of CelebA identity IDs to retain
        batch_size:     Batch size
        transform:      Optional torchvision transform
        max_per_id:     Cap images per identity

    Returns:
        (target_loader, retain_loader)
    """
    target_ds = CelebAFlatIdentityDataset(
        img_dir, identity_file, [str(forget_id)],
        transform=transform, max_per_id=max_per_id
    )
    retain_ds = CelebAFlatIdentityDataset(
        img_dir, identity_file, [str(i) for i in retain_ids],
        transform=transform, max_per_id=max_per_id
    )

    target_loader = DataLoader(
        target_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False
    )
    retain_loader = DataLoader(
        retain_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False
    )
    return target_loader, retain_loader


def get_embedding_dataloaders(
    cache_dir: str,
    forget_id: str,
    retain_ids: List[str],
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders backed by pre-computed ArcFace embedding .npy files.
    Much faster than loading raw images during training.
    """
    target_ds = EmbeddingDataset(cache_dir, [str(forget_id)])
    retain_ds = EmbeddingDataset(cache_dir, [str(i) for i in retain_ids])

    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=True)
    retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True)
    return target_loader, retain_loader


# ---------------------------------------------------------------------------
# Original FaceIdentityDataset (backward-compatible, kept for legacy main.py)
# ---------------------------------------------------------------------------

class FaceIdentityDataset(Dataset):
    """
    Legacy dataset that assumes data_dir/{identity_id}/*.jpg directory structure.
    Kept for backward compatibility with the original main.py.
    """

    def __init__(
        self,
        data_dir: str,
        splits_file: str,
        split_name: str = "Face Set 1",
        transform=None,
        is_target: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])

        with open(splits_file, "r") as f:
            data = json.load(f)

        self.split_info = None
        for split in data["splits"]:
            if split["set"] == split_name and split["track"] == "face":
                self.split_info = split
                break

        if not self.split_info:
            raise ValueError(f"Split {split_name} for face track not found in {splits_file}")

        self.target_id = self.split_info["forget_id"]
        self.retain_ids = self.split_info["retain_ids"]
        self.image_paths: List[Path] = []
        self.identities: List[str] = []
        self._load_image_paths(is_target)

    def _load_image_paths(self, is_target: bool):
        ids_to_load = [self.target_id] if is_target else self.retain_ids
        for identity in ids_to_load:
            id_dir = self.data_dir / str(identity)
            if id_dir.exists():
                for img_path in sorted(id_dir.glob("*.jpg")):
                    self.image_paths.append(img_path)
                    self.identities.append(identity)
            else:
                print(f"[WARNING] Directory for identity {identity} not found at {id_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        identity = self.identities[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"[ERROR] Loading {img_path}: {e}")
            image = torch.zeros((3, 256, 256))
        return {"image": image, "identity": identity, "path": str(img_path)}


# ---------------------------------------------------------------------------
# Utility: load CelebA attribute dict for given images
# ---------------------------------------------------------------------------

def load_attribute_labels(
    attr_file: str,
    image_names: List[str],
) -> Dict[str, Dict[str, int]]:
    """
    Parse list_attr_celeba_hq.txt and return {filename: {attr_name: +1/-1}} 
    for the given image filenames.
    """
    result: Dict[str, Dict[str, int]] = {}
    target_set = set(image_names)

    with open(attr_file, "r") as f:
        header = f.readline().strip().split()  # attribute names
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            fname = parts[0]
            if fname in target_set or not target_set:
                vals = {attr: int(v) for attr, v in zip(header, parts[1:])}
                result[fname] = vals

    return result


if __name__ == "__main__":
    # Quick smoke test
    splits_path = "validation-splits.json"
    img_dir = "../CelebAHQ/Img/img_celeba"
    identity_file = "../CelebAHQ/Anno/identity_CelebA.txt"

    if os.path.exists(identity_file):
        print("Testing CelebAFlatIdentityDataset...")
        t_loader, r_loader = get_celeba_dataloaders(
            img_dir, identity_file,
            forget_id="3422", retain_ids=["5230", "5239", "1539"],
            batch_size=4, max_per_id=20,
        )
        print(f"  Target batches: {len(t_loader)}, Retain batches: {len(r_loader)}")
        batch = next(iter(t_loader))
        print(f"  Batch keys: {list(batch.keys())}, image shape: {batch['image'].shape}")
    else:
        print(f"identity_CelebA.txt not found at {identity_file} — run download first.")
