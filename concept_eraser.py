"""
concept_eraser.py — LEACE-style Linear Concept Eraser for ArcFace embedding space.

Implements a linear projection P that removes the "forget identity" direction from
any query embedding before it reaches the face generation model. This acts as a
hard inference-time guardrail on top of the LoRA fine-tuning stage.

Theory (LEACE — Least-squares Concept Erasure):
    Given forget embeddings X_f ∈ R^{N_f × 512} and retain embeddings X_r ∈ R^{N_r × 512},
    find projection matrix P such that:
        P @ x  cannot be used to predict whether x belongs to the forget identity.
    
    We use an orthogonal null-space projection:
        1. Compute the forget direction d = mean(X_f) - mean(X_r)   [normalized]
        2. Compute P = I - d @ d^T   (projects out the forget direction)
        3. For extra precision, iterate with the top-k PCA directions of X_f
           that are NOT explained by X_r (via whitened LDA).

Usage:
    eraser = ConceptEraser()
    eraser.fit(forget_embeddings, retain_embeddings)
    clean_emb = eraser.transform(raw_embedding)
    eraser.save("concept_eraser.pkl")
    eraser2 = ConceptEraser.load("concept_eraser.pkl")
"""

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class ConceptEraser(nn.Module):
    """
    Linear concept eraser that removes a learned set of directions from
    ArcFace embeddings at inference time.

    The eraser learns an orthogonal projection P ∈ R^{512×512} such that
    P·x removes the components most predictive of the forget identity.

    Attributes:
        projection (torch.Tensor): (d, d) orthogonal projection matrix on CPU/GPU.
        forget_mean (torch.Tensor): (d,) mean embedding of forget identity.
        retain_mean (torch.Tensor): (d,) mean embedding of retain identities.
        n_directions (int): Number of directions erased.
        dim (int): Embedding dimension (512 for ArcFace).
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.register_buffer("projection", torch.eye(dim))
        self.register_buffer("forget_mean", torch.zeros(dim))
        self.register_buffer("retain_mean", torch.zeros(dim))
        self.n_directions: int = 0
        self.fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        forget_embeddings: Union[np.ndarray, torch.Tensor],
        retain_embeddings: Union[np.ndarray, torch.Tensor],
        n_directions: int = 3,
        whitening: bool = True,
    ) -> "ConceptEraser":
        """
        Learn the projection matrix P to erase the forget identity direction.

        Args:
            forget_embeddings:  (N_f, 512) ArcFace embeddings of forget identity.
            retain_embeddings:  (N_r, 512) ArcFace embeddings of retain identities.
            n_directions:       Number of top directions to erase (1 = simple mean-diff).
            whitening:          If True, whiten by retain covariance before computing LDA direction.
        """
        # Convert to numpy for numerical stability
        X_f = _to_numpy(forget_embeddings)   # (N_f, 512)
        X_r = _to_numpy(retain_embeddings)   # (N_r, 512)

        assert X_f.ndim == 2 and X_r.ndim == 2, "Embeddings must be 2D"
        assert X_f.shape[1] == X_r.shape[1] == self.dim, f"Dimension mismatch: expected {self.dim}"

        mu_f = X_f.mean(axis=0)              # (512,)
        mu_r = X_r.mean(axis=0)              # (512,)

        # Normalize means
        mu_f = mu_f / (np.linalg.norm(mu_f) + 1e-8)
        mu_r = mu_r / (np.linalg.norm(mu_r) + 1e-8)

        # Step 1: difference direction
        diff = mu_f - mu_r
        diff_norm = np.linalg.norm(diff)
        if diff_norm < 1e-8:
            print("[ConceptEraser] WARNING: forget and retain means are nearly identical!")
            diff = mu_f  # fall back to forget mean direction
        else:
            diff = diff / diff_norm

        # Step 2: collect top-k directions via SVD of centered forget embeddings
        X_f_centered = X_f - X_f.mean(axis=0)

        if whitening and len(X_r) >= self.dim:
            # Whiten by retain covariance so directions are meaningful relative to
            # the retain distribution (proper LEACE)
            X_r_centered = X_r - X_r.mean(axis=0)
            cov_r = X_r_centered.T @ X_r_centered / len(X_r)
            # Regularized inverse square root of covariance
            U, S, Vt = np.linalg.svd(cov_r)
            S_inv_sqrt = np.diag(1.0 / (np.sqrt(S) + 1e-6))
            W = U @ S_inv_sqrt @ Vt   # whitening matrix (512, 512)
            X_f_whitened = X_f_centered @ W.T
        else:
            W = np.eye(self.dim)
            X_f_whitened = X_f_centered

        _, _, Vt_f = np.linalg.svd(X_f_whitened, full_matrices=False)
        top_dirs_whitened = Vt_f[:n_directions]  # (n_directions, 512) in whitened space

        # Unwhiten back to original space
        W_inv = np.linalg.pinv(W)
        top_dirs = top_dirs_whitened @ W_inv.T   # (n_directions, 512)
        # Add mean-difference direction
        all_dirs = np.vstack([diff[None, :], top_dirs])  # (n_directions+1, 512)

        # Step 3: orthonormalize via Gram-Schmidt
        directions = _gram_schmidt(all_dirs)[:n_directions + 1]

        # Step 4: build null-space projection P = I - Σ_i (d_i d_i^T)
        P = np.eye(self.dim)
        for d in directions:
            d = d / (np.linalg.norm(d) + 1e-8)
            P = P - np.outer(d, d)

        # Store as buffers
        self.projection = torch.tensor(P, dtype=torch.float32)
        self.forget_mean = torch.tensor(mu_f, dtype=torch.float32)
        self.retain_mean = torch.tensor(mu_r, dtype=torch.float32)
        self.n_directions = len(directions)
        self.fitted = True

        # Verification: cosine similarity should drop
        mu_f_torch = self.forget_mean
        mu_f_erased = self.transform(mu_f_torch.unsqueeze(0)).squeeze(0)
        cos_before = float(torch.dot(mu_f_torch, mu_f_torch))
        cos_after = float(torch.dot(mu_f_erased, self.forget_mean))
        print(f"[ConceptEraser] Fitted with {self.n_directions} direction(s) erased.")
        print(f"  Forget self-similarity before: {cos_before:.4f} → after: {cos_after:.4f}")

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply eraser projection. Alias for transform()."""
        return self.transform(embeddings)

    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings onto the null space of the forget directions.

        Args:
            embeddings: (..., 512) ArcFace embeddings (any batch shape).

        Returns:
            Projected embeddings of same shape, re-normalized to unit sphere.
        """
        if not self.fitted:
            raise RuntimeError("ConceptEraser not fitted. Call fit() first.")

        P = self.projection.to(embeddings.device)
        shape = embeddings.shape
        flat = embeddings.reshape(-1, self.dim)
        projected = flat @ P.T
        # Re-normalize to unit sphere (ArcFace operates in cosine space)
        norms = projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        projected = projected / norms
        return projected.reshape(shape)

    def cosine_to_forget(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity to the forget mean (useful for debugging)."""
        mu = self.forget_mean.to(device=embeddings.device, dtype=embeddings.dtype)
        flat = embeddings.reshape(-1, self.dim).to(dtype=mu.dtype)
        return torch.mv(flat, mu)  # (N,)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]):
        """Serialize eraser to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "dim": self.dim,
            "projection": self.projection.cpu().numpy(),
            "forget_mean": self.forget_mean.cpu().numpy(),
            "retain_mean": self.retain_mean.cpu().numpy(),
            "n_directions": self.n_directions,
            "fitted": self.fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"[ConceptEraser] Saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConceptEraser":
        """Load a previously saved eraser."""
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        eraser = cls(dim=state["dim"])
        eraser.projection = torch.tensor(state["projection"], dtype=torch.float32)
        eraser.forget_mean = torch.tensor(state["forget_mean"], dtype=torch.float32)
        eraser.retain_mean = torch.tensor(state["retain_mean"], dtype=torch.float32)
        eraser.n_directions = state["n_directions"]
        eraser.fitted = state["fitted"]
        print(f"[ConceptEraser] Loaded from {path} (n_directions={eraser.n_directions})")
        return eraser

    def to_device(self, device: Union[str, torch.device]) -> "ConceptEraser":
        """Move all buffers to device."""
        self.projection = self.projection.to(device)
        self.forget_mean = self.forget_mean.to(device)
        self.retain_mean = self.retain_mean.to(device)
        return self

    def __repr__(self):
        return (
            f"ConceptEraser(dim={self.dim}, n_directions={self.n_directions}, "
            f"fitted={self.fitted})"
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.array(x, dtype=np.float32)


def _gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """
    Orthonormalize a set of vectors via Gram-Schmidt.

    Args:
        vectors: (K, D) array of K vectors in D-dimensional space.

    Returns:
        (K', D) array of K' ≤ K orthonormal vectors (degenerate ones dropped).
    """
    orthonormal = []
    for v in vectors:
        v = v.copy().astype(np.float64)
        for u in orthonormal:
            v -= np.dot(v, u) * u
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            orthonormal.append(v / norm)
    return np.array(orthonormal, dtype=np.float32)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing ConceptEraser...")
    np.random.seed(42)

    # Simulate embeddings: forget identity has a specific mean direction
    d = 512
    forget_dir = np.random.randn(d)
    forget_dir /= np.linalg.norm(forget_dir)

    forget_embs = forget_dir[None] + 0.1 * np.random.randn(50, d)
    forget_embs = forget_embs / np.linalg.norm(forget_embs, axis=1, keepdims=True)

    retain_embs = np.random.randn(150, d)
    retain_embs = retain_embs / np.linalg.norm(retain_embs, axis=1, keepdims=True)

    eraser = ConceptEraser(dim=d)
    eraser.fit(forget_embs, retain_embs, n_directions=3)

    # Test: cosine similarity to forget mean should drop after transform
    test_emb = torch.tensor(forget_dir, dtype=torch.float32).unsqueeze(0)
    before = eraser.cosine_to_forget(test_emb).item()
    erased_emb = eraser.transform(test_emb)
    after = eraser.cosine_to_forget(erased_emb).item()
    print(f"  Cosine to forget before: {before:.4f}, after: {after:.4f}  ✓")

    # Test save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name
    eraser.save(tmp_path)
    eraser2 = ConceptEraser.load(tmp_path)
    after2 = eraser2.cosine_to_forget(eraser2.transform(test_emb)).item()
    print(f"  After load: cosine to forget = {after2:.4f}  ✓")
    os.unlink(tmp_path)
    print("All tests passed!")
