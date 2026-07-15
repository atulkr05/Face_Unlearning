import os
import glob
import numpy as np
import torch
from PIL import Image
from evaluate import load_arcface_model, extract_embedding
from collections import defaultdict

def verify_diversity(eval_dir):
    device = "cuda:0"
    arcface_app = load_arcface_model()
    
    print(f"Verifying diversity for images in {eval_dir}...")
    
    # Load all retain images
    img_paths = glob.glob(os.path.join(eval_dir, "retain", "*.png"))
    
    embs_by_id = defaultdict(list)
    for p in img_paths:
        filename = os.path.basename(p)
        identity = filename.split("_")[0]
        
        img = Image.open(p).convert("RGB")
        emb = extract_embedding(arcface_app, np.array(img))
        if emb is not None:
            embs_by_id[identity].append(emb)
            
    # Calculate within-identity variance (mean cosine similarity to centroid)
    centroids = {}
    print("\n--- Within-Identity Similarity (Higher means tighter cluster) ---")
    for rid, embs in embs_by_id.items():
        embs_tensor = torch.tensor(np.array(embs))
        centroid = embs_tensor.mean(dim=0, keepdim=True)
        centroids[rid] = centroid
        
        # Cosine sim to centroid
        sims = torch.nn.functional.cosine_similarity(embs_tensor, centroid)
        print(f"ID {rid} (N={len(embs)}): Mean Sim to Centroid = {sims.mean().item():.4f} (Var = {sims.var().item():.4f})")
        
    print("\n--- Between-Identity Distance (Cosine Similarity between Centroids) ---")
    rids = list(centroids.keys())
    for i in range(len(rids)):
        for j in range(i+1, len(rids)):
            id1 = rids[i]
            id2 = rids[j]
            sim = torch.nn.functional.cosine_similarity(centroids[id1], centroids[id2]).item()
            print(f"Centroid Sim {id1} vs {id2}: {sim:.4f} (Lower means more distinct)")
            
    # As a baseline, what is the distance between the TRUE CelebA embeddings for these identities?
    print("\n--- Ground Truth CelebA Distances ---")
    from precompute_embeddings import load_mean_embedding
    emb_dir = "/DATA2/Atul/2027/challenge/face_unlearning/embeddings"
    true_centroids = {}
    for rid in rids:
        m = load_mean_embedding(emb_dir, rid)
        if m is not None:
            true_centroids[rid] = torch.tensor(m).unsqueeze(0)
            
    for i in range(len(rids)):
        for j in range(i+1, len(rids)):
            id1 = rids[i]
            id2 = rids[j]
            sim = torch.nn.functional.cosine_similarity(true_centroids[id1], true_centroids[id2]).item()
            print(f"True CelebA Sim {id1} vs {id2}: {sim:.4f}")

if __name__ == "__main__":
    import os
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.8/targets/x86_64-linux/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    verify_diversity("/DATA2/Atul/2027/challenge/face_unlearning/checkpoints/Face_Set_2/evaluation")
