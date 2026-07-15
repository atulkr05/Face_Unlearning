import os
import json
import torch
from pathlib import Path
from ip_adapter_pipeline import load_ip_adapter_faceid_pipeline
from evaluate import run_full_evaluation, load_arcface_model

def evaluate_baseline(split_name, splits_file, emb_dir, ip_adapter_path, out_dir):
    device = "cuda:0"
    
    with open(splits_file) as f:
        splits = json.load(f)
    split_info = next((s for s in splits["splits"]
                       if s["set"] == split_name and s["track"] == "face"), None)
    
    forget_id = split_info["forget_id"]
    retain_ids = split_info["retain_ids"]
    
    print(f"\n[Baseline Evaluation] {split_name}")
    print(f"Forget: {forget_id}, Retain: {retain_ids}")
    
    pipe = load_ip_adapter_faceid_pipeline(
        base_model_id="runwayml/stable-diffusion-v1-5",
        ip_adapter_path=ip_adapter_path,
        device=device,
        dtype=torch.float16,
    )
    
    arcface_app = load_arcface_model()
    
    out_path = Path(out_dir) / split_name.replace(" ", "_")
    
    results = run_full_evaluation(
        pipe=pipe,
        concept_eraser=None,
        arcface_app=arcface_app,
        forget_id=forget_id,
        retain_ids=retain_ids,
        embedding_cache_dir=emb_dir,
        output_dir=out_path,
        n_samples_per_id=30,
        device=device,
    )
    
    print(f"\nBaseline Results for {split_name}:")
    print(f"FA: {results['FA']:.4f}")
    print(f"RA: {results['RA']:.4f}")
    return results

if __name__ == "__main__":
    import os
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.8/targets/x86_64-linux/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    
    splits_file = "/DATA2/Atul/2027/challenge/face_unlearning/validation-splits.json"
    emb_dir = "/DATA2/Atul/2027/challenge/face_unlearning/embeddings"
    ip_adapter_path = "/DATA2/Atul/2027/challenge/face_unlearning/checkpoints/ip_adapter_faceid/ip-adapter-faceid_sd15.bin"
    out_dir = "/DATA2/Atul/2027/challenge/face_unlearning/baseline_eval"
    
    evaluate_baseline("Face Set 3", splits_file, emb_dir, ip_adapter_path, out_dir)
    evaluate_baseline("Face Set 3", splits_file, emb_dir, ip_adapter_path, out_dir)
