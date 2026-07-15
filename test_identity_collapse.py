import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from ip_adapter_pipeline import load_ip_adapter_faceid_pipeline, IPAttnProcessor, load_ip_lora_adapter
from concept_eraser import ConceptEraser
from precompute_embeddings import load_mean_embedding

def generate_case(pipe, face_proj, eraser, emb, prompt, num_samples, apply_eraser=False):
    device = pipe.device
    
    emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
    if apply_eraser and eraser is not None:
        emb_t = eraser.transform(emb_t.unsqueeze(0)).squeeze(0)
        
    with torch.no_grad():
        ip_tokens = face_proj(emb_t.unsqueeze(0).half())
        
    images = pipe(
        prompt=[prompt] * num_samples,
        ip_adapter_face_emb=emb_t.unsqueeze(0), # Note: the pipeline internally does not use this if we override processors
        num_inference_steps=30,
        guidance_scale=7.5,
        cross_attention_kwargs={"ip_hidden_states": ip_tokens} # We must pass ip_tokens manually if we handle projection outside
    ).images
    return images

def main():
    device = "cuda"
    out_dir = "/DATA2/Atul/2027/challenge/face_unlearning/experiments/identity_collapse"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load base pipeline
    print("Loading base pipeline...")
    pipe_base = load_ip_adapter_faceid_pipeline(
        base_model_id="runwayml/stable-diffusion-v1-5",
        ip_adapter_path="/DATA2/Atul/2027/challenge/face_unlearning/checkpoints/ip_adapter_faceid/ip-adapter-faceid_sd15.bin",
        device=device,
        dtype=torch.float16
    )
    face_proj = pipe_base.face_proj_model
    
    # Load LoRA pipeline
    print("Loading LoRA pipeline...")
    pipe_lora = load_ip_adapter_faceid_pipeline(
        base_model_id="runwayml/stable-diffusion-v1-5",
        ip_adapter_path="/DATA2/Atul/2027/challenge/face_unlearning/checkpoints/ip_adapter_faceid/ip-adapter-faceid_sd15.bin",
        device=device,
        dtype=torch.float16
    )
    pipe_lora = load_ip_lora_adapter(pipe_lora, "/DATA2/Atul/2027/challenge/face_unlearning/checkpoints/Face_Set_2/lora_adapter/")
    
    # Load Eraser
    print("Loading Eraser...")
    eraser = ConceptEraser.load("/DATA2/Atul/2027/challenge/face_unlearning/checkpoints/Face_Set_2/concept_eraser.pkl")
    eraser.to_device(device)
    
    # Embeddings
    print("Loading embeddings...")
    emb_cache = "/DATA2/Atul/2027/challenge/face_unlearning/embeddings"
    forget_emb = load_mean_embedding(emb_cache, "3376")
    retain_emb = load_mean_embedding(emb_cache, "608")
    
    cases = [
        ("Case1_Original", pipe_base, False),
        ("Case2_EraserOnly", pipe_base, True),
        ("Case3_LoRAOnly", pipe_lora, False),
        ("Case4_LoRA_Eraser", pipe_lora, True)
    ]
    
    prompt = "A photo of a person, high quality"
    
    for case_name, pipe, apply_eraser in cases:
        print(f"Generating {case_name}...")
        # Forget identity
        print("  Forget ID 3376")
        imgs_f = generate_case(pipe, face_proj, eraser, forget_emb, prompt, 2, apply_eraser=apply_eraser)
        for i, img in enumerate(imgs_f):
            img.save(os.path.join(out_dir, f"{case_name}_forget_3376_{i}.png"))
            
        # Retain identity (Eraser is theoretically only applied to forget in evaluation, 
        # but let's test what happens if we don't apply it to retain as in evaluate.py)
        print("  Retain ID 608")
        imgs_r = generate_case(pipe, face_proj, eraser, retain_emb, prompt, 2, apply_eraser=False)
        for i, img in enumerate(imgs_r):
            img.save(os.path.join(out_dir, f"{case_name}_retain_608_{i}.png"))

if __name__ == "__main__":
    main()
