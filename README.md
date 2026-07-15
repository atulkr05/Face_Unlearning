# Face Identity Unlearning Challenge 2026

This repository contains our submission for the Face Identity Unlearning Challenge. Our approach selectively unlearns a "forget" identity from the **IP-Adapter-FaceID** (Stable Diffusion 1.5) model while preserving other identities. 

We achieved this by jointly fine-tuning the UNet cross-attention LoRA adapters and the `face_proj_model` (MLP) using a combination of **ConceptEraser** (to remove identity from the embedding space) and **Elastic Weight Consolidation (EWC)** (to preserve the weights critical for retain identities).

## Hardware and Runtime Requirements
- **OS**: Linux (tested on Ubuntu 22.04)
- **GPU**: NVIDIA GPU with at least 24GB VRAM (tested on NVIDIA H200 NVL)
- **RAM**: 32GB+
- **CUDA**: 12.1+

## Installation and Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/face_unlearning_challenge.git
   cd face_unlearning_challenge
   ```

2. **Set up the Python environment**
   We recommend using `conda` or `venv`.
   ```bash
   python3 -m venv face_env
   source face_env/bin/activate
   pip install -r requirements.txt
   ```

## Data Preparation

1. **CelebA-HQ Dataset**: 
   Ensure you have the CelebA-HQ dataset downloaded. Place the flat images in `CelebAHQ/Img/img_celeba` and the identity annotations in `CelebAHQ/Anno/identity_CelebA.txt`.

2. **Precompute ArcFace Embeddings**:
   Our pipeline requires precomputed InsightFace embeddings for the identities.
   ```bash
   python precompute_embeddings.py \
       --img_dir /path/to/CelebAHQ/Img/img_celeba \
       --identity_file /path/to/CelebAHQ/Anno/identity_CelebA.txt \
       --output_dir embeddings/ \
       --identity_ids 3422 5230 5239 1539 3376 3602 608 7405
   ```

## Exact Commands to Reproduce Outputs

To run the full unlearning pipeline and evaluation for the challenge validation splits, use the provided bash script. This script automatically handles fitting the ConceptEraser, fine-tuning the model, and generating the ArcFace evaluation metrics.

```bash
# Run both Face Set 1 and Face Set 2 in parallel (requires 2 GPUs)
bash run_experiment.sh "all"

# Or run them individually:
bash run_experiment.sh "Face Set 1"
bash run_experiment.sh "Face Set 2"
```

## Testing Instructions & Inference Entry Points

If you wish to load the unlearned model and test it manually (e.g., using a custom script or a Jupyter Notebook), you can load the pipeline and inject the unlearned LoRA and `face_proj_model` adapters as follows:

```python
import torch
from ip_adapter_pipeline import load_ip_adapter_faceid_pipeline, load_ip_lora_adapter
from concept_eraser import ConceptEraser

device = "cuda"

# 1. Load the base model
pipe = load_ip_adapter_faceid_pipeline(
    base_model_id="runwayml/stable-diffusion-v1-5",
    ip_adapter_path="checkpoints/ip_adapter_faceid/ip-adapter-faceid_sd15.bin",
    device=device,
    dtype=torch.float16
)

# 2. Inject the unlearned weights
pipe = load_ip_lora_adapter(pipe, "checkpoints/Face_Set_1/lora_adapter/")

# 3. Load the ConceptEraser (optional, applies embedding filtering)
eraser = ConceptEraser.load("checkpoints/Face_Set_1/concept_eraser.pkl")

# 4. Generate an image
identity_tensor = torch.randn(1, 512, device=device, dtype=torch.float32) # Replace with real ArcFace embedding
# erased_tensor = eraser.transform(identity_tensor) # Apply eraser if needed

pipe(
    prompt="A photo of a person",
    ip_adapter_face_emb=identity_tensor.unsqueeze(0),
    num_inference_steps=25
).images[0].save("test.png")
```

## Expected Output Structure

After running `run_experiment.sh`, the outputs will be organized as follows:
```text
checkpoints/
├── Face_Set_1/
│   ├── concept_eraser.pkl
│   ├── lora_adapter/
│   │   ├── face_proj_model.pt
│   │   ├── ip_lora_weights.pt
│   │   └── adapter_config.json
│   └── evaluation/
│       ├── results.json
│       ├── forget/ (Generated images for forget ID)
│       └── retain/ (Generated images for retain IDs)
└── Face_Set_2/
    └── ...
```

## Troubleshooting Notes

*   **`RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float`**: This occurs if the `face_proj_model` is not correctly cast back to `float16` after loading the unlearned weights. Ensure you call `pipe.face_proj_model.to(torch.float16)` before running inference.
*   **`Failed to load library libonnxruntime_providers_cuda.so`**: This is a harmless warning from InsightFace indicating it could not find cuDNN for the CUDA Execution Provider and is falling back to CPU. It does not affect the unlearning or Stable Diffusion generation quality, it only slightly slows down embedding extraction.
*   **Identity Collapse**: If the model generates generic "prototype" faces for the retain identities, ensure that `face_proj_model` is unfrozen during training and that the `beta` (retain loss weight) in `run_experiment.sh` is set to `5.0`.
