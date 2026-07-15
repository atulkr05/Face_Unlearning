import torch
from ip_adapter_pipeline import load_ip_adapter_faceid_pipeline, load_ip_lora_adapter

device = "cuda:0"
pipe = load_ip_adapter_faceid_pipeline(
    base_model_id="runwayml/stable-diffusion-v1-5",
    ip_adapter_path="/DATA2/Atul/2027/challenge/face_unlearning/checkpoints/ip_adapter_faceid/ip-adapter-faceid_sd15.bin",
    device=device,
    dtype=torch.float16
)
print("Pipeline UNet dtype:", pipe.unet.dtype)
print("Face proj dtype before load:", pipe.face_proj_model.proj[0].weight.dtype)

pipe = load_ip_lora_adapter(pipe, "/DATA2/Atul/2027/challenge/face_unlearning/checkpoints/Face_Set_2/lora_adapter/", device)

print("Face proj dtype after load:", pipe.face_proj_model.proj[0].weight.dtype)

emb = torch.randn(1, 512, device=device, dtype=torch.float16)
try:
    tokens = pipe.face_proj_model(emb)
    print("Success! Tokens shape:", tokens.shape)
except Exception as e:
    print("Failed!")
    print(e)
