import os
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline

# Create directories
models_dir = Path("D:/latent_diffusion_research/models")
sd_dir = models_dir / "sd_checkpoints"
controlnet_dir = models_dir / "controlnet"
os.makedirs(sd_dir, exist_ok=True)
os.makedirs(controlnet_dir, exist_ok=True)

# Download SD 1.5
print("Downloading SD 1.5...")
StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                        torch_dtype=torch.float16).save_pretrained(sd_dir / "sd-v1-5")

# If you want to download ControlNet models
print("Downloading ControlNet models...")
# Canny model
hf_hub_download(repo_id="lllyasviel/ControlNet-v1-1", 
                filename="control_v11p_sd15_canny.pth", 
                local_dir=str(controlnet_dir))

# Depth model
hf_hub_download(repo_id="lllyasviel/ControlNet-v1-1", 
                filename="control_v11f1p_sd15_depth.pth", 
                local_dir=str(controlnet_dir))

# Pose model
hf_hub_download(repo_id="lllyasviel/ControlNet-v1-1", 
                filename="control_v11p_sd15_openpose.pth", 
                local_dir=str(controlnet_dir))

print("Download complete!")