import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = ROOT_DIR / "models"
SD_CHECKPOINTS_DIR = MODELS_DIR / "sd_checkpoints"
CONTROLNET_DIR = MODELS_DIR / "controlnet"

# Results directories
RESULTS_DIR = ROOT_DIR / "results"
IMAGES_DIR = RESULTS_DIR / "images"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SD_CHECKPOINTS_DIR, 
                   CONTROLNET_DIR, IMAGES_DIR, METRICS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    "sd_1_5": {
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "local_path": SD_CHECKPOINTS_DIR / "sd-v1-5"
    },
    "controlnet": {
        "canny": {
            "filename": "control_v11p_sd15_canny.pth",
            "repo_id": "lllyasviel/ControlNet-v1-1",
            "local_path": CONTROLNET_DIR / "control_v11p_sd15_canny.pth",
            "diffusers_path": CONTROLNET_DIR / "canny"
        },
        "depth": {
            "filename": "control_v11f1p_sd15_depth.pth",
            "repo_id": "lllyasviel/ControlNet-v1-1",
            "local_path": CONTROLNET_DIR / "control_v11f1p_sd15_depth.pth",
            "diffusers_path": CONTROLNET_DIR / "depth"
        },
        "pose": {
            "filename": "control_v11p_sd15_openpose.pth",
            "repo_id": "lllyasviel/ControlNet-v1-1",
            "local_path": CONTROLNET_DIR / "control_v11p_sd15_openpose.pth",
            "diffusers_path": CONTROLNET_DIR / "pose"
        }
    }
}

# Training configurations
TRAIN_CONFIG = {
    "batch_size": 4,
    "learning_rate": 1e-5,
    "num_epochs": 10,
    "save_every": 1000,
    "eval_every": 500
}

# Generation configurations
GENERATION_CONFIG = {
    "guidance_scale": 7.5,
    "steps": 50,
    "height": 512,
    "width": 512
}

# Print confirmation message
print(f"Configured project at {ROOT_DIR}")

# Function to get model paths
def get_model_path(model_type, variant=None):
    """Get path to model checkpoint"""
    if model_type == "sd":
        return MODEL_CONFIGS["sd_1_5"]["local_path"]
    elif model_type == "controlnet" and variant in MODEL_CONFIGS["controlnet"]:
        return MODEL_CONFIGS["controlnet"][variant]["diffusers_path"]
    else:
        raise ValueError(f"Unknown model type: {model_type} or variant: {variant}")

# Configuration complete
config = {
    "directories": {
        "root": ROOT_DIR,
        "data": DATA_DIR,
        "models": MODELS_DIR,
        "results": RESULTS_DIR
    },
    "model_configs": MODEL_CONFIGS,
    "train_config": TRAIN_CONFIG,
    "generation_config": GENERATION_CONFIG
}