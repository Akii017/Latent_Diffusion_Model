import argparse
import os
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from PIL import Image
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="results/images", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for text conditioning")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        
    print(f"Loading model from {args.checkpoint}...")
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    print(f"Generating {args.num_samples} images with prompt: '{args.prompt}'")
    
    # Generate images
    for i in range(args.num_samples):
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps
        ).images[0]
        
        # Create a timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Format filename
        prompt_slug = args.prompt.replace(" ", "_")[:30]
        filename = f"{timestamp}_{prompt_slug}_{i}.png"
        save_path = output_dir / filename
        
        # Save image
        image.save(save_path)
        print(f"Saved image to {save_path}")

if __name__ == "__main__":
    main()