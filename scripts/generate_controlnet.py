import sys
import os
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
import argparse
import datetime
from pathlib import Path
from src.models.controlnet import create_controlnet_handler

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using ControlNet")
    parser.add_argument("--sd_model", type=str, required=True, help="Path to SD model checkpoint")
    parser.add_argument("--controlnet_model", type=str, required=True, help="Path to ControlNet checkpoint")
    parser.add_argument("--control_type", type=str, default="canny", choices=["canny", "depth", "pose"], 
                        help="Type of conditioning")
    parser.add_argument("--condition_image", type=str, required=True, help="Path to conditioning image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="results/images", help="Output directory")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading models: SD from {args.sd_model}, ControlNet ({args.control_type}) from {args.controlnet_model}")
    
    # Create ControlNet handler
    handler = create_controlnet_handler(
        args.sd_model,
        args.controlnet_model,
        args.control_type
    )
    
    print(f"Generating {args.num_samples} images with prompt: '{args.prompt}'")
    print(f"Using condition image: {args.condition_image}")
    
    # Generate images
    images = handler.generate_images(
        prompt=args.prompt,
        conditioning_image=args.condition_image,
        negative_prompt=args.negative_prompt,
        num_images=args.num_samples,
        guidance_scale=args.guidance_scale,
        steps=args.steps
    )
    
    # Save images
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = args.prompt.replace(" ", "_")[:30]
    control_type = args.control_type
    
    for i, image in enumerate(images):
        filename = f"{timestamp}_{control_type}_{prompt_slug}_{i}.png"
        save_path = output_dir / filename
        image.save(save_path)
        print(f"Saved image to {save_path}")

if __name__ == "__main__":
    main()