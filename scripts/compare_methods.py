import sys
import os
sys.path.append(os.path.abspath("."))  # Adds the current directory to path
import argparse
import torch
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.controlnet import create_controlnet_handler
from diffusers import StableDiffusionPipeline
from src.evaluation.metrics import get_evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Compare different image generation methods")
    parser.add_argument("--sd_model", type=str, required=True, help="Path to SD model checkpoint")
    parser.add_argument("--controlnet_models", type=str, nargs="+", required=True, 
                        help="Paths to ControlNet model checkpoints")
    parser.add_argument("--control_types", type=str, nargs="+", required=True,
                        help="Types of control, same order as controlnet_models")
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Directory with input images for conditioning")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="JSON file with prompts")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per prompt and method")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for generation")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of denoising steps")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directories
    base_output_dir = Path(args.output_dir)
    image_output_dir = base_output_dir / "images" / "comparison"
    metrics_output_dir = base_output_dir / "metrics"
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(metrics_output_dir, exist_ok=True)
    
    # Load prompts
    with open(args.prompts_file, 'r') as f:
        prompts = json.load(f)
    
    # Load input images
    image_paths = list(Path(args.image_dir).glob("*.png")) + list(Path(args.image_dir).glob("*.jpg"))
    
    # Make sure we have matching numbers
    min_count = min(len(prompts), len(image_paths))
    prompts = prompts[:min_count]
    image_paths = image_paths[:min_count]
    
    # Initialize models
    print("Initializing models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize base SD model
    sd_pipeline = StableDiffusionPipeline.from_pretrained(
        args.sd_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    
    # Initialize ControlNet models
    controlnet_handlers = []
    for model_path, control_type in zip(args.controlnet_models, args.control_types):
        handler = create_controlnet_handler(
            args.sd_model,
            model_path,
            control_type
        )
        controlnet_handlers.append((handler, control_type))
    
    # Initialize evaluator
    evaluator = get_evaluator(device)
    
    # Generate images with different methods
    results = {
        "unconditional": [],
        "controlnet": {}
    }
    
    for control_type in args.control_types:
        results["controlnet"][control_type] = []
    
    print("Generating images...")
    for i, (prompt, image_path) in enumerate(tqdm(zip(prompts, image_paths), total=min_count)):
        input_image = Image.open(image_path).convert("RGB")
        
        # Create a directory for this comparison
        sample_dir = image_output_dir / f"sample_{i:04d}"
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save input image
        input_image.save(sample_dir / "input.png")
        
        # Generate with base SD
        unconditional_images = []
        for j in range(args.num_samples):
            with torch.no_grad():
                image = sd_pipeline(
                    prompt=prompt,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.steps
                ).images[0]
            
            image_path = sample_dir / f"unconditional_{j}.png"
            image.save(image_path)
            unconditional_images.append(image)
        
        results["unconditional"].append(unconditional_images)
        
        # Generate with each ControlNet
        for handler, control_type in controlnet_handlers:
            controlnet_images = []
            
            # Prepare conditioning image
            condition_image = handler.prepare_conditioning_image(image_path)
            condition_image.save(sample_dir / f"condition_{control_type}.png")
            
            # Generate images
            with torch.no_grad():
                images = handler.generate_images(
                    prompt=prompt,
                    conditioning_image=condition_image,
                    num_images=args.num_samples,
                    guidance_scale=args.guidance_scale,
                    steps=args.steps
                )
            
            for j, image in enumerate(images):
                image_path = sample_dir / f"{control_type}_{j}.png"
                image.save(image_path)
                controlnet_images.append(image)
                
            results["controlnet"][control_type].append(controlnet_images)
        
    # Create comparison visualization
    print("Creating comparison visualizations...")
    for i in range(min_count):
        fig, axes = plt.subplots(1 + len(args.control_types), args.num_samples + 1, figsize=(4 * (args.num_samples + 1), 4 * (1 + len(args.control_types))))
        
        # Input image
        axes[0, 0].imshow(Image.open(image_output_dir / f"sample_{i:04d}" / "input.png"))
        axes[0, 0].set_title("Input Image")
        axes[0, 0].axis('off')
        
        # Unconditional results
        for j in range(args.num_samples):
            axes[0, j+1].imshow(Image.open(image_output_dir / f"sample_{i:04d}" / f"unconditional_{j}.png"))
            axes[0, j+1].set_title(f"Unconditional {j+1}")
            axes[0, j+1].axis('off')
        
        # ControlNet results
        for k, control_type in enumerate(args.control_types):
            # Conditioning image
            axes[k+1, 0].imshow(Image.open(image_output_dir / f"sample_{i:04d}" / f"condition_{control_type}.png"))
            axes[k+1, 0].set_title(f"{control_type.capitalize()} Condition")
            axes[k+1, 0].axis('off')
            
            # Generated images
            for j in range(args.num_samples):
                axes[k+1, j+1].imshow(Image.open(image_output_dir / f"sample_{i:04d}" / f"{control_type}_{j}.png"))
                axes[k+1, j+1].set_title(f"{control_type.capitalize()} {j+1}")
                axes[k+1, j+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(image_output_dir / f"comparison_{i:04d}.png")
        plt.close()
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = {}
    
    # CLIP scores
    clip_scores = {
        "unconditional": evaluator.calculate_clip_score(
            [img for sublist in results["unconditional"] for img in sublist],
            prompts * args.num_samples
        )
    }
    
    for control_type in args.control_types:
        clip_scores[control_type] = evaluator.calculate_clip_score(
            [img for sublist in results["controlnet"][control_type] for img in sublist],
            prompts * args.num_samples
        )
    
    metrics["clip_scores"] = clip_scores
    
    # Control alignment scores for each ControlNet type
    alignment_scores = {}
    for control_type, handler_tuple in zip(args.control_types, controlnet_handlers):
        handler, _ = handler_tuple
        
        # Get conditioning images
        condition_images = []
        for image_path in image_paths:
            condition_image = handler.prepare_conditioning_image(image_path)
            condition_images.append(condition_image)
        
        # Calculate alignment
        # Calculate alignment
        alignment_scores[control_type] = evaluator.calculate_control_alignment(
            condition_images * args.num_samples,
            [img for sublist in results["controlnet"][control_type] for img in sublist],
            control_type
        )
    
    metrics["alignment_scores"] = alignment_scores
    
    # Save metrics
    with open(metrics_output_dir / "comparison_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate summary report
    create_summary_report(metrics, base_output_dir)
    
    print(f"Comparison complete! Results saved to {base_output_dir}")

def create_summary_report(metrics, output_dir):
    """Create a markdown summary report of the comparison results"""
    report_path = output_dir / "comparison_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Image Generation Methods Comparison\n\n")
        
        # CLIP Scores
        f.write("## CLIP Score Comparison\n\n")
        f.write("Higher is better (text-image alignment)\n\n")
        f.write("| Method | CLIP Score |\n")
        f.write("|--------|------------|\n")
        
        clip_scores = metrics["clip_scores"]
        f.write(f"| Unconditional | {clip_scores['unconditional']:.4f} |\n")
        
        for control_type, score in clip_scores.items():
            if control_type != "unconditional":
                f.write(f"| {control_type.capitalize()} | {score:.4f} |\n")
        
        f.write("\n")
        
        # Alignment Scores
        f.write("## Control Alignment Scores\n\n")
        f.write("Higher is better (condition-output alignment)\n\n")
        f.write("| Control Type | Alignment Score |\n")
        f.write("|--------------|----------------|\n")
        
        alignment_scores = metrics["alignment_scores"]
        for control_type, score in alignment_scores.items():
            f.write(f"| {control_type.capitalize()} | {score:.4f} |\n")
        
        f.write("\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("This comparison evaluated different image generation methods using Stable Diffusion with various conditioning approaches.\n\n")
        
        # Find best method by CLIP score
        best_clip_method = max(clip_scores.items(), key=lambda x: x[1])
        f.write(f"- Best text-image alignment: **{best_clip_method[0].capitalize()}** (CLIP Score: {best_clip_method[1]:.4f})\n")
        
        # Find best control method by alignment
        if alignment_scores:
            best_alignment_method = max(alignment_scores.items(), key=lambda x: x[1])
            f.write(f"- Best condition-output alignment: **{best_alignment_method[0].capitalize()}** (Alignment Score: {best_alignment_method[1]:.4f})\n")
        
        f.write("\nSee the images directory for visual comparisons of the generated samples.\n")

if __name__ == "__main__":
    main()