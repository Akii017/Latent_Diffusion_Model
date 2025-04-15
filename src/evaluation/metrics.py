import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision import models
import clip
from scipy import linalg
import lpips
import cv2
from pathlib import Path
import json

class EvaluationMetrics:
    """Class for evaluating image generation metrics"""
    
    def __init__(self, device=None):
        """
        Initialize evaluation metrics
        
        Args:
            device: Torch device (defaults to CUDA if available)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models required for metrics calculation"""
        # Initialize models lazily to save resources
        self.inception_model = None
        self.clip_model = None
        self.lpips_model = None
    
    def _get_inception_model(self):
        """Get Inception model for FID calculation"""
        if self.inception_model is None:
            # Load pre-trained Inception model
            self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
            self.inception_model.fc = torch.nn.Identity()  # Remove final FC layer
            self.inception_model.eval()
            self.inception_model.to(self.device)
            
            # Define transform for inception model
            self.inception_transform = T.Compose([
                T.Resize(299),
                T.CenterCrop(299),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return self.inception_model
    
    def _get_clip_model(self):
        """Get CLIP model for CLIP score calculation"""
        if self.clip_model is None:
            # Load CLIP model
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        return self.clip_model, self.clip_preprocess
    
    def _get_lpips_model(self):
        """Get LPIPS model for perceptual similarity"""
        if self.lpips_model is None:
            # Load LPIPS model
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        
        return self.lpips_model
    
    def compute_inception_features(self, images):
        """
        Compute Inception features for a batch of images
        
        Args:
            images: List of PIL images
            
        Returns:
            Numpy array of features
        """
        model = self._get_inception_model()
        
        # Convert images to tensors
        tensors = []
        for img in images:
            tensor = self.inception_transform(img).unsqueeze(0)
            tensors.append(tensor)
        
        batch = torch.cat(tensors, dim=0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = model(batch)
        
        return features.cpu().numpy()
    
    def calculate_fid(self, real_images, generated_images):
        """
        Calculate Frechet Inception Distance between real and generated images
        
        Args:
            real_images: List of real PIL images
            generated_images: List of generated PIL images
            
        Returns:
            FID score (lower is better)
        """
        # Get features
        real_features = self.compute_inception_features(real_images)
        gen_features = self.compute_inception_features(generated_images)
        
        # Calculate mean and covariance
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
        
        # Calculate FID
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # Numerical stability
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return float(fid)
    
    def calculate_clip_score(self, images, prompts):
        """
        Calculate CLIP score between images and prompts
        
        Args:
            images: List of PIL images
            prompts: List of text prompts
            
        Returns:
            Average CLIP score (higher is better)
        """
        model, preprocess = self._get_clip_model()
        
        # Preprocess images
        image_inputs = []
        for img in images:
            image_input = preprocess(img).unsqueeze(0).to(self.device)
            image_inputs.append(image_input)
        
        image_inputs = torch.cat(image_inputs, dim=0)
        
        # Tokenize text
        text_tokens = clip.tokenize(prompts).to(self.device)
        
        # Calculate similarity
        with torch.no_grad():
            image_features = model.encode_image(image_inputs)
            text_features = model.encode_text(text_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Calculate similarity
            similarities = (100.0 * image_features @ text_features.T).diag()
            
        return similarities.mean().item()
    
    def calculate_lpips(self, images1, images2):
        """
        Calculate LPIPS perceptual similarity between pairs of images
        
        Args:
            images1: List of PIL images
            images2: List of PIL images
            
        Returns:
            Average LPIPS score (lower is better)
        """
        model = self._get_lpips_model()
        
        # Convert to tensors
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1)  # Scale to [-1, 1]
        ])
        
        scores = []
        for img1, img2 in zip(images1, images2):
            tensor1 = transform(img1).unsqueeze(0).to(self.device)
            tensor2 = transform(img2).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                dist = model(tensor1, tensor2)
            
            scores.append(dist.item())
        
        return np.mean(scores)
    
    def calculate_control_alignment(self, control_images, generated_images, control_type="canny"):
        """
        Calculate alignment between control images and generated images
        
        Args:
            control_images: List of PIL control images
            generated_images: List of PIL generated images
            control_type: Type of control ('canny', 'depth', 'pose')
            
        Returns:
            Average alignment score (higher is better)
        """
        if control_type == "canny":
            # For Canny edges, calculate edge overlap
            scores = []
            for control_img, gen_img in zip(control_images, generated_images):
                # Convert to numpy arrays
                control_np = np.array(control_img.convert('L'))
                gen_np = np.array(gen_img.convert('L'))
                
                if gen_np.shape != control_np.shape:
                    gen_np = cv2.resize(gen_np, (control_np.shape[1], control_np.shape[0]), interpolation=cv2.INTER_LINEAR)
                # Detect edges in generated image
                gen_edges = cv2.Canny(gen_np, 100, 200)
                
                # Calculate overlap (accounting for edge thickness differences)
                control_dilated = cv2.dilate(control_np, np.ones((3, 3), np.uint8), iterations=2)
                overlap = np.logical_and(control_dilated > 0, gen_edges > 0).sum()
                total_edges = np.sum(control_dilated > 0)
                
                if total_edges > 0:
                    score = overlap / total_edges
                else:
                    score = 0
                    
                scores.append(score)
            
            return np.mean(scores)
        
        # Add metrics for other control types
        elif control_type == "depth":
            # Simple correlation between depth maps
            # In a real implementation, you'd need to extract depth from the generated image
            return 0.5  # Placeholder
            
        elif control_type == "pose":
            # Pose similarity metric
            # In a real implementation, you'd need to extract pose from the generated image
            return 0.5  # Placeholder
            
        else:
            return 0.0
    
    def evaluate_and_save(self, real_dir, generated_dir, prompts, control_images=None, 
                          control_type=None, output_path="results/metrics/evaluation.json"):
        """
        Run full evaluation and save results
        
        Args:
            real_dir: Directory with real images
            generated_dir: Directory with generated images
            prompts: List of prompts used for generation
            control_images: Optional list of control images
            control_type: Optional control type
            output_path: Path to save results
            
        Returns:
            Dictionary with metrics
        """
        # Load images
        real_images = [Image.open(p) for p in Path(real_dir).glob("*.png")]
        generated_images = [Image.open(p) for p in Path(generated_dir).glob("*.png")]
        
        # Ensure we have matching numbers
        min_count = min(len(real_images), len(generated_images))
        real_images = real_images[:min_count]
        generated_images = generated_images[:min_count]
        
        # Calculate metrics
        metrics = {
            "fid": self.calculate_fid(real_images, generated_images),
            "clip_score": self.calculate_clip_score(generated_images, prompts[:min_count])
        }
        
        # Add control alignment if applicable
        if control_images and control_type:
            control_images = control_images[:min_count]
            metrics["control_alignment"] = self.calculate_control_alignment(
                control_images, generated_images, control_type
            )
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics

# Create a helper function to get the metrics evaluator
def get_evaluator(device=None):
    """Get an instance of the EvaluationMetrics class"""
    return EvaluationMetrics(device)