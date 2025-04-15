# src/training/trainer.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path
import wandb

from src.config import config

class DiffusionTrainer:
    """Trainer for latent diffusion models"""
    
    def __init__(self, model, train_dataloader, val_dataloader=None, 
                 lr=config.LEARNING_RATE, device="cuda"):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS * len(train_dataloader)
        )
        
        # Checkpoint directory
        self.checkpoint_dir = config.MODEL_DIR / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Logging
        self.log_dir = config.BASE_DIR / "logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            # Get data
            images = batch["image"].to(self.device)
            conditions = batch["condition"].to(self.device) if "condition" in batch else None
            control_inputs = batch["control"].to(self.device) if "control" in batch else None
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            t = torch.randint(0, config.TIMESTEPS, (images.shape[0],), device=self.device)
            
            # Different loss calculation based on model type
            if hasattr(self.model, "controlnet") and control_inputs is not None:
                # For ControlNet model
                loss = self.train_step_controlnet(images, t, conditions, control_inputs)
            else:
                # For standard diffusion model
                loss = self.train_step_diffusion(images, t, conditions)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Log to wandb
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": self.scheduler.get_last_lr()[0]
            })
        
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    
    def train_step_diffusion(self, images, t, conditions=None):
        """Training step for standard diffusion model"""
        return self.model.p_losses(images, t, conditions)
    
    def train_step_controlnet(self, images, t, conditions, control_inputs):
        """Training step for ControlNet model"""
        # Encode images to latent space
        latents = self.model.encode_image(images)
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.model.ldm.q_sample(latents, t, noise=noise)
        
        # Predict noise with controlnet
        pred_noise = self.model(noisy_latents, t, conditions, control_inputs)
        
        # Calculate loss
        loss = nn.functional.mse_loss(pred_noise, noise)
        
        return loss
    
    def validate(self, epoch):
        """Validate the model"""
        if self.val_dataloader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Get data
                images = batch["image"].to(self.device)
                conditions = batch["condition"].to(self.device) if "condition" in batch else None
                control_inputs = batch["control"].to(self.device) if "control" in batch else None
                
                # Random timesteps
                t = torch.randint(0, config.TIMESTEPS, (images.shape[0],), device=self.device)
                
                # Calculate loss
                if hasattr(self.model, "controlnet") and control_inputs is not None:
                    # For ControlNet model
                    loss = self.validate_step_controlnet(images, t, conditions, control_inputs)
                else:
                    # For standard diffusion model
                    loss = self.validate_step_diffusion(images, t, conditions)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        
        # Log to wandb
        wandb.log({"val_loss": avg_loss})
        
        return avg_loss
    
    def validate_step_diffusion(self, images, t, conditions=None):
        """Validation step for standard diffusion model"""
        return self.model.p_losses(images, t, conditions)
    
    def validate_step_controlnet(self, images, t, conditions, control_inputs):
        """Validation step for ControlNet model"""
        # Encode images to latent space
        latents = self.model.encode_image(images)
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.model.ldm.q_sample(latents, t, noise=noise)
        
        # Predict noise with controlnet
        pred_noise = self.model(noisy_latents, t, conditions, control_inputs)
        
        # Calculate loss
        loss = nn.functional.mse_loss(pred_noise, noise)
        
        return loss
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch}.pth"
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"Loaded checkpoint from {checkpoint_path}, epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]
    
    def train(self, num_epochs=config.NUM_EPOCHS, resume_from=None):
        """Full training loop"""
        # Initialize wandb
        wandb.init(project="latent-diffusion-research", config={
            "model_type": config.MODEL_TYPE,
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "epochs": num_epochs,
        })
        
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        for epoch in range(start_epoch, num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} train loss: {train_loss:.6f}")
            
            # Validation
            if epoch % config.EVAL_INTERVAL == 0:
                val_loss = self.validate(epoch)
                print(f"Epoch {epoch} validation loss: {val_loss:.6f}")
                
                # Generate samples
                self.generate_samples(epoch)
            
            # Save checkpoint
            if epoch % config.SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch, val_loss if epoch % config.EVAL_INTERVAL == 0 else None)
        
        # Final save
        self.save_checkpoint(num_epochs, self.validate(num_epochs))
        
        # Close wandb
        wandb.finish()
    
    @torch.no_grad()
    def generate_samples(self, epoch, num_samples=4):
        """Generate and save sample images"""
        self.model.eval()
        
        # Create sampling directory
        sample_dir = config.RESULTS_DIR / "samples" / f"epoch_{epoch}"
        sample_dir.mkdir(exist_ok=True, parents=True)
        
        # Sample from validation dataset
        if self.val_dataloader is not None:
            batch = next(iter(self.val_dataloader))
            conditions = batch["condition"].to(self.device) if "condition" in batch else None
            control_inputs = batch["control"].to(self.device) if "control" in batch else None
            
            # Generate samples
            shape = (num_samples, config.LATENT_DIM, config.IMAGE_SIZE // 8, config.IMAGE_SIZE // 8)
            
            if hasattr(self.model, "controlnet") and control_inputs is not None:
                # For ControlNet model (use only first num_samples)
                samples = self.model.sample(
                    shape, 
                    self.device, 
                    conditions[:num_samples] if conditions is not None else None,
                    control_inputs[:num_samples]
                )
            else:
                # For standard diffusion model
                samples = self.model.sample(shape, self.device, conditions[:num_samples] if conditions is not None else None)
            
            # Save samples
            for i, sample in enumerate(samples):
                # Convert tensor to PIL image and save
                sample = sample.cpu().permute(1, 2, 0).numpy()
                sample = (sample * 255).clip(0, 255).astype(np.uint8)
                
                # Save with PIL
                from PIL import Image
                Image.fromarray(sample).save(sample_dir / f"sample_{i}.png")
            
            # Log to wandb
            wandb.log({
                "samples": [wandb.Image(str(sample_dir / f"sample_{i}.png")) for i in range(num_samples)]
            })