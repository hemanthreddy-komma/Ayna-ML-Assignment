import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from model import ConditionalUNet
from dataset import create_dataloaders

class PolygonTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize wandb
        wandb.init(project="polygon-coloring-unet", config=config)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.color_to_idx = create_dataloaders(
            config['data_dir'], 
            config['batch_size'], 
            config['image_size']
        )
        
        # Initialize model
        self.model = ConditionalUNet(
            n_channels=3, 
            n_classes=3, 
            num_colors=len(self.color_to_idx),
            bilinear=config['bilinear']
        ).to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # Create checkpoint directory
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        # Track best validation loss
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device)
            targets = batch['output'].to(self.device)
            color_indices = batch['color_idx'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs, color_indices)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to wandb every 100 batches
            if batch_idx % 100 == 0:
                wandb.log({
                    'train_loss_step': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                inputs = batch['input'].to(self.device)
                targets = batch['output'].to(self.device)
                color_indices = batch['color_idx'].to(self.device)
                
                outputs = self.model(inputs, color_indices)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                pbar.set_postfix({'val_loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def log_sample_images(self, epoch):
        """Log sample predictions to wandb"""
        self.model.eval()
        with torch.no_grad():
            # Get a batch from validation set
            batch = next(iter(self.val_loader))
            inputs = batch['input'][:4].to(self.device)  # Take first 4 samples
            targets = batch['output'][:4].to(self.device)
            color_indices = batch['color_idx'][:4].to(self.device)
            color_names = batch['color_name'][:4]
            
            outputs = self.model(inputs, color_indices)
            
            # Convert tensors to images for logging
            images = []
            for i in range(4):
                # Convert tensors to PIL images
                input_img = transforms.ToPILImage()(inputs[i].cpu())
                target_img = transforms.ToPILImage()(targets[i].cpu())
                output_img = transforms.ToPILImage()(outputs[i].cpu())
                
                # Create a combined image
                combined = Image.new('RGB', (input_img.width * 3, input_img.height))
                combined.paste(input_img, (0, 0))
                combined.paste(target_img, (input_img.width, 0))
                combined.paste(output_img, (input_img.width * 2, 0))
                
                images.append(wandb.Image(
                    combined, 
                    caption=f"Color: {color_names[i]} | Input | Target | Prediction"
                ))
            
            wandb.log({f"predictions_epoch_{epoch}": images})
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'color_to_idx': self.color_to_idx,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {val_loss:.6f}")
    
    def train(self):
        print("Starting training...")
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log metrics
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Log sample images every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.log_sample_images(epoch + 1)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch + 1, val_loss, is_best)
        
        print("Training completed!")
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Train UNet for Polygon Coloring')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--bilinear', action='store_true', help='Use bilinear upsampling')
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'image_size': args.image_size,
        'checkpoint_dir': args.checkpoint_dir,
        'bilinear': args.bilinear
    }
    
    trainer = PolygonTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
