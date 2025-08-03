import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class PolygonDataset(Dataset):
    def __init__(self, data_dir, split='training', transform=None, augment=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.augment = augment
        
        # Load data mapping
        json_path = os.path.join(data_dir, split, 'data.json')
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Create color to index mapping
        colors = set()
        for item in self.data:
            colors.add(item['color'])
        self.color_to_idx = {color: idx for idx, color in enumerate(sorted(colors))}
        self.idx_to_color = {idx: color for color, idx in self.color_to_idx.items()}
        
        print(f"Found {len(self.color_to_idx)} unique colors: {list(self.color_to_idx.keys())}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load input and output images
        input_path = os.path.join(self.data_dir, self.split, 'inputs', item['input'])
        output_path = os.path.join(self.data_dir, self.split, 'outputs', item['output'])
        
        input_img = Image.open(input_path).convert('RGB')
        output_img = Image.open(output_path).convert('RGB')
        
        # Get color index
        color_idx = self.color_to_idx[item['color']]
        
        # Apply transforms
        if self.transform:
            # Apply same transform to both input and output for consistency
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            input_img = self.transform(input_img)
            
            torch.manual_seed(seed)
            output_img = self.transform(output_img)
        
        return {
            'input': input_img,
            'output': output_img,
            'color_idx': torch.tensor(color_idx, dtype=torch.long),
            'color_name': item['color']
        }

def get_transforms(image_size=256, augment=False):
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    
    if augment:
        transform_list.insert(-1, transforms.RandomHorizontalFlip(p=0.5))
        transform_list.insert(-1, transforms.RandomRotation(degrees=15))
        transform_list.insert(-1, transforms.ColorJitter(brightness=0.1, contrast=0.1))
    
    return transforms.Compose(transform_list)

def create_dataloaders(data_dir, batch_size=16, image_size=256, num_workers=4):
    # Create transforms
    train_transform = get_transforms(image_size, augment=True)
    val_transform = get_transforms(image_size, augment=False)
    
    # Create datasets
    train_dataset = PolygonDataset(data_dir, 'training', train_transform)
    val_dataset = PolygonDataset(data_dir, 'validation', val_transform)
    
    # Ensure both datasets have same color mapping
    val_dataset.color_to_idx = train_dataset.color_to_idx
    val_dataset.idx_to_color = train_dataset.idx_to_color
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.color_to_idx

if __name__ == "__main__":
    # Test the dataset
    data_dir = "dataset"  # Update this path
    train_loader, val_loader, color_to_idx = create_dataloaders(data_dir, batch_size=4)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Color mapping: {color_to_idx}")
    
    # Test a batch
    for batch in train_loader:
        print(f"Input shape: {batch['input'].shape}")
        print(f"Output shape: {batch['output'].shape}")
        print(f"Color indices: {batch['color_idx']}")
        print(f"Color names: {batch['color_name']}")
        break
