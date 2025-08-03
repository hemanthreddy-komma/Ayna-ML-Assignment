import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    from model import ConditionalUNet
    
    # Get model parameters from checkpoint
    num_colors = len(checkpoint['color_to_idx'])
    config = checkpoint.get('config', {})
    bilinear = config.get('bilinear', False)
    
    # Initialize model
    model = ConditionalUNet(
        n_channels=3, 
        n_classes=3, 
        num_colors=num_colors,
        bilinear=bilinear
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['color_to_idx']

def preprocess_image(image_path, image_size=256):
    """Preprocess input image for inference"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def postprocess_output(output_tensor):
    """Convert model output to PIL image"""
    # Remove batch dimension and convert to PIL
    output = output_tensor.squeeze(0).cpu()
    output = torch.clamp(output, 0, 1)  # Ensure values are in [0, 1]
    return transforms.ToPILImage()(output)

def visualize_prediction(input_img, target_img, prediction_img, color_name, save_path=None):
    """Visualize input, target, and prediction side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(input_img)
    axes[0].set_title('Input Polygon')
    axes[0].axis('off')
    
    if target_img is not None:
        axes[1].imshow(target_img)
        axes[1].set_title('Target')
        axes[1].axis('off')
    else:
        axes[1].axis('off')
    
    axes[2].imshow(prediction_img)
    axes[2].set_title(f'Prediction ({color_name})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    mse = torch.nn.functional.mse_loss(predictions, targets)
    mae = torch.nn.functional.l1_loss(predictions, targets)
    
    # PSNR calculation
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    return {
        'MSE': mse.item(),
        'MAE': mae.item(),
        'PSNR': psnr.item()
    }

def generate_synthetic_polygon(shape='triangle', size=256, color='white'):
    """Generate synthetic polygon for testing"""
    from PIL import ImageDraw
    
    img = Image.new('RGB', (size, size), 'black')
    draw = ImageDraw.Draw(img)
    
    center = size // 2
    radius = size // 3
    
    if shape == 'triangle':
        points = [
            (center, center - radius),
            (center - radius * 0.866, center + radius * 0.5),
            (center + radius * 0.866, center + radius * 0.5)
        ]
        draw.polygon(points, fill=color)
    elif shape == 'square':
        points = [
            (center - radius, center - radius),
            (center + radius, center - radius),
            (center + radius, center + radius),
            (center - radius, center + radius)
        ]
        draw.polygon(points, fill=color)
    elif shape == 'pentagon':
        points = []
        for i in range(5):
            angle = i * 2 * np.pi / 5 - np.pi / 2
            x = center + radius * np.cos(angle)
            y = center + radius * np.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color)
    elif shape == 'hexagon':
        points = []
        for i in range(6):
            angle = i * 2 * np.pi / 6
            x = center + radius * np.cos(angle)
            y = center + radius * np.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color)
    
    return img

if __name__ == "__main__":
    # Test synthetic polygon generation
    shapes = ['triangle', 'square', 'pentagon', 'hexagon']
    
    fig, axes = plt.subplots(1, len(shapes), figsize=(16, 4))
    
    for i, shape in enumerate(shapes):
        img = generate_synthetic_polygon(shape, color='white')
        axes[i].imshow(img)
        axes[i].set_title(shape.capitalize())
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
