"""
Setup script to prepare the environment and download dataset
"""
import os
import zipfile
import requests
from pathlib import Path

def download_dataset(url, extract_path="./"):
    """Download and extract dataset"""
    print("Downloading dataset...")
    
    # Create directory if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)
    
    # Download file
    response = requests.get(url, stream=True)
    zip_path = os.path.join(extract_path, "dataset.zip")
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print("Extracting dataset...")
    
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Remove zip file
    os.remove(zip_path)
    
    print("Dataset downloaded and extracted successfully!")

def setup_directories():
    """Create necessary directories"""
    directories = [
        "checkpoints",
        "logs",
        "outputs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        "torch",
        "torchvision", 
        "wandb",
        "PIL",
        "matplotlib",
        "tqdm",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == "PIL":
                try:
                    __import__("pillow")
                except ImportError:
                    missing_packages.append("pillow")
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print("Missing packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("All required packages are installed!")
        return True

def main():
    print("=== UNet Polygon Coloring Setup ===")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Setup directories
    setup_directories()
    
    # Ask user if they want to download dataset
    download = input("\nDo you want to download the dataset? (y/n): ").lower().strip()
    
    if download == 'y':
        dataset_url = input("Enter dataset URL: ").strip()
        if dataset_url:
            try:
                download_dataset(dataset_url)
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                print("Please download the dataset manually and extract it to the 'dataset' folder.")
        else:
            print("No URL provided. Please download the dataset manually.")
    
    print("\n=== Setup Complete ===")
    print("Next steps:")
    print("1. Ensure your dataset is in the 'dataset' folder")
    print("2. Run training: python scripts/train.py")
    print("3. Use inference notebook: jupyter notebook inference_demo.ipynb")

if __name__ == "__main__":
    main()
