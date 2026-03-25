import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse
# Import MNC from MNC.py
from MNC import calculate_mnc

def load_image(path):
    """Load an image from path and convert to numpy array."""
    if os.path.exists(path):
        img = Image.open(path)
        return np.array(img)
    else:
        print(f"Warning: Image not found at {path}")
        return None

def calculate_image_metrics(img1, img2):
    """Calculate PSNR and SSIM between two images."""
    # Convert to grayscale if needed for comparison
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1_gray = np.mean(img1, axis=2).astype(np.uint8)
    else:
        img1_gray = img1
        
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2_gray = np.mean(img2, axis=2).astype(np.uint8)
    else:
        img2_gray = img2
    
    # Ensure images have the same dimensions
    if img1_gray.shape != img2_gray.shape:
        print(f"Warning: Images have different shapes: {img1_gray.shape} vs {img2_gray.shape}")
        # Resize the second image to match the first
        img2_gray = np.array(Image.fromarray(img2_gray).resize(
            (img1_gray.shape[1], img1_gray.shape[0]), Image.LANCZOS))
    
    # Calculate metrics
    psnr_value = psnr(img1_gray, img2_gray)
    ssim_value = ssim(img1_gray, img2_gray)
    
    return psnr_value, ssim_value

def calculate_kernel_metrics(kernel1, kernel2):
    """Calculate MSE and MNC between two kernels."""
    # Ensure kernels have the same dimensions
    if kernel1.shape != kernel2.shape:
        print(f"Warning: Kernels have different shapes: {kernel1.shape} vs {kernel2.shape}")
        # Resize the second kernel to match the first
        kernel2 = np.array(Image.fromarray(kernel2).resize(
            (kernel1.shape[1], kernel1.shape[0]), Image.LANCZOS))
    
    # Calculate metrics
    mse_value = mse(kernel1.flatten(), kernel2.flatten())
    mnc_value = calculate_mnc(kernel1, kernel2)
    
    return mse_value, mnc_value

def visualize_comparison(path1, path2, is_image=True):
    """
    Visualize two images/kernels side by side and calculate appropriate metrics.
    
    Args:
        path1: Path to the first image/kernel
        path2: Path to the second image/kernel
        is_image: Boolean indicating if we're comparing images (True) or kernels (False)
    """
    # Load images/kernels
    item1 = load_image(path1)
    item2 = load_image(path2)
    
    if item1 is None or item2 is None:
        print("Error: Could not load one or both items.")
        return
    
    # Calculate metrics based on type
    if is_image:
        metric1_value, metric2_value = calculate_image_metrics(item1, item2)
        metric1_name, metric2_name = "PSNR", "SSIM"
        metric1_format, metric2_format = f"{metric1_value:.2f} dB", f"{metric2_value:.4f}"
    else:
        metric1_value, metric2_value = calculate_kernel_metrics(item1, item2)
        metric1_name, metric2_name = "MSE", "MNC"
        metric1_format, metric2_format = f"{metric1_value:.6f}", f"{metric2_value:.6f}"
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot first item
    if len(item1.shape) == 3:  # Color image
        axes[0].imshow(item1)
    else:  # Grayscale image or kernel
        axes[0].imshow(item1, cmap='gray')
    axes[0].set_title(f"Item 1: {os.path.basename(path1)}")
    axes[0].axis('off')
    
    # Plot second item
    if len(item2.shape) == 3:  # Color image
        axes[1].imshow(item2)
    else:  # Grayscale image or kernel
        axes[1].imshow(item2, cmap='gray')
    axes[1].set_title(f"Item 2: {os.path.basename(path2)}")
    axes[1].axis('off')
    
    # Add metrics as suptitle
    item_type = "Image" if is_image else "Kernel"
    plt.suptitle(f"{item_type} Comparison Metrics: {metric1_name} = {metric1_format}, {metric2_name} = {metric2_format}")
    plt.tight_layout()
    plt.show()

def visualize_results(base_dir="Datasets/Conv_Mode/Base", 
                      results_dir="results/Conv_Mode",
                      is_image=True):
    """
    Visualize the base items and their corresponding predicted results.
    
    Args:
        base_dir: Directory containing the base items
        results_dir: Directory containing the model's predicted results
        is_image: Boolean indicating if we're comparing images (True) or kernels (False)
    """
    # Convert backslashes to forward slashes for cross-platform compatibility
    base_dir = base_dir.replace('\\', '/')
    results_dir = results_dir.replace('\\', '/')
    
    # Check if directories exist
    for dir_path, dir_name in [(base_dir, "Base"), (results_dir, "Results")]:
        if not os.path.exists(dir_path):
            print(f"Error: {dir_name} directory '{dir_path}' does not exist.")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
            return
    
    # Load base images
    try:
        base_files = os.listdir(base_dir)
        base_images = {}
        for file in base_files:
            if file.startswith('im') and file.endswith('.png'):
                # Extract the image number (e.g., "1" from "im1_Base.png")
                img_num = file.split('_')[0][2:]  # Remove "im" prefix
                base_images[img_num] = os.path.join(base_dir, file)
        
        print(f"Found {len(base_images)} base images in {base_dir}")
        if base_images:
            print("Base image keys:", list(base_images.keys()))
    except Exception as e:
        print(f"Error loading base images: {e}")
        return
    
    # Load result images
    try:
        result_files = os.listdir(results_dir)
        result_images = {}
        
        print(f"Found {len(result_files)} files in results directory")
        if result_files:
            print("Sample result filenames:", result_files[:5] if len(result_files) > 5 else result_files)
        
        # Try to match result images with base images
        # This depends on your result file naming convention
        # Let's assume result files have names that include the base image number
        for file in result_files:
            if file.endswith('.png') or file.endswith('.jpg'):
                # Try to find which base image this result corresponds to
                for img_num in base_images.keys():
                    if f"im{img_num}" in file:
                        result_images[img_num] = os.path.join(results_dir, file)
                        break
        
        print(f"Found {len(result_images)} matching result images")
        if result_images:
            print("Matched result image keys:", list(result_images.keys()))
    except Exception as e:
        print(f"Error loading result images: {e}")
        return
    
    # Display base items with their corresponding results
    for img_num, base_path in base_images.items():
        if img_num in result_images:
            result_path = result_images[img_num]
            visualize_comparison(base_path, result_path, is_image)
        else:
            print(f"No matching result found for base item {img_num}")

if __name__ == "__main__":
    # Example usage:
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize and compare images or kernels.')
    parser.add_argument('--imgs', action='store_true', dest='is_image',
                        help='Compare images using PSNR and SSIM')
    parser.add_argument('--kernels', action='store_false', dest='is_image',
                        help='Compare kernels using MSE and MNC')
    parser.set_defaults(is_image=True)
    parser.add_argument('--base_dir', type=str, default="Datasets/Conv_Mode/Base",
                        help='Directory containing the base images/kernels')
    parser.add_argument('--results_dir', type=str, default="results/Conv_Mode",
                        help='Directory containing the predicted results')
    parser.add_argument('path1', nargs='?', type=str, help='Path to first image/kernel (optional)')
    parser.add_argument('path2', nargs='?', type=str, help='Path to second image/kernel (optional)')
    
    args = parser.parse_args()
    
    if args.path1 and args.path2:
        # If two paths are provided as command line arguments
        visualize_comparison(args.path1, args.path2, is_image=args.is_image)
    else:
        # Use the directory comparison function
        visualize_results(args.base_dir, args.results_dir, is_image=args.is_image) 