import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import shutil

def save_kernel_visualization(kernel, output_dir):
    """
    Save the kernel as both numpy array and visual representation
    """
    base_dir = Path(output_dir) / "Bases"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Save kernel as numpy array
    np.save(base_dir / "convolution_kernel.npy", kernel)
    
    # Create visual representation of the kernel
    plt.figure(figsize=(10, 10))
    plt.imshow(kernel, cmap='viridis')
    plt.colorbar()
    plt.title("Convolution Kernel")
    plt.savefig(base_dir / "convolution_kernel.png")
    plt.close()

def apply_iterative_convolution(image, severity, kernel_size=5):
    """
    Apply convolution iteratively based on severity level (1-10)
    Each severity level applies convolution 2*severity times
    
    Args:
        image: Input image (numpy array)
        severity: Integer from 1 to 10 indicating number of convolution iterations
        kernel_size: Size of the convolution kernel (default: 5)
    
    Returns:
        Convolved image, kernel
    """
    # Create a Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, sigma=1)
    kernel = kernel @ kernel.T  # Make 2D kernel
    
    # Apply convolution iteratively based on severity (2 times per severity level)
    result = image.copy()
    for _ in range(2 * severity):  # Doubled the iterations
        result = cv2.filter2D(result, -1, kernel)
    
    return result, kernel

def process_directory(input_dir, output_dir):
    """
    Process all images in input directory and create convolved versions
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each severity level and total
    total_dir = output_path / "total"
    total_dir.mkdir(exist_ok=True)
    
    for severity in range(1, 11):
        severity_dir = output_path / f"severity_{severity}"
        severity_dir.mkdir(exist_ok=True)
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Process each image in input directory
    input_path = Path(input_dir)
    
    # Copy original images to Bases directory
    bases_dir = output_path / "Bases"
    bases_dir.mkdir(exist_ok=True)
    
    kernel = None  # Will store the kernel for later saving
    
    for img_path in input_path.iterdir():
        if img_path.suffix.lower() in valid_extensions:
            # Copy original image to Bases directory
            shutil.copy2(img_path, bases_dir / img_path.name)
            
            # Read image
            image = cv2.imread(str(img_path))
            
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue
                
            # Process image with different convolution iterations
            for severity in range(1, 11):
                convolved_image, kernel = apply_iterative_convolution(image, severity)
                
                # Create output filename
                output_filename = f"{img_path.stem}_conv{severity}{img_path.suffix}"
                
                # Save to severity-specific directory
                severity_filepath = output_path / f"severity_{severity}" / output_filename
                cv2.imwrite(str(severity_filepath), convolved_image)
                
                # Save to total directory
                total_filepath = total_dir / output_filename
                cv2.imwrite(str(total_filepath), convolved_image)
            
            print(f"Processed: {img_path.name}")
    
    # Save the kernel visualization (only needs to be done once)
    if kernel is not None:
        save_kernel_visualization(kernel, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Apply iterative convolution to images with different severity levels')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('output_dir', help='Output directory for convolved images')
    parser.add_argument('--kernel_size', type=int, default=5, 
                        help='Size of the convolution kernel (default: 5)')
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir)
    print("Processing complete!")

if __name__ == "__main__":
    main() 