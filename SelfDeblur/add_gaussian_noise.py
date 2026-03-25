import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import shutil

def add_gaussian_noise(image, severity):
    """
    Add Gaussian noise to an image based on severity level (1-10)
    
    Args:
        image: Input image (numpy array)
        severity: Integer from 1 to 10 indicating noise level
    
    Returns:
        Noisy image
    """
    # Convert to grayscale if image is RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert severity (1-10) to standard deviation (0-50)
    std = (severity - 1) * 5.5  # This maps severity 1->0, 10->49.5
    
    # Generate Gaussian noise
    gaussian_noise = np.random.normal(0, std, image.shape).astype(np.float32)
    
    # Add noise to image
    noisy_image = cv2.add(image.astype(np.float32), gaussian_noise)
    
    # Clip values to valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def process_directory(input_dir, output_dir):
    """
    Process all images in input directory and create noisy versions
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
    for img_path in input_path.iterdir():
        if img_path.suffix.lower() in valid_extensions:
            # Read image
            image = cv2.imread(str(img_path))
            
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue
                
            # Process image with different noise levels
            for severity in range(1, 11):
                noisy_image = add_gaussian_noise(image, severity)
                
                # Create output filename
                output_filename = f"{img_path.stem}_noise{severity}{img_path.suffix}"
                
                # Save to severity-specific directory
                severity_filepath = output_path / f"severity_{severity}" / output_filename
                cv2.imwrite(str(severity_filepath), noisy_image)
                
                # Save to total directory
                total_filepath = total_dir / output_filename
                cv2.imwrite(str(total_filepath), noisy_image)
            
            print(f"Processed: {img_path.name}")

def main():
    parser = argparse.ArgumentParser(description='Add Gaussian noise to images with different severity levels')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('output_dir', help='Output directory for noisy images')
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir)
    print("Processing complete!")

if __name__ == "__main__":
    main() 