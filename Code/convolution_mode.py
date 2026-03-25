import os
import numpy as np
import time
import random
from PIL import Image
import tensorflow as tf

from kernels import load_kernels, KERNEL_TYPES
from utils import normalize_kernel
from convolute import apply_kernels, convolve_image
from control import sample_data

def run_convolution_mode(dataset_path, sample_percentage, num_kernels, grayscale, images_per_kernel=1):
    """
    Run the pipeline in convolution mode, which applies kernels to images
    and saves the convolved images with names in the format imX_kernelY.png
    
    Args:
        dataset_path: Path to the dataset containing images
        sample_percentage: Percentage of data to sample
        num_kernels: Number of distinct kernels to generate
        grayscale: Whether to process images in grayscale mode
        images_per_kernel: Number of images to process per kernel
    """
    print("Running in convolution mode...")
    print(f"Will process images from {dataset_path}")
    print(f"Using {num_kernels} kernels")
    print(f"Processing up to {images_per_kernel} images per kernel")
    print(f"Grayscale mode: {grayscale}")
    
    # Create output directories
    output_dir = os.path.join(os.getcwd(), "Conv_Mode")
    kernels_dir = os.path.join(output_dir, "Kernels")
    convolved_dir = os.path.join(output_dir, "Convolved_Images")
    base_dir = os.path.join(output_dir, "Base")
    os.makedirs(kernels_dir, exist_ok=True)
    os.makedirs(convolved_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    
    # Step 1: Generate kernels (same as in control.py)
    print("Step 1: Generating kernels...")
    kernels_list = generate_kernels(num_kernels, 21)
    
    # Normalize kernels if needed
    for i in range(len(kernels_list)):
        kernel, kernel_type = kernels_list[i]
        if not np.isclose(np.sum(kernel), 1.0, rtol=1e-5):
            kernel = normalize_kernel(kernel)
            kernels_list[i] = (kernel, kernel_type)
    
    print(f"Generated {len(kernels_list)} kernels")
    
    # Step 2: Save kernels and create mapping file
    print("Step 2: Saving kernels and creating mapping file...")
    kernel_mapping = {}
    
    # Step 3: Sample images from the dataset - use all available images up to images_per_kernel
    print("Step 3: Sampling images from the dataset...")
    all_image_paths = sample_data(dataset_path, sample_percentage)
    
    # Use all available images up to images_per_kernel
    sampled_image_paths = all_image_paths
    if len(sampled_image_paths) > images_per_kernel:
        sampled_image_paths = sampled_image_paths[:images_per_kernel]
    
    total_images = len(sampled_image_paths)
    print(f"Sampled {total_images} images")
    
    # Create a text file to map kernel numbers to kernel types with additional statistics
    mapping_file_path = os.path.join(kernels_dir, "kernel_mapping.txt")
    with open(mapping_file_path, 'w') as f:
        # Write header with general information
        f.write("CONVOLUTION MODE STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total kernels: {num_kernels}\n")
        f.write(f"Total images processed: {total_images}\n")
        f.write(f"Total convolutions: {total_images * num_kernels}\n")
        f.write(f"Grayscale mode: {grayscale}\n")
        f.write(f"Dataset path: {dataset_path}\n")
        f.write(f"Sample percentage: {sample_percentage}\n\n")
        
        # Write kernel-specific information
        f.write("KERNEL DETAILS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Kernel #':<10}{'Type':<20}{'Shape':<15}{'Min':<10}{'Max':<10}{'Mean':<10}{'Sum':<10}\n")
        f.write("-" * 80 + "\n")
        
        for i, (kernel, kernel_type) in enumerate(kernels_list):
            kernel_number = i + 1  # 1-based indexing for user-friendly display
            kernel_mapping[kernel_number] = kernel_type
            
            # Calculate kernel statistics
            kernel_shape = kernel.shape
            kernel_min = np.min(kernel)
            kernel_max = np.max(kernel)
            kernel_mean = np.mean(kernel)
            kernel_sum = np.sum(kernel)
            
            # Save kernel as numpy file
            kernel_path = os.path.join(kernels_dir, f"kernel_{kernel_number}.npy")
            np.save(kernel_path, kernel)
            
            # Save kernel as a simple grayscale image
            kernel_img_path = os.path.join(kernels_dir, f"kernel_{kernel_number}.png")
            # Normalize kernel to 0-255 range for image
            kernel_normalized = ((kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel)) * 255).astype(np.uint8)
            # Create a simple grayscale image from the kernel
            kernel_img = Image.fromarray(kernel_normalized, mode='L')
            kernel_img.save(kernel_img_path)
            
            # Write to mapping file
            f.write(f"{kernel_number:<10}{kernel_type:<20}{str(kernel_shape):<15}{kernel_min:<10.5f}{kernel_max:<10.5f}{kernel_mean:<10.5f}{kernel_sum:<10.5f}\n")
    
    print(f"Saved {len(kernels_list)} kernels and mapping file to {kernels_dir}")
    
    # Step 4: Save base images and apply kernels
    print("Step 4: Saving base images and applying kernels...")
    
    # Load and save base images
    loaded_images = []
    for img_idx, img_path in enumerate(sampled_image_paths):
        img_number = img_idx + 1  # 1-based indexing
        
        # Load the image
        img = Image.open(img_path)
        
        # Save the original image to the Base directory
        base_img_path = os.path.join(base_dir, f"im{img_number}_base.png")
        img.save(base_img_path)
        
        # Convert to grayscale if specified
        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        
        # Add to loaded images for processing
        loaded_images.append(np.array(img))
        
        print(f"Saved base image {img_number}/{len(sampled_image_paths)}")
    
    # Step 5: Apply kernels to images and save results
    print("Step 5: Applying kernels to images and saving results...")
    
    # Process each image with each kernel
    for img_idx, img_array in enumerate(loaded_images):
        img_number = img_idx + 1  # 1-based indexing
        print(f"Processing image {img_number}/{len(loaded_images)}...")
        
        # Apply each kernel to the current image
        for kernel_idx, (kernel, kernel_type) in enumerate(kernels_list):
            kernel_number = kernel_idx + 1  # 1-based indexing
            
            # Convert kernel to tensor and reshape for convolution
            kernel_tensor = tf.constant(kernel, dtype=tf.float32)
            kernel_tensor = tf.reshape(kernel_tensor, [kernel.shape[0], kernel.shape[1], 1, 1])
            
            # Apply convolution using the same function as in convolute.py
            # Add padding='SAME' parameter to ensure proper border handling
            convolved_img = convolve_image(img_array, kernel_tensor, grayscale=grayscale, padding='SAME')
            
            # Convert to numpy array if it's a tensor
            if hasattr(convolved_img, 'numpy'):
                convolved_img = convolved_img.numpy()
            
            # Ensure we have a numpy array
            if not isinstance(convolved_img, np.ndarray):
                convolved_img = np.array(convolved_img)
            
            # Convert to uint8 if not already
            if convolved_img.dtype != np.uint8:
                # Normalize to [0, 255] range
                convolved_img = ((convolved_img - np.min(convolved_img)) / 
                               (np.max(convolved_img) - np.min(convolved_img) + 1e-8) * 255).astype(np.uint8)
            
            # Save the convolved image
            output_filename = f"im{img_number}_kernel{kernel_number}.png"
            output_path = os.path.join(convolved_dir, output_filename)
            
            # Handle grayscale vs RGB images
            if convolved_img.ndim == 2 or (convolved_img.ndim == 3 and convolved_img.shape[-1] == 1):
                # Handle grayscale images
                if convolved_img.ndim == 3:
                    convolved_img = convolved_img.squeeze()  # Remove single channel dimension
                Image.fromarray(convolved_img, mode='L').save(output_path)
            else:
                # Handle RGB images
                if convolved_img.shape[-1] != 3:
                    print(f"Warning: Unexpected image shape {convolved_img.shape}, attempting to reshape")
                    try:
                        convolved_img = convolved_img.reshape(convolved_img.shape[0], convolved_img.shape[1], 3)
                    except ValueError as e:
                        print(f"Failed to reshape image: {e}")
                        # Fallback to grayscale
                        if convolved_img.ndim > 2:
                            convolved_img = convolved_img.squeeze()
                        Image.fromarray(convolved_img, mode='L').save(output_path)
                        continue
                Image.fromarray(convolved_img, mode='RGB').save(output_path)
    
    print(f"Convolution mode completed. Results saved to {output_dir}")

def generate_kernels(num_kernels, kernel_size):
    """
    Generate kernels using the same function as in control.py
    """
    all_kernels = load_kernels(num_kernels, kernel_size)
    if not all_kernels:
        raise ValueError("No kernels were loaded")
    
    # Take the first num_kernels kernels
    if len(all_kernels) < num_kernels:
        raise ValueError(f"Need at least {num_kernels} kernels, but only found {len(all_kernels)}")
    
    # Count unique kernels
    unique_kernels = set()
    for kernel, _ in all_kernels:
        kernel_hash = hash(kernel.tobytes())
        unique_kernels.add(kernel_hash)
    print(f"Generated {len(all_kernels)} kernels, {len(unique_kernels)} are unique")
    
    return all_kernels[:num_kernels]
