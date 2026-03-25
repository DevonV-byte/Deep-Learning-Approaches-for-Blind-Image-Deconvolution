import os
import random
import numpy as np
import tensorflow as tf

from augmentation import perform_augmentation
from convolute import apply_kernels
from PIL import Image

def run_test_mode(sampled_data, sampled_test_data, augmentation_rate, model):
    # Select 10 random images for testing
    test_size = 500
    test_image_paths = random.sample(sampled_test_data, test_size)  # Choose 200 random images
    print(f"Selected test images: {test_image_paths}")

    kernels = []  # Store kernels for uniqueness check
    augmented_data_kernels = {}  # Dictionary to store augmented data and their kernels

    for i, test_image_path in enumerate(test_image_paths):
        print(f"Processing test image {i + 1}/{len(test_image_paths)}: {test_image_path}")

        # Augment the test image
        augmented_test_data = perform_augmentation([test_image_path], augmentation_rate)

        # Apply kernels to the augmented test data
        kernel_image_pairs = apply_kernels(augmented_test_data)

        # Extract the kernel (it’s the same for all pairs, so we take the first one)
        kernel = kernel_image_pairs[0][0]  # Extract the kernel from the first pair
        kernels.append(kernel)  # Add kernel to the list for uniqueness check

        # Add augmented data and kernel to the dictionary
        augmented_data_kernels[tuple(kernel.flatten())] = [image for _, image in kernel_image_pairs]  # Convert kernel to tuple for hashable dictionary key

        if test_size <= 10:
            # Create a subdirectory for each image's output
            output_dir = os.path.join("test_output", f"image_{i + 1}")
            os.makedirs(output_dir, exist_ok=True)

            # Save outputs for the current test image: should be turned off for big test sizes
            save_test_output(test_image_path, augmented_test_data, kernel_image_pairs, kernel, output_dir)

    # Check if all kernels are unique
    are_kernels_unique = check_kernels_uniqueness(kernels)
    if not are_kernels_unique:
        print("Warning: Some kernels are not unique!")

    # Check if the data for testing is different than the data for training
    check_data_difference(sampled_data, test_image_paths)

    # Evaluate the model
    evaluation_results = evaluate_model(model, augmented_data_kernels)
    print(f"Model Evaluation Results: {evaluation_results}")

    print("Test mode completed. Outputs saved.")


def save_test_output(base_image_path, augmented_data, kernel_image_pairs, kernel, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Save the base image
    base_image = Image.open(base_image_path)
    base_image.save(os.path.join(output_dir, "base_image.png"))

    # Save augmented images
    for i, aug_image in enumerate(augmented_data):
        try:
            if isinstance(aug_image, np.ndarray):
                aug_image = Image.fromarray(aug_image)
            aug_image.save(os.path.join(output_dir, f"augmented_image_{i + 1}.png"))
        except Exception as e:
            print(f"Skipping invalid augmented image at index {i}: {e}")
    # Save the single kernel
    kernel_path = os.path.join(output_dir, "kernel.txt")
    np.savetxt(kernel_path, kernel.reshape(32, 32), fmt="%.4f")  # Assuming 32x32 kernel size

    # Save convolved images
    for i, (_, convolved_image) in enumerate(kernel_image_pairs):
        try:
            if isinstance(convolved_image, np.ndarray):
                convolved_image = Image.fromarray(convolved_image)
            convolved_image.save(os.path.join(output_dir, f"convolved_image_{i + 1}.png"))
        except Exception as e:
            print(f"Skipping invalid convolved image at index {i}: {e}")

def check_kernels_uniqueness(kernel_list):
    print("Checking if all kernels are unique...")
    kernel_set = set()  # To store unique kernels as flattened arrays
    
    for i, kernel in enumerate(kernel_list):
        # Convert the kernel to a tuple (hashable) to store in the set
        kernel_tuple = tuple(kernel.flatten())
        if kernel_tuple in kernel_set:
            print(f"Duplicate kernel found at index {i + 1}.")
            return False
        kernel_set.add(kernel_tuple)

    print("All kernels are unique.")
    return True

def evaluate_model(model, augmented_data_kernels):
    print("Evaluating model...")

    # Prepare input (images) and output (kernels) data
    x_data = []
    y_data = []

    for kernel, images in augmented_data_kernels.items():
        for image in images:
            x_data.append(image)  # Input: augmented image
            y_data.append(kernel)  # Output: associated kernel

    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)

    # Reshape images if necessary (e.g., adding a channel dimension for grayscale images)
    if len(x_data.shape) == 3:  # If grayscale and no channel dimension
        x_data = np.expand_dims(x_data, axis=-1)

    # Use the model to predict kernels
    predictions = model.predict(x_data)

    # Calculate evaluation metrics (e.g., Mean Squared Error)
    mse = np.mean((predictions - y_data) ** 2)  # Mean Squared Error
    print(f"Model Evaluation: Mean Squared Error = {mse:.6f}")

    # Return evaluation metrics
    return {"mse": mse}

def check_data_difference(sampled_data, sampled_test_data):
    print("Checking for overlap between sampled training and test data...")
    overlap = set(sampled_data).intersection(set(sampled_test_data))

    if overlap:
        print(f"Warning: There are {len(overlap)} overlapping images between training and test data.")
        print(f"Overlapping images: {list(overlap)[:10]}...")  # Display the first 10 overlaps for debugging
    else:
        print("No overlap found between training and test data.")

def evaluate_kernel_functionality(original_kernel, predicted_kernel, test_image):
    """
    Evaluate how similarly the predicted kernel behaves compared to the original
    """
    # Apply both kernels to the same test image
    original_result = convolve_image(test_image, original_kernel)
    predicted_result = convolve_image(test_image, predicted_kernel)
    
    # Calculate SSIM between the two convolution results
    ssim = tf.image.ssim(
        tf.convert_to_tensor(original_result, dtype=tf.float32),
        tf.convert_to_tensor(predicted_result, dtype=tf.float32),
        max_val=255.0
    )
    
    # Calculate normalized cross-correlation
    orig_norm = (original_result - np.mean(original_result)) / np.std(original_result)
    pred_norm = (predicted_result - np.mean(predicted_result)) / np.std(predicted_result)
    ncc = np.mean(orig_norm * pred_norm)
    
    return {
        'ssim': float(ssim.numpy()),
        'ncc': float(ncc),
        'cosine_similarity': float(tf.keras.losses.cosine_similarity(
            original_kernel.flatten(),
            predicted_kernel.flatten()
        ).numpy())
    }
