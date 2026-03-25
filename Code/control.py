import os
import argparse
import random
import numpy as np
import tensorflow as tf
import time
import json
from skimage.metrics import structural_similarity as ssim

from augmentation import perform_augmentation
from convolute import apply_kernels, convolve_image
from kernels import load_kernels
from training import train_neural_network, distributed_predict_step, kernel_mse_loss
from test_mode import run_test_mode
from PIL import Image
from utils import normalize_kernel
from utils import split_data

def sample_data(dataset_path, sample_percentage):
    print("Step 1: Sampling data...")
    all_files = sorted(
        [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    )  # Ensure consistent ordering
    if not all_files:
        raise ValueError(f"No images found in the dataset path: {dataset_path}")

    total_files = len(all_files)
    sample_size = max(1, int(total_files * abs(sample_percentage)))

    if sample_percentage > 0:
        # Take the first `sample_size` images
        sampled_data = all_files[:sample_size]
    elif sample_percentage < 0:
        # Take the last `sample_size` images
        sampled_data = all_files[-sample_size:]
    else:
        raise ValueError("Sample percentage must be non-zero.")

    return sampled_data

def augment_images(sampled_data, augmentation_rate, grayscale=False):
    print("Step 2: Data augmentation...")
    
    # If augmentation_rate is 0, skip augmentation and just load the original images
    if augmentation_rate == 0:
        print("Augmentation rate is 0, skipping augmentation...")
        augmented_data = []
        for image_path in sampled_data:
            # Load image
            img = Image.open(image_path)
            
            # Convert to grayscale if specified
            if grayscale:
                img = img.convert('L')
            else:
                img = img.convert('RGB')
                
            # Convert to numpy array
            augmented_data.append(np.array(img))
        
        print(f"Original data size: {len(augmented_data)}.")
        return augmented_data
    
    # Otherwise, perform normal augmentation
    augmented_data = perform_augmentation(sampled_data, augmentation_rate=augmentation_rate, grayscale=grayscale)
    print(f"Augmented data size: {len(augmented_data)}.")
    return augmented_data

def apply_kernels_to_batch(augmented_data, kernel, kernel_type, grayscale=False):
    """
    Apply the given `kernel` to the `augmented_data`.
    Returns (convolved_data, kernel, kernel_type) so we keep the same structure.
    """
    print("Step 3: Applying kernels...")
    result = apply_kernels(augmented_data, user_kernel=kernel, user_kernel_type=kernel_type, grayscale=grayscale)
    if result is None:
        raise ValueError("Kernel application failed")
    
    convolved_data, kernel, kernel_type = result
    print("Kernels applied.")
    return convolved_data, kernel, kernel_type

def train_model(convolved_data, kernel, strategy, current_model=None):
    print("Step 4: Training the neural network...")
    
    if current_model is not None:
        print("Continuing training with existing model...")
    
    trained_model = train_neural_network(convolved_data, kernel, strategy, current_model)
    
    if trained_model is None:
        raise RuntimeError("Training failed to return a valid model")
        
    print("Training step completed.")
    return trained_model

def calculate_kernel_similarity(original_kernel, predicted_kernel):
    """Calculate MSE and MNC between original and predicted kernel"""
    # Get the kernel size from the input kernel
    kernel_size = int(np.sqrt(original_kernel.size))
    
    # Reshape kernels to 2D if needed
    original = original_kernel.reshape(kernel_size, kernel_size)
    predicted = predicted_kernel.reshape(kernel_size, kernel_size)
    
    # Calculate MSE
    mse_score = np.mean((original - predicted) ** 2)
    
    # Calculate MNC (Mean Normalized Cross-correlation)
    # Normalize both kernels
    orig_norm = (original - np.mean(original)) / (np.std(original) + 1e-8)
    pred_norm = (predicted - np.mean(predicted)) / (np.std(predicted) + 1e-8)
    
    # Calculate cross-correlation
    mnc_score = np.mean(orig_norm * pred_norm)
    
    return float(mse_score), float(mnc_score)

def save_visualization_data(batch_index, batch_data, augmented_data, kernel, convolved_data, 
                          model_outputs, kernel_type, aug_time, conv_time, train_time, 
                          predicted_convolution, grayscale=False, val_augmented=None):
    start_time = time.time()
    
    # Define output directory
    output_dir = os.path.join(os.getcwd(), "visualization")
    os.makedirs(output_dir, exist_ok=True)
    batch_dir = os.path.join(output_dir, f"batch_{batch_index}")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Create kernels directory if it doesn't exist (only once)
    kernels_dir = os.path.join(output_dir, "kernels")
    os.makedirs(kernels_dir, exist_ok=True)
    
    # Save kernel only if it hasn't been saved before
    kernel_np = np.asarray(kernel[0] if isinstance(kernel, list) else kernel)
    if isinstance(kernel_np, tf.Tensor):
        kernel_np = kernel_np.numpy()
    
    kernel_size = int(np.sqrt(kernel_np.size))
    if kernel_np.ndim == 1:
        kernel_np = kernel_np.reshape(kernel_size, kernel_size)
    
    # Normalize kernel to [0, 1] range
    kernel_np = (kernel_np - np.min(kernel_np)) / (np.max(kernel_np) - np.min(kernel_np))
    
    # Save kernel only if it doesn't exist
    kernel_path = os.path.join(kernels_dir, f"kernel_{kernel_type}.png")
    kernel_array_path = os.path.join(kernels_dir, f"kernel_{kernel_type}.npy")
    
    if not os.path.exists(kernel_path):
        # Save as image
        kernel_img = (kernel_np * 255).astype(np.uint8)
        Image.fromarray(kernel_img, mode='L').save(kernel_path)
        # Save raw kernel data
        np.save(kernel_array_path, kernel_np)
    
    # Create validation subdirectory if validation data exists
    if val_augmented is not None:
        val_dir = os.path.join(batch_dir, "validation")
        os.makedirs(val_dir, exist_ok=True)

    # Convert model outputs to numpy array
    model_outputs = np.array(model_outputs)
    if isinstance(model_outputs[0], tf.Tensor):
        model_outputs = model_outputs.numpy()
    
    # Normalize model outputs
    model_outputs = model_outputs / (np.sum(model_outputs, axis=1, keepdims=True) + 1e-8)
    
    # Find best prediction by comparing MSE and MNC with original kernel
    best_mse = float('inf')
    best_mnc = -1
    best_output = None
    
    for output in model_outputs:
        output_reshaped = output.reshape(kernel_size, kernel_size)
        mse_score, mnc_score = calculate_kernel_similarity(kernel_np, output_reshaped)
        # Use MNC as primary metric (higher is better) and MSE as secondary (lower is better)
        if mnc_score > best_mnc or (mnc_score == best_mnc and mse_score < best_mse):
            best_mse = mse_score
            best_mnc = mnc_score
            best_output = output_reshaped
    
    # Use the best metrics and best output model
    mse_value = best_mse
    mnc_value = best_mnc
    model_output = best_output

    # Save training data
    # 1. Save the first original image in the batch
    first_image_path = batch_data[0]
    original_image = Image.open(first_image_path)
    original_image.save(os.path.join(batch_dir, "original.png"))

    # 2. Save augmented images (limit to 3 for simplicity)
    for i, aug_image in enumerate(augmented_data[:3]):
        aug_image_path = os.path.join(batch_dir, f"augmented_{i + 1}.png")
        Image.fromarray(aug_image).save(aug_image_path)

    # 3. Save the kernel
    kernel_array_path = os.path.join(batch_dir, "kernel.npy")
    # Normalize kernel to [0, 1] range
    kernel_np = (kernel_np - np.min(kernel_np)) / (np.max(kernel_np) - np.min(kernel_np))
    np.save(kernel_array_path, kernel_np)

    # 4. Save convolved data
    convolved_image = convolved_data[0].numpy() if isinstance(convolved_data[0], tf.Tensor) else convolved_data[0]
    if not isinstance(convolved_image, np.ndarray):
        convolved_image = np.array(convolved_image)
    if convolved_image.dtype != np.uint8:
        convolved_image = ((convolved_image - np.min(convolved_image)) / 
                         (np.max(convolved_image) - np.min(convolved_image)) * 255).astype(np.uint8)
    
    if convolved_image.ndim == 2 or (convolved_image.ndim == 3 and convolved_image.shape[-1] == 1):
        if convolved_image.ndim == 3:
            convolved_image = convolved_image.squeeze()
        Image.fromarray(convolved_image, mode='L').save(os.path.join(batch_dir, "convolved.png"))
    else:
        if convolved_image.shape[-1] != 3:
            convolved_image = convolved_image.reshape(convolved_image.shape[0], convolved_image.shape[1], 3)
        Image.fromarray(convolved_image, mode='RGB').save(os.path.join(batch_dir, "convolved.png"))
    
    np.save(os.path.join(batch_dir, "convolved.npy"), convolved_image)

    # 5. Save model output
    model_output_array_path = os.path.join(batch_dir, "model_output.npy")
    np.save(model_output_array_path, model_output)

    # Save validation data if available
    if val_augmented is not None and len(val_augmented) > 0:
        # Save validation images
        for i, val_image in enumerate(val_augmented[:3]):  # Save first 3 validation images
            val_image_path = os.path.join(val_dir, f"val_image_{i + 1}.png")
            Image.fromarray(val_image).save(val_image_path)
            np.save(os.path.join(val_dir, f"val_image_{i + 1}.npy"), val_image)
        
        # Save validation convolutions
        for i, val_conv in enumerate(convolved_data[:3]):  # Save first 3 validation convolutions
            val_conv = val_conv.numpy() if isinstance(val_conv, tf.Tensor) else val_conv
            if val_conv.dtype != np.uint8:
                val_conv = ((val_conv - np.min(val_conv)) / 
                          (np.max(val_conv) - np.min(val_conv)) * 255).astype(np.uint8)
            
            if val_conv.ndim == 2 or (val_conv.ndim == 3 and val_conv.shape[-1] == 1):
                if val_conv.ndim == 3:
                    val_conv = val_conv.squeeze()
                Image.fromarray(val_conv, mode='L').save(os.path.join(val_dir, f"val_convolved_{i + 1}.png"))
            else:
                Image.fromarray(val_conv, mode='RGB').save(os.path.join(val_dir, f"val_convolved_{i + 1}.png"))
            np.save(os.path.join(val_dir, f"val_convolved_{i + 1}.npy"), val_conv)
        
        # Save predicted kernels
        for i, pred_kernel in enumerate(predicted_convolution[:3]):  # Save first 3 predicted kernels
            pred_kernel = pred_kernel.numpy() if isinstance(pred_kernel, tf.Tensor) else pred_kernel
            
            # Ensure pred_kernel is 2D
            if pred_kernel.ndim == 1:
                kernel_size = int(np.sqrt(pred_kernel.size))
                pred_kernel = pred_kernel.reshape(kernel_size, kernel_size)
            
            # Normalize kernel to [0, 1] range for visualization
            pred_kernel_norm = (pred_kernel - np.min(pred_kernel)) / (np.max(pred_kernel) - np.min(pred_kernel))
            
            # Save as image (grayscale)
            pred_kernel_img = (pred_kernel_norm * 255).astype(np.uint8)
            Image.fromarray(pred_kernel_img, mode='L').save(os.path.join(val_dir, f"predicted_kernel_{i + 1}.png"))
            
            # Save raw kernel data
            np.save(os.path.join(val_dir, f"predicted_kernel_{i + 1}.npy"), pred_kernel)

    # Create batch info
    batch_info = {
        "batch_index": batch_index,
        "kernel_type": kernel_type,
        "original_images": len(batch_data),
        "augmented_images": len(augmented_data),
        "timing": {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "processing_time": None,
            "augmentation_time": f"{aug_time:.2f} seconds",
            "convolution_time": f"{conv_time:.2f} seconds",
            "training_time": f"{train_time:.2f} seconds"
        },
        "kernel_stats": {
            "shape": list(kernel_np.shape),
            "min": float(np.min(kernel_np)),
            "max": float(np.max(kernel_np)),
            "mean": float(np.mean(kernel_np)),
            "std": float(np.std(kernel_np)),
            "median": float(np.median(kernel_np)),
            "sum": float(np.sum(kernel_np))
        },
        "model_output_stats": {
            "shape": list(model_output.shape),
            "min": float(np.min(model_output)),
            "max": float(np.max(model_output)),
            "mean": float(np.mean(model_output)),
            "std": float(np.std(model_output))
        },
        "memory_usage": {
            "kernel_size": kernel_np.nbytes / (1024 * 1024),  # Size in MB
            "model_output_size": model_output.nbytes / (1024 * 1024)  # Size in MB
        },
        "kernel_comparison": {
            "mse": mse_value,
            "mnc": mnc_value
        }
    }
    
    # Update timing information
    batch_info["timing"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    batch_info["timing"]["processing_time"] = time.time() - start_time
    
    # Save batch info
    with open(os.path.join(batch_dir, "batch_info.json"), 'w') as f:
        json.dump(batch_info, f, indent=4)

    return batch_info

def update_global_statistics(visualization_dir, batch_info, aug_time, conv_time, train_time, validation_metrics=None):
    stats_file = os.path.join(visualization_dir, "global_statistics.json")
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = {
            "pipeline_info": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_update": None,
                "total_processing_time": 0,
                "average_batch_time": 0,
                "augmentation_time": 0,
                "convolution_time": 0,
                "training_time": 0
            },
            "total_batches": 0,
            "total_original_images": 0,
            "total_augmented_images": 0,
            "kernel_types": {},
            "kernel_type_batches": {},
            "kernel_stats": {
                "min_value_seen": float('inf'),
                "max_value_seen": float('-inf'),
                "avg_kernel_size": 0
            },
            "memory_usage": {
                "total_kernel_memory": 0,
                "total_output_memory": 0
            }
        }
    
    # Update basic statistics
    stats["total_batches"] += 1
    stats["total_original_images"] += batch_info["original_images"]
    stats["total_augmented_images"] += batch_info["augmented_images"]
    
    # Update kernel statistics
    stats["kernel_stats"].update({
        "min_value_seen": 0.0,
        "max_value_seen": 1.0,
        "avg_kernel_size": 32
    })
    
    # Update memory usage
    stats["memory_usage"]["total_kernel_memory"] += batch_info["memory_usage"]["kernel_size"]
    stats["memory_usage"]["total_output_memory"] += batch_info["memory_usage"]["model_output_size"]
    
    # Update kernel type statistics and batch tracking
    kernel_types = batch_info["kernel_types"]  # Now using kernel_types instead of kernel_type
    batch_number = batch_info["batch_index"]
    
    for kernel_type in kernel_types:
        if kernel_type not in stats["kernel_types"]:
            stats["kernel_types"][kernel_type] = 0
            stats["kernel_type_batches"][kernel_type] = []
        stats["kernel_types"][kernel_type] += 1
        
        if batch_number not in stats["kernel_type_batches"][kernel_type]:
            stats["kernel_type_batches"][kernel_type].append(batch_number)
            stats["kernel_type_batches"][kernel_type].sort()
    
    # Update timing information (store raw values)
    stats["pipeline_info"]["augmentation_time"] = aug_time
    stats["pipeline_info"]["convolution_time"] = conv_time
    stats["pipeline_info"]["training_time"] = train_time
    stats["pipeline_info"]["total_processing_time"] = aug_time + conv_time + train_time
    stats["pipeline_info"]["average_batch_time"] = stats["pipeline_info"]["total_processing_time"] / stats["total_batches"]
    stats["pipeline_info"]["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize metrics lists if they don't exist
    if "mse_values" not in stats:
        stats["mse_values"] = []
    if "mnc_values" not in stats:
        stats["mnc_values"] = []
    
    # Add current batch's MSE and MNC values if validation metrics are provided
    if validation_metrics:
        if validation_metrics["mse_loss"] is not None:
            stats["mse_values"].append({
                "batch": batch_info["batch_index"],
                "kernel_types": batch_info["kernel_types"],  # Updated to use kernel_types
                "mse": validation_metrics["mse_loss"]
            })
        if validation_metrics["mnc"] is not None:
            stats["mnc_values"].append({
                "batch": batch_info["batch_index"],
                "kernel_types": batch_info["kernel_types"],  # Updated to use kernel_types
                "mnc": validation_metrics["mnc"]
            })
    
    # Initialize validation_metrics list if it doesn't exist
    if "validation_metrics" not in stats:
        stats["validation_metrics"] = []
    
    # Add validation metrics if provided
    if validation_metrics:
        stats["validation_metrics"].append({
            "batch": batch_info["batch_index"],
            "kernel_types": batch_info["kernel_types"],  # Updated to use kernel_types
            "custom_loss": validation_metrics["custom_loss"],
            "loss_function": validation_metrics["loss_function"],
            "mse_loss": validation_metrics["mse_loss"],
            "mnc": validation_metrics["mnc"],
            "num_samples": validation_metrics["num_samples"]
        })
    
    # Save updated statistics
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

def generate_kernels(num_kernels):    
    all_kernels = load_kernels(num_kernels)
    if not all_kernels:
        raise ValueError("No kernels were loaded")
    
    # Modified code: Take the first num_kernels kernels
    if len(all_kernels) < num_kernels:
        raise ValueError(f"Need at least {num_kernels} kernels, but only found {len(all_kernels)}")
    
    # After generating kernels
    unique_kernels = set()
    for kernel, _ in all_kernels:
        kernel_hash = hash(kernel.tobytes())
        unique_kernels.add(kernel_hash)
    print(f"Generated {len(all_kernels)} kernels, {len(unique_kernels)} are unique")
    
    return all_kernels[:num_kernels]

def apply_round_robin_kernels_to_batch(images, kernels_list, grayscale=False):
    """
    For each image in the batch, pick a kernel in round-robin order
    and apply it to the image.
    Returns:
      - convolved_images: list of images after convolution.
      - kernel_assignments: list of tuples (kernel, kernel_type) used per image.
    """
    convolved_images = []
    kernel_assignments = []
    num_kernels = len(kernels_list)
    
    for idx, img in enumerate(images):
        # Round-robin selection: cycle through kernels regardless of batch size.
        kernel_idx = idx % num_kernels
        kernel, kernel_type = kernels_list[kernel_idx]
        
        # Wrap the single image in a list so apply_kernels processes it as a batch
        convolved_img_list, _, _ = apply_kernels([img], kernel, grayscale=grayscale)
        # Extract the result (the first element, since we passed a single image)
        convolved_img = convolved_img_list[0]
        
        convolved_images.append(convolved_img)
        kernel_assignments.append((kernel, kernel_type))
        
    return convolved_images, kernel_assignments

def evaluate_validation_metrics(model, val_convolved, val_kernel_assignments, strategy, batch_size=16, loss_function='mse'):
    """
    Evaluate model performance on validation data using the specified loss function.
    
    Args:
        model: The trained model
        val_convolved: List of validation images after convolution
        val_kernel_assignments: List of (kernel, kernel_type) tuples for validation data
        strategy: TensorFlow distribution strategy
        batch_size: Batch size for evaluation
        loss_function: Name of the loss function to use (must be a key in LOSS_FUNCTIONS)
        
    Returns:
        Dictionary with validation metrics
    """
    # Extract kernels from assignments
    val_kernels = [k for k, _ in val_kernel_assignments]
    
    # Get the kernel size from the first kernel
    kernel_size = int(np.sqrt(val_kernels[0].size))
    
    # Convert validation data to tensors
    val_images = np.array(val_convolved).astype(np.float32)
    val_kernels = np.array([k.flatten() for k in val_kernels]).astype(np.float32)
    
    # Create TensorFlow datasets
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_kernels))
    val_dataset = val_dataset.batch(batch_size)
    
    # Get the loss function
    from training import LOSS_FUNCTIONS
    if loss_function not in LOSS_FUNCTIONS:
        print(f"Warning: Loss function '{loss_function}' not found. Using MSE loss instead.")
        loss_function = 'mse'
    
    loss_fn = LOSS_FUNCTIONS[loss_function]
    
    # Run predictions on CPU
    all_predictions = []
    all_targets = []
    
    with tf.device('/cpu:0'):
        for x_batch, y_batch in val_dataset:
            # Get predictions
            pred_batch = model(x_batch, training=False)
            all_predictions.append(pred_batch.numpy())
            all_targets.append(y_batch.numpy())
    
    if all_predictions:
        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # Calculate loss using the specified loss function
        with tf.device('/cpu:0'):
            # Convert to tensors
            pred_tensor = tf.convert_to_tensor(all_predictions)
            target_tensor = tf.convert_to_tensor(all_targets)
            # Calculate loss
            custom_loss = loss_fn(target_tensor, pred_tensor).numpy()
        
        # Calculate MSE and MNC for each prediction
        mse_scores = []
        mnc_scores = []
        
        for i in range(len(all_predictions)):
            true_kernel = all_targets[i].reshape(kernel_size, kernel_size)
            pred_kernel = all_predictions[i].reshape(kernel_size, kernel_size)
            
            # Calculate MSE
            mse = np.mean((true_kernel - pred_kernel) ** 2)
            mse_scores.append(mse)
            
            # Calculate MNC
            # Normalize both kernels
            true_norm = (true_kernel - np.mean(true_kernel)) / (np.std(true_kernel) + 1e-8)
            pred_norm = (pred_kernel - np.mean(pred_kernel)) / (np.std(pred_kernel) + 1e-8)
            mnc = np.mean(true_norm * pred_norm)
            mnc_scores.append(mnc)
        
        # Calculate averages
        avg_mse = np.mean(mse_scores)
        avg_mnc = np.mean(mnc_scores)
        
        return {
            "custom_loss": float(custom_loss),
            "loss_function": loss_function,
            "mse_loss": float(avg_mse),
            "mnc": float(avg_mnc),
            "num_samples": len(all_predictions)
        }
    
    return {
        "custom_loss": None,
        "loss_function": loss_function,
        "mse_loss": None,
        "mnc": None,
        "num_samples": 0
    }

def main(dataset_path, sample_percentage, batches, augmentation_rate, test_mode, 
         num_kernels, grayscale, epochs, learning_rate, loss_function, conv_mode, images_per_kernel=1):
    
    # If convolution mode is enabled, call the main function from convolution_mode.py
    if conv_mode:
        from convolution_mode import run_convolution_mode
        print("Running in convolution mode...")
        run_convolution_mode(dataset_path, sample_percentage, num_kernels, grayscale, images_per_kernel)
        return
    
    print(f"Starting pipeline with loss function: {loss_function}...")
    total_start_time = time.time()

    # --- Step 1: Sample Data ---
    start_time = time.time()
    sampled_data = sample_data(dataset_path, sample_percentage)
    print(f"Step 1 - Sample data: {len(sampled_data)} images. Time taken: {time.time() - start_time:.2f} sec")

    # --- Step 2: Split Data into Training and Validation ---
    train_data, val_data = split_data(sampled_data, validation_split=0.2)
    print(f"Data split: {len(train_data)} training images, {len(val_data)} validation images")

    # --- Step 3: Create Training Batches ---
    start_time = time.time()
    total_samples = len(train_data)
    batch_size = max(1, total_samples // batches)
    train_batches = []
    for i in range(batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        if start_idx < end_idx:
            train_batches.append(train_data[start_idx:end_idx])
    print(f"Created {len(train_batches)} training batches (~{batch_size} images each)")

    # --- Initialize model & distribution strategy ---
    strategy = tf.distribute.MirroredStrategy()
    model = None

    # --- Generate and (if needed) normalize kernels ---
    kernels_list = generate_kernels(num_kernels)
    for i in range(len(kernels_list)):
        kernel, kernel_type = kernels_list[i]
        if not np.isclose(np.sum(kernel), 1.0, rtol=1e-5):
            kernel = normalize_kernel(kernel)
            kernels_list[i] = (kernel, kernel_type)

    # --- Timing Accumulators ---
    total_augmentation_time = 0.0
    total_convolution_time = 0.0
    total_training_time = 0.0
    
    # --- Validation Metrics History ---
    validation_metrics_history = []

    # --- Process each Training Batch ---
    for batch_index, batch_data in enumerate(train_batches):
        print(f"\nProcessing training batch {batch_index + 1}/{len(train_batches)}...")

        # Augmentation
        start_time = time.time()
        augmented_data = augment_images(batch_data, augmentation_rate, grayscale=grayscale)
        aug_time = time.time() - start_time
        total_augmentation_time += aug_time

        # Apply round-robin kernel to each image in the batch
        start_time = time.time()
        convolved_data, kernel_assignments = apply_round_robin_kernels_to_batch(
            augmented_data, kernels_list, grayscale=grayscale
        )
        conv_time = time.time() - start_time
        total_convolution_time += conv_time

        # Training
        start_time = time.time()
        model, _ = train_neural_network(convolved_data, kernel_assignments, strategy, model,
                                epochs=epochs, learning_rate=learning_rate, batch_size=16,
                                loss_function=loss_function)
        train_time = time.time() - start_time
        total_training_time += train_time

        # --- Evaluate on Validation Set ---
        validation_metrics = None
        if model is not None:
            print("Evaluating on validation set...")
            # Ensure we have at least one sample per kernel type
            unique_kernel_types = sorted({kt for _, kt in kernels_list})
            samples_per_kernel_type = 1  # At least one sample per kernel type
            num_val_samples = samples_per_kernel_type * len(unique_kernel_types)
            
            # Select validation data
            selected_val_data = random.sample(val_data, min(num_val_samples, len(val_data)))
            
            # Augment and apply round-robin kernels to validation images
            val_augmented = augment_images(selected_val_data, augmentation_rate, grayscale=grayscale)
            val_convolved, val_kernel_assignments = apply_round_robin_kernels_to_batch(
                val_augmented, kernels_list, grayscale=grayscale
            )
            
            # Get model predictions on validation data
            val_images = np.array(val_convolved).astype(np.float32)
            with tf.device('/cpu:0'):
                val_predictions = model.predict(val_images, batch_size=batch_size)
            
            # Evaluate validation metrics using the specified loss function
            validation_metrics = evaluate_validation_metrics(
                model, val_convolved, val_kernel_assignments, strategy, 
                batch_size=batch_size, loss_function=loss_function
            )
            validation_metrics_history.append(validation_metrics)
            print(f"Validation {loss_function.upper()} Loss: {validation_metrics['custom_loss']:.6f}, MSE Loss: {validation_metrics['mse_loss']:.6f}, MNC: {validation_metrics['mnc']:.4f}")
            
            # For visualization, save all kernel types used in this batch
            batch_dir = os.path.join(os.getcwd(), "visualization", f"batch_{batch_index}")
            os.makedirs(batch_dir, exist_ok=True)
            
            # Save original kernels used in this batch
            kernels_dir = os.path.join(batch_dir, "original_kernels")
            os.makedirs(kernels_dir, exist_ok=True)
            
            # Save each unique kernel type used in this batch
            for kernel, kernel_type in val_kernel_assignments:
                kernel_np = np.array(kernel)
                # Normalize kernel to [0, 1] range for visualization
                kernel_norm = (kernel_np - np.min(kernel_np)) / (np.max(kernel_np) - np.min(kernel_np))
                
                # Save as image
                kernel_img = (kernel_norm * 255).astype(np.uint8)
                Image.fromarray(kernel_img, mode='L').save(
                    os.path.join(kernels_dir, f"kernel_{kernel_type}.png")
                )
                
                # Save raw kernel data
                np.save(
                    os.path.join(kernels_dir, f"kernel_{kernel_type}.npy"),
                    kernel_np
                )
            
            # Create batch info for visualization
            batch_info = {
                "batch_index": batch_index,
                "kernel_types": list({kt for _, kt in val_kernel_assignments}),
                "original_images": len(batch_data),
                "augmented_images": len(augmented_data),
                "timing": {
                    "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "processing_time": aug_time + conv_time + train_time,
                    "augmentation_time": f"{aug_time:.2f} seconds",
                    "convolution_time": f"{conv_time:.2f} seconds",
                    "training_time": f"{train_time:.2f} seconds"
                },
                "kernel_stats": {
                    "num_kernel_types": len(unique_kernel_types),
                    "kernel_types_used": list({kt for _, kt in val_kernel_assignments})
                },
                "memory_usage": {
                    "kernel_size": sum(np.array(k).nbytes for k, _ in val_kernel_assignments) / (1024 * 1024),  # Size in MB
                    "model_output_size": val_predictions[0].nbytes / (1024 * 1024)  # Size in MB
                },
                "loss_function": loss_function
            }
            
            save_visualization_data(
                batch_index, batch_data, val_augmented, val_kernel_assignments[0][0],  # Use first kernel as reference
                val_convolved, val_predictions, val_kernel_assignments[0][1],  # Use first kernel type as reference
                aug_time, conv_time, train_time,
                val_predictions, grayscale=grayscale, val_augmented=val_augmented
            )
            
            # Update global statistics with validation metrics
            update_global_statistics(
                os.path.join(os.getcwd(), "visualization"), 
                batch_info, aug_time, conv_time, train_time,
                validation_metrics
            )

    print("\nTraining completed.")
    print(f"Total batches processed: {len(train_batches)}")
    print(f"Total training time: {total_training_time:.2f} sec")

    # --- Test Mode (if enabled) ---
    if test_mode:
        print("Running test mode...")
        sampled_test_data = sample_data(dataset_path, -sample_percentage)
        run_test_mode(sampled_data, sampled_test_data, augmentation_rate, model)

    print(f"\nPipeline finished. Total time: {time.time() - total_start_time:.2f} sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serial Pipeline Control Script")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--sample-percentage", type=float, default=0.01, help="Percentage of data to sample")
    parser.add_argument("--batches", type=int, default=1, help="Number of batches to process")
    parser.add_argument("--augmentation-rate", type=int, choices=[0, 1, 2, 3], default=1, help="Augmentation expansion rate: 0=None, 1=Moderate, 2=Significant, 3=Aggressive")
    parser.add_argument("--test", action="store_true", help="Enable test mode to save intermediate outputs for a single image")
    parser.add_argument("--num-kernels", type=int, default=1, help="Number of distinct kernels to generate and rotate through across all batches")
    parser.add_argument("--grayscale", action="store_true", help="Process images in grayscale mode")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--loss-function", type=str, 
                        choices=['mse', 'l1', 'huber', 'kl', 'wasserstein', 'gradient', 'ms_ssim', 'hybrid', 'mse_mnc'], 
                        default='mse_mnc', help="Loss function to use for training")
    parser.add_argument("--conv-mode", action="store_true", help="Enable convolution mode to save convolved images")
    parser.add_argument("--images-per-kernel", type=int, default=5, help="Number of images to process per kernel in convolution mode")

    args = parser.parse_args()

    main(args.dataset_path, args.sample_percentage, args.batches, args.augmentation_rate, args.test, 
         args.num_kernels, args.grayscale, args.epochs, args.learning_rate, args.loss_function, 
         args.conv_mode, args.images_per_kernel)
