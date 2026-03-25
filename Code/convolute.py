import tensorflow as tf
import numpy as np
import random
from utils import normalize_kernel

def apply_kernels(augmented_data, user_kernel=None, user_kernel_type=None, grayscale=False):
    strategy = tf.distribute.MirroredStrategy()
    # print(f"Applying kernels to augmented data of size {len(augmented_data)} using TensorFlow with {strategy.num_replicas_in_sync} GPUs...")

    convolved_data = []

    with strategy.scope():
        if user_kernel is None:
            print("No kernel provided by previous step. Exiting.")
            return
        else:
            kernel = user_kernel
            kernel_type = user_kernel_type

        # Process generated kernel
        kernel_tensor = tf.constant(kernel, dtype=tf.float32)
        # Normalize kernel to sum to 1 (same as predicted kernel)
        kernel_tensor = kernel_tensor / (tf.reduce_sum(kernel_tensor) + tf.keras.backend.epsilon())
        kernel_tensor = tf.reshape(kernel_tensor, [kernel.shape[0], kernel.shape[1], 1, 1])

        # Apply the same kernel to all images in the batch
        for image in augmented_data:
            convolved_image = convolve_image(image, kernel_tensor, grayscale=grayscale)
            convolved_data.append(convolved_image)

    return convolved_data, kernel, kernel_type

def convolve_image(image, kernel, grayscale=False, padding='SAME'):
    """
    Apply convolution to an image using the given kernel
    
    Args:
        image: Input image as numpy array
        kernel: Convolution kernel as tensor
        grayscale: Whether the image is grayscale
        padding: Padding method ('SAME' or 'VALID')
    
    Returns:
        Convolved image
    """
    # Convert image to tensor if it's not already
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Reshape image for convolution
    if grayscale:
        # For grayscale, add channel dimension
        if len(image.shape) == 2:
            image = tf.expand_dims(image, axis=-1)
        # Add batch dimension
        image = tf.expand_dims(image, axis=0)
    else:
        # For RGB, ensure we have 3 channels
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = tf.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            # Expand single channel to RGB
            image = tf.concat([image, image, image], axis=-1)
        # Add batch dimension
        image = tf.expand_dims(image, axis=0)
    
    # Apply convolution with specified padding
    if grayscale:
        # For grayscale images
        result = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding=padding)
    else:
        # For RGB images, apply to each channel separately
        channels = tf.unstack(image, axis=-1)
        convolved_channels = []
        for channel in channels:
            channel = tf.expand_dims(channel, axis=-1)
            convolved = tf.nn.conv2d(channel, kernel, strides=[1, 1, 1, 1], padding=padding)
            convolved_channels.append(tf.squeeze(convolved, axis=-1))
        result = tf.stack(convolved_channels, axis=-1)
    
    # Remove batch dimension
    result = tf.squeeze(result, axis=0)
    
    return result
