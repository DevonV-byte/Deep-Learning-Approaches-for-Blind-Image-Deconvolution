import tensorflow as tf
import numpy as np
from utils import normalize_kernel

# ------------------------- Custom Losses / Metrics -------------------------- #

def kernel_2d_loss(y_true, y_pred, alpha=0.5):
    """
    2D-based hybrid loss:
      Loss = alpha * MSE_2D + (1 - alpha) * (1 - SSIM)
    y_true, y_pred: shape (batch_size, 32*32)
    """
    # Reshape flattened kernel vectors into 2D (and add a channel dimension)
    y_true_2d = tf.reshape(y_true, [-1, 32, 32, 1])
    y_pred_2d = tf.reshape(y_pred, [-1, 32, 32, 1])

    # 1) MSE in 2D
    mse_2d = tf.reduce_mean(tf.square(y_true_2d - y_pred_2d))

    # 2) SSIM (structural similarity), range ~ [0..1] => higher is better
    ssim_val = tf.image.ssim(y_true_2d, y_pred_2d, max_val=1.0)
    ssim_mean = tf.reduce_mean(ssim_val)

    # Hybrid combination: minimize (MSE) + (1 - SSIM)
    # Weighted by alpha for MSE, (1 - alpha) for SSIM portion
    loss = alpha * mse_2d + (1 - alpha) * (1.0 - ssim_mean)
    return loss

def kernel_mse_loss(y_true, y_pred):
    """
    Pure MSE loss for 2D kernels.
    y_true, y_pred: shape (batch_size, 32*32)
    """
    # Reshape flattened kernel vectors into 2D (and add a channel dimension)
    y_true_2d = tf.reshape(y_true, [-1, 32, 32, 1])
    y_pred_2d = tf.reshape(y_pred, [-1, 32, 32, 1])

    # MSE in 2D
    mse_2d = tf.reduce_mean(tf.square(y_true_2d - y_pred_2d))
    return mse_2d

def kernel_l1_loss(y_true, y_pred):
    """
    L1 Loss (Mean Absolute Error) for 2D kernels.
    y_true, y_pred: shape (batch_size, 32*32)
    """
    # Reshape flattened kernel vectors into 2D (and add a channel dimension)
    y_true_2d = tf.reshape(y_true, [-1, 32, 32, 1])
    y_pred_2d = tf.reshape(y_pred, [-1, 32, 32, 1])
    
    # L1 loss in 2D
    l1_2d = tf.reduce_mean(tf.abs(y_true_2d - y_pred_2d))
    return l1_2d

def kernel_huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber Loss for 2D kernels.
    y_true, y_pred: shape (batch_size, 32*32)
    delta: threshold where the loss changes from quadratic to linear
    """
    # Reshape flattened kernel vectors into 2D (and add a channel dimension)
    y_true_2d = tf.reshape(y_true, [-1, 32, 32, 1])
    y_pred_2d = tf.reshape(y_pred, [-1, 32, 32, 1])
    
    # Huber loss in 2D
    huber_2d = tf.keras.losses.Huber(delta=delta)(y_true_2d, y_pred_2d)
    return huber_2d

def kernel_kl_divergence(y_true, y_pred, epsilon=1e-8):
    """
    KL Divergence for 2D kernels.
    y_true, y_pred: shape (batch_size, 32*32)
    """
    # Reshape flattened kernel vectors into 2D
    y_true_2d = tf.reshape(y_true, [-1, 32, 32])
    y_pred_2d = tf.reshape(y_pred, [-1, 32, 32])
    
    # Normalize to ensure they sum to 1 (probability distributions)
    y_true_norm = y_true_2d / (tf.reduce_sum(y_true_2d, axis=[1, 2], keepdims=True) + epsilon)
    y_pred_norm = y_pred_2d / (tf.reduce_sum(y_pred_2d, axis=[1, 2], keepdims=True) + epsilon)
    
    # Add small epsilon to avoid log(0)
    y_true_norm = y_true_norm + epsilon
    y_pred_norm = y_pred_norm + epsilon
    
    # KL divergence: sum(p * log(p/q))
    kl_div = tf.reduce_mean(
        tf.reduce_sum(
            y_true_norm * tf.math.log(y_true_norm / y_pred_norm), 
            axis=[1, 2]
        )
    )
    return kl_div

def kernel_wasserstein_distance(y_true, y_pred):
    """
    Earth Mover's Distance (Wasserstein Distance) for 2D kernels.
    This is a simplified approximation as the true Wasserstein distance
    requires solving an optimal transport problem.
    
    y_true, y_pred: shape (batch_size, 32*32)
    """
    # Reshape flattened kernel vectors into 2D
    y_true_2d = tf.reshape(y_true, [-1, 32, 32])
    y_pred_2d = tf.reshape(y_pred, [-1, 32, 32])
    
    # Sort values (simplified approximation)
    y_true_sorted = tf.sort(tf.reshape(y_true_2d, [-1, 32*32]), axis=1)
    y_pred_sorted = tf.sort(tf.reshape(y_pred_2d, [-1, 32*32]), axis=1)
    
    # L1 distance between sorted distributions
    wass_dist = tf.reduce_mean(tf.abs(y_true_sorted - y_pred_sorted))
    return wass_dist

def kernel_gradient_loss(y_true, y_pred):
    """
    Edge-Aware or Gradient Loss for 2D kernels.
    Computes the difference in gradients between true and predicted kernels.
    
    y_true, y_pred: shape (batch_size, 32*32)
    """
    # Reshape flattened kernel vectors into 2D (and add a channel dimension)
    y_true_2d = tf.reshape(y_true, [-1, 32, 32, 1])
    y_pred_2d = tf.reshape(y_pred, [-1, 32, 32, 1])
    
    # Compute gradients using Sobel filters
    def compute_gradients(image):
        sobel_x = tf.image.sobel_edges(image)[:, :, :, :, 0]
        sobel_y = tf.image.sobel_edges(image)[:, :, :, :, 1]
        return sobel_x, sobel_y
    
    true_grad_x, true_grad_y = compute_gradients(y_true_2d)
    pred_grad_x, pred_grad_y = compute_gradients(y_pred_2d)
    
    # Compute L1 loss on gradients
    grad_loss_x = tf.reduce_mean(tf.abs(true_grad_x - pred_grad_x))
    grad_loss_y = tf.reduce_mean(tf.abs(true_grad_y - pred_grad_y))
    
    return grad_loss_x + grad_loss_y

def kernel_ms_ssim_loss(y_true, y_pred):
    """
    Multi-Scale SSIM Loss for 2D kernels.
    Upsamples the kernels to 128x128 before computing MS-SSIM.
    
    y_true, y_pred: shape (batch_size, 32*32)
    """
    # Reshape flattened kernel vectors into 2D (and add a channel dimension)
    y_true_2d = tf.reshape(y_true, [-1, 32, 32, 1])
    y_pred_2d = tf.reshape(y_pred, [-1, 32, 32, 1])
    
    # Upsample kernels to 128x128 to allow for multi-scale analysis
    # MS-SSIM requires larger images to compute meaningful statistics at multiple scales
    y_true_upsampled = tf.image.resize(y_true_2d, [128, 128], method='bilinear')
    y_pred_upsampled = tf.image.resize(y_pred_2d, [128, 128], method='bilinear')
    
    # Compute MS-SSIM on upsampled kernels
    ms_ssim_val = tf.image.ssim_multiscale(y_true_upsampled, y_pred_upsampled, max_val=1.0)
    ms_ssim_mean = tf.reduce_mean(ms_ssim_val)
    
    # Convert to a loss (1 - MS_SSIM)
    return 1.0 - ms_ssim_mean

def kernel_mse_mnc_loss(y_true, y_pred):
    """
    Combined loss function using MSE and MNC (Mean Normalized Cross-correlation).
    MNC is inverted (1 - MNC) since we want to minimize the loss.
    
    Args:
        y_true: True kernel values
        y_pred: Predicted kernel values
        
    Returns:
        Combined loss value
    """
    # Get the batch size
    batch_size = tf.shape(y_true)[0]
    
    # Reshape to 2D kernels if needed
    if len(y_true.shape) == 2:
        # Calculate the square root of the flattened dimension
        dim = tf.cast(tf.sqrt(tf.cast(tf.shape(y_true)[1], tf.float32)), tf.int32)
        y_true = tf.reshape(y_true, [batch_size, dim, dim])
        y_pred = tf.reshape(y_pred, [batch_size, dim, dim])
    
    # Calculate MSE
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Calculate MNC
    # Normalize both kernels
    y_true_mean = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
    y_pred_mean = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)
    y_true_std = tf.math.reduce_std(y_true, axis=[1, 2], keepdims=True) + 1e-8
    y_pred_std = tf.math.reduce_std(y_pred, axis=[1, 2], keepdims=True) + 1e-8
    
    y_true_norm = (y_true - y_true_mean) / y_true_std
    y_pred_norm = (y_pred - y_pred_mean) / y_pred_std
    
    # Calculate cross-correlation
    mnc = tf.reduce_mean(y_true_norm * y_pred_norm)
    
    # Invert MNC since we want to minimize the loss
    mnc_loss = 1.0 - mnc
    
    # Combine losses (equal weighting)
    total_loss = mse + mnc_loss
    
    return total_loss

# Dictionary mapping loss function names to their implementations
LOSS_FUNCTIONS = {
    'mse': kernel_mse_loss,
    'l1': kernel_l1_loss,
    'huber': kernel_huber_loss,
    'kl': kernel_kl_divergence,
    'wasserstein': kernel_wasserstein_distance,
    'gradient': kernel_gradient_loss,
    'ms_ssim': kernel_ms_ssim_loss,
    'hybrid': kernel_2d_loss,
    'mse_mnc': kernel_mse_mnc_loss
}

# ------------------------------- Model Code --------------------------------- #

@tf.function(reduce_retracing=True)
def predict_step(model, x):
    return model(x, training=False)

@tf.function(reduce_retracing=True)
def distributed_predict_step(strategy, model, x):
    """Runs model inference on each replica and returns concatenated results."""
    def replica_predict_step(inputs):
        return model(inputs, training=False)
    per_replica_outputs = strategy.run(replica_predict_step, args=(x,))
    gathered_outputs = strategy.gather(per_replica_outputs, axis=0)
    return gathered_outputs

def create_lightweight_model(input_shape, kernel_size=32*32):
    """
    Create a lightweight model for kernel prediction.
    
    Args:
        input_shape: Shape of input images
        kernel_size: Size of the output kernel (default: 32*32)
    """
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=input_shape),
        tf.keras.layers.Rescaling(1./255),
        
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        
        # Ensure output matches the kernel size
        tf.keras.layers.Dense(kernel_size, activation=None)
    ])

def train_neural_network(X, kernels, strategy, model=None, epochs=5, learning_rate=0.001, batch_size=16, loss_function='mse'):
    """
    Train a neural network to predict kernels from convolved images.

    Args:
        X: List of convolved images
        kernels: List of kernels used for convolution (or list of tuples (kernel, kernel_type))
        strategy: TensorFlow distribution strategy
        model: Existing model to continue training, or None to create a new one
        epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        batch_size: Batch size for training
        loss_function: Name of the loss function to use (must be a key in LOSS_FUNCTIONS)
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Convert X to numpy array
    X = np.array(X).astype(np.float32)
    
    # If kernels is a list of tuples (kernel, kernel_type), extract the first element
    if isinstance(kernels, list) and isinstance(kernels[0], (list, tuple)):
        kernel_list = [normalize_kernel(k) for k, _ in kernels]
    else:
        kernel_list = [normalize_kernel(kernels)] * len(X)
    
    # Get the kernel size from the first kernel
    kernel_size = kernel_list[0].size
    
    # Build the target labels by flattening each normalized kernel
    y = tf.convert_to_tensor(
        np.array([k.flatten() for k in kernel_list]).astype(np.float32)
    )

    # Get the loss function
    if loss_function not in LOSS_FUNCTIONS:
        print(f"Warning: Loss function '{loss_function}' not found. Using MSE loss instead.")
        loss_function = 'mse'
    
    loss_fn = LOSS_FUNCTIONS[loss_function]

    # Build/compile model under distribution strategy
    if strategy is not None:
        with strategy.scope():
            if model is None:
                model = create_lightweight_model(X.shape[1:], kernel_size=kernel_size)
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate),
                    loss=loss_fn,
                    run_eagerly=False
                )
    else:
        if model is None:
            model = create_lightweight_model(X.shape[1:], kernel_size=kernel_size)
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss=loss_fn,
                run_eagerly=False
            )

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Train
    history = model.fit(
        dataset,
        epochs=epochs,
        verbose=2
    )

    return model, history

