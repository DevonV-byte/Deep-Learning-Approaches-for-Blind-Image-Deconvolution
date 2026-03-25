import numpy as np

def normalize_kernel(kernel):
    """
    Standardized kernel normalization function to be used across the pipeline.
    Ensures kernel sums to 1 and maintains relative relationships between values.
    """
    
    # Convert to numpy array if not already
    kernel = np.array(kernel)
    
    # Ensure positive values and maintain relative relationships
    kernel = kernel - np.min(kernel)
    
    # Normalize to sum to 1
    kernel = kernel / (np.sum(kernel) + 1e-8)
    
    return kernel

def split_data(data, validation_split=0.1):
    if not 0 <= validation_split <= 1:
        raise ValueError("validation_split must be between 0 and 1")
        
    split_idx = int(len(data) * (1 - validation_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data