import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import cv2

# Import from your existing kernels module
from kernels import generate_kernel, KERNEL_TYPES

# Replace the torch imports with a try-except block
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. MS-SSIM metric will use a fallback implementation.")

# Create output directory for plots
os.makedirs("kernel_comparison_results", exist_ok=True)

def generate_kernel_dataset(n_samples=10000):
    """Generate a dataset of kernels with equal distribution across types"""
    kernels = []
    labels = []
    samples_per_type = n_samples // len(KERNEL_TYPES)
    
    print(f"Generating {n_samples} kernels ({samples_per_type} per type)...")
    
    for kernel_type in tqdm(KERNEL_TYPES):
        for _ in range(samples_per_type):
            kernel = generate_kernel(kernel_type)
            kernels.append(kernel)
            labels.append(kernel_type)
    
    return np.array(kernels), np.array(labels)

def compute_mse(k1, k2):
    """Mean squared error"""
    return mean_squared_error(k1, k2)

def compute_l1(k1, k2):
    """L1 distance (Mean absolute error)"""
    return np.mean(np.abs(k1 - k2))

def compute_huber(k1, k2, delta=1.0):
    """Huber loss"""
    diff = np.abs(k1 - k2)
    mask = diff <= delta
    return np.mean(
        mask * 0.5 * diff**2 + 
        ~mask * delta * (diff - 0.5 * delta)
    )

def compute_kl_divergence(k1, k2):
    """KL divergence (with small epsilon to avoid division by zero)"""
    eps = 1e-10
    k1_safe = k1 + eps
    k2_safe = k2 + eps
    return np.sum(k1_safe * np.log(k1_safe / k2_safe))

def compute_wasserstein(k1, k2):
    """1D Wasserstein distance (Earth Mover's Distance)"""
    # Flatten 2D kernels to 1D for wasserstein calculation
    return wasserstein_distance(k1.flatten(), k2.flatten())

def compute_gradient_diff(k1, k2):
    """Difference in gradients"""
    grad_x1 = cv2.Sobel(k1, cv2.CV_64F, 1, 0, ksize=3)
    grad_y1 = cv2.Sobel(k1, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag1 = np.sqrt(grad_x1**2 + grad_y1**2)
    
    grad_x2 = cv2.Sobel(k2, cv2.CV_64F, 1, 0, ksize=3)
    grad_y2 = cv2.Sobel(k2, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag2 = np.sqrt(grad_x2**2 + grad_y2**2)
    
    return np.mean(np.abs(grad_mag1 - grad_mag2))

def compute_ssim(k1, k2):
    """Structural Similarity Index"""
    return ssim(k1, k2, data_range=1.0)

def compute_ms_ssim(k1, k2):
    """Multi-Scale Structural Similarity Index"""
    if TORCH_AVAILABLE:
        # Convert to torch tensors
        k1_t = torch.from_numpy(k1).unsqueeze(0).unsqueeze(0).float()
        k2_t = torch.from_numpy(k2).unsqueeze(0).unsqueeze(0).float()
        
        # Check if kernel size is large enough for MS-SSIM
        kernel_size = k1.shape[0]
        
        # Use proper MS-SSIM implementation from PyTorch if kernel is large enough
        if kernel_size > 160:  # MS-SSIM requires this minimum size
            try:
                from pytorch_msssim import ms_ssim
                return ms_ssim(k1_t, k2_t, data_range=1.0).item()
            except (ImportError, AssertionError):
                # Fallback if pytorch_msssim is not available or other error
                return ssim(k1, k2, data_range=1.0)
        else:
            # For small kernels, use regular SSIM
            return ssim(k1, k2, data_range=1.0)
    else:
        # Fallback to regular SSIM if torch is not available
        return ssim(k1, k2, data_range=1.0)

def compute_all_metrics(kernels):
    """Compute all pairwise metrics for a sample of kernels"""
    n_samples = len(kernels)
    n_metrics = 8  # Number of metrics we're using
    
    # Initialize metric names and storage
    metric_names = ['mse', 'l1', 'huber', 'kl', 'wasserstein', 'gradient', 'ssim', 'ms_ssim']
    metrics = {name: np.zeros((n_samples, n_samples)) for name in metric_names}
    
    print("Computing pairwise metrics...")
    for i in tqdm(range(n_samples)):
        for j in range(i, n_samples):  # Only compute upper triangle
            # Compute all metrics
            metrics['mse'][i, j] = compute_mse(kernels[i], kernels[j])
            metrics['l1'][i, j] = compute_l1(kernels[i], kernels[j])
            metrics['huber'][i, j] = compute_huber(kernels[i], kernels[j])
            metrics['kl'][i, j] = compute_kl_divergence(kernels[i], kernels[j])
            metrics['wasserstein'][i, j] = compute_wasserstein(kernels[i], kernels[j])
            metrics['gradient'][i, j] = compute_gradient_diff(kernels[i], kernels[j])
            metrics['ssim'][i, j] = compute_ssim(kernels[i], kernels[j])
            metrics['ms_ssim'][i, j] = compute_ms_ssim(kernels[i], kernels[j])
            
            # Mirror values to lower triangle (for symmetric metrics)
            if i != j:
                metrics['mse'][j, i] = metrics['mse'][i, j]
                metrics['l1'][j, i] = metrics['l1'][i, j]
                metrics['huber'][j, i] = metrics['huber'][i, j]
                metrics['wasserstein'][j, i] = metrics['wasserstein'][i, j]
                metrics['gradient'][j, i] = metrics['gradient'][i, j]
                metrics['ssim'][j, i] = metrics['ssim'][i, j]
                metrics['ms_ssim'][j, i] = metrics['ms_ssim'][i, j]
                
                # KL divergence is not symmetric
                metrics['kl'][j, i] = compute_kl_divergence(kernels[j], kernels[i])
    
    return metrics, metric_names

def visualize_kernel_samples(kernels, labels, n_per_type=3):
    """Visualize sample kernels of each type"""
    fig, axes = plt.subplots(len(KERNEL_TYPES), n_per_type, figsize=(n_per_type*3, len(KERNEL_TYPES)*3))
    
    for i, kernel_type in enumerate(KERNEL_TYPES):
        # Get indices of this kernel type
        indices = np.where(labels == kernel_type)[0]
        # Select n_per_type random samples
        selected = np.random.choice(indices, size=min(n_per_type, len(indices)), replace=False)
        
        for j, idx in enumerate(selected):
            ax = axes[i, j]
            im = ax.imshow(kernels[idx], cmap='viridis')
            ax.set_title(f"{kernel_type}")
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("kernel_comparison_results/kernel_samples.png", dpi=300)
    plt.close()

def visualize_metric_distributions(metrics, metric_names, labels):
    """Visualize the distribution of each metric for same-type vs different-type kernels"""
    n_samples = len(labels)
    
    # Create a figure for each metric
    for metric_name in metric_names:
        plt.figure(figsize=(10, 6))
        
        # Collect same-type and different-type distances
        same_type_dists = []
        diff_type_dists = []
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):  # Only upper triangle
                if labels[i] == labels[j]:
                    same_type_dists.append(metrics[metric_name][i, j])
                else:
                    diff_type_dists.append(metrics[metric_name][i, j])
        
        # Plot distributions
        sns.histplot(same_type_dists, kde=True, label="Same Type", alpha=0.6)
        sns.histplot(diff_type_dists, kde=True, label="Different Types", alpha=0.6)
        
        plt.title(f"Distribution of {metric_name} metric")
        plt.xlabel(f"{metric_name} value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"kernel_comparison_results/{metric_name}_distribution.png", dpi=300)
        plt.close()

def visualize_metric_heatmaps(metrics, metric_names, labels):
    """Visualize heatmaps of each metric"""
    # Sort indices by kernel type for better visualization
    sorted_indices = np.argsort(labels)
    sorted_labels = labels[sorted_indices]
    
    # Create a figure for each metric
    for metric_name in metric_names:
        plt.figure(figsize=(12, 10))
        
        # Get the sorted metric matrix
        sorted_metric = metrics[metric_name][sorted_indices][:, sorted_indices]
        
        # Plot heatmap
        sns.heatmap(sorted_metric, cmap="viridis")
        
        # Add type boundaries
        type_boundaries = []
        current_type = sorted_labels[0]
        for i, label in enumerate(sorted_labels):
            if label != current_type:
                type_boundaries.append(i)
                current_type = label
        
        # Add lines at type boundaries
        for boundary in type_boundaries:
            plt.axhline(y=boundary, color='r', linestyle='-', linewidth=0.5)
            plt.axvline(x=boundary, color='r', linestyle='-', linewidth=0.5)
        
        plt.title(f"Heatmap of {metric_name} metric")
        plt.tight_layout()
        plt.savefig(f"kernel_comparison_results/{metric_name}_heatmap.png", dpi=300)
        plt.close()

def visualize_dimensionality_reduction(metrics, metric_names, labels):
    """Visualize 2D projections of kernels based on each metric"""
    for metric_name in metric_names:
        # Get the distance matrix for this metric
        distance_matrix = metrics[metric_name]
        
        # Apply t-SNE for visualization
        tsne = TSNE(n_components=2, metric='precomputed', random_state=42, init='random')
        tsne_result = tsne.fit_transform(distance_matrix)
        
        # Create a DataFrame for easy plotting
        df = pd.DataFrame({
            'x': tsne_result[:, 0],
            'y': tsne_result[:, 1],
            'kernel_type': labels
        })
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=df, x='x', y='y', hue='kernel_type', palette='tab10', s=100, alpha=0.7)
        plt.title(f"t-SNE visualization using {metric_name} metric")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"kernel_comparison_results/{metric_name}_tsne.png", dpi=300)
        plt.close()

def evaluate_metric_separability(metrics, metric_names, labels):
    """Evaluate how well each metric separates different kernel types"""
    results = {}
    
    for metric_name in metric_names:
        # Calculate within-class and between-class distances
        within_class_distances = []
        between_class_distances = []
        
        unique_labels = np.unique(labels)
        n_samples = len(labels)
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if labels[i] == labels[j]:
                    within_class_distances.append(metrics[metric_name][i, j])
                else:
                    between_class_distances.append(metrics[metric_name][i, j])
        
        # Calculate statistics
        within_mean = np.mean(within_class_distances)
        between_mean = np.mean(between_class_distances)
        
        # Separability score: ratio of between-class to within-class distances
        # Higher is better - means the metric distinguishes different types well
        separability = between_mean / within_mean if within_mean > 0 else float('inf')
        
        results[metric_name] = {
            'within_class_mean': within_mean,
            'between_class_mean': between_mean,
            'separability_score': separability
        }
    
    # Create a DataFrame for easy comparison
    df = pd.DataFrame({
        'Metric': list(results.keys()),
        'Within-Class Mean': [results[m]['within_class_mean'] for m in results],
        'Between-Class Mean': [results[m]['between_class_mean'] for m in results],
        'Separability Score': [results[m]['separability_score'] for m in results]
    })
    
    # Sort by separability score (higher is better)
    df = df.sort_values('Separability Score', ascending=False)
    
    # Plot the separability scores
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Metric', y='Separability Score')
    plt.title('Metric Separability Scores (higher is better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("kernel_comparison_results/metric_separability_scores.png", dpi=300)
    plt.close()
    
    # Save the results to CSV
    df.to_csv("kernel_comparison_results/metric_evaluation.csv", index=False)
    
    return df

def main():
    # Number of kernels to generate (adjust based on your computational resources)
    n_samples = 1000  # Using 1000 for demonstration, increase for better results
    
    # Generate kernels
    kernels, labels = generate_kernel_dataset(n_samples)
    
    # Visualize sample kernels
    visualize_kernel_samples(kernels, labels)
    
    # Compute metrics on a subset for efficiency
    subset_size = min(500, n_samples)  # Adjust based on your computational resources
    subset_indices = np.random.choice(n_samples, subset_size, replace=False)
    subset_kernels = kernels[subset_indices]
    subset_labels = labels[subset_indices]
    
    # Compute all metrics
    metrics, metric_names = compute_all_metrics(subset_kernels)
    
    # Visualize metric distributions
    visualize_metric_distributions(metrics, metric_names, subset_labels)
    
    # Visualize metric heatmaps
    visualize_metric_heatmaps(metrics, metric_names, subset_labels)
    
    # Visualize dimensionality reduction
    visualize_dimensionality_reduction(metrics, metric_names, subset_labels)
    
    # Evaluate metric separability
    separability_results = evaluate_metric_separability(metrics, metric_names, subset_labels)
    
    print("\nMetric Evaluation Results (sorted by separability):")
    print(separability_results)
    print("\nBest metric for kernel comparison:", separability_results.iloc[0]['Metric'])
    
    print("\nAll results saved to the 'kernel_comparison_results' directory.")

if __name__ == "__main__":
    main() 