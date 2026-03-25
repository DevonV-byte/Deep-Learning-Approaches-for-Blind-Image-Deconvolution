import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
from kernels import generate_kernel, KERNEL_TYPES

# Import metrics from kernel_comparison.py
from kernel_comparison import (
    compute_mse, compute_l1, compute_huber, compute_kl_divergence,
    compute_wasserstein, compute_gradient_diff, compute_ssim, compute_ms_ssim
)

# Create output directory
os.makedirs("metric_verification_results", exist_ok=True)

def generate_kernel_pairs(n_pairs=50, same_type_ratio=0.5):
    """
    Generate pairs of kernels with a mix of same-type and different-type pairs.
    
    Args:
        n_pairs: Number of kernel pairs to generate
        same_type_ratio: Fraction of pairs that should be of the same kernel type
        
    Returns:
        List of tuples (kernel1, kernel2, type1, type2)
    """
    pairs = []
    n_same_type = int(n_pairs * same_type_ratio)
    n_diff_type = n_pairs - n_same_type
    
    print(f"Generating {n_pairs} kernel pairs ({n_same_type} same type, {n_diff_type} different type)...")
    
    # Generate same-type pairs
    for _ in tqdm(range(n_same_type)):
        kernel_type = random.choice(KERNEL_TYPES)
        kernel1 = generate_kernel(kernel_type)
        kernel2 = generate_kernel(kernel_type)
        pairs.append((kernel1, kernel2, kernel_type, kernel_type))
    
    # Generate different-type pairs
    for _ in tqdm(range(n_diff_type)):
        type1, type2 = random.sample(KERNEL_TYPES, 2)
        kernel1 = generate_kernel(type1)
        kernel2 = generate_kernel(type2)
        pairs.append((kernel1, kernel2, type1, type2))
    
    return pairs

def compute_all_metrics_for_pair(kernel1, kernel2):
    """Compute all metrics for a pair of kernels"""
    metrics = {
        'MSE': compute_mse(kernel1, kernel2),
        'L1': compute_l1(kernel1, kernel2),
        'Huber': compute_huber(kernel1, kernel2),
        'KL Divergence': compute_kl_divergence(kernel1, kernel2),
        'Wasserstein': compute_wasserstein(kernel1, kernel2),
        'Gradient Diff': compute_gradient_diff(kernel1, kernel2),
        'SSIM': compute_ssim(kernel1, kernel2),
        'MS-SSIM': compute_ms_ssim(kernel1, kernel2)
    }
    return metrics

def visualize_kernel_pairs_with_metrics(pairs, n_samples=10, sort_by_metric=None):
    """
    Visualize kernel pairs with their metric scores.
    
    Args:
        pairs: List of (kernel1, kernel2, type1, type2) tuples
        n_samples: Number of pairs to visualize
        sort_by_metric: If provided, sort pairs by this metric before visualization
    """
    # Compute metrics for all pairs
    pair_metrics = []
    for kernel1, kernel2, type1, type2 in tqdm(pairs, desc="Computing metrics"):
        metrics = compute_all_metrics_for_pair(kernel1, kernel2)
        pair_metrics.append((kernel1, kernel2, type1, type2, metrics))
    
    # Sort pairs if requested
    if sort_by_metric:
        # For similarity metrics like SSIM, higher is more similar
        if sort_by_metric in ['SSIM', 'MS-SSIM']:
            pair_metrics.sort(key=lambda x: x[4][sort_by_metric], reverse=True)
        # For distance metrics, lower is more similar
        else:
            pair_metrics.sort(key=lambda x: x[4][sort_by_metric])
    else:
        # Shuffle to get a random sample
        random.shuffle(pair_metrics)
    
    # Select sample pairs to visualize
    selected_pairs = pair_metrics[:n_samples]
    
    # Create a figure for each metric
    for metric_name in selected_pairs[0][4].keys():
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
        fig.suptitle(f'Kernel Pairs Sorted by {sort_by_metric if sort_by_metric else "Random"}', fontsize=16)
        
        for i, (kernel1, kernel2, type1, type2, metrics) in enumerate(selected_pairs):
            # Plot first kernel
            axes[i, 0].imshow(kernel1, cmap='viridis')
            axes[i, 0].set_title(f"Type: {type1}")
            axes[i, 0].axis('off')
            
            # Plot second kernel
            axes[i, 1].imshow(kernel2, cmap='viridis')
            axes[i, 1].set_title(f"Type: {type2}")
            axes[i, 1].axis('off')
            
            # Plot absolute difference
            diff = np.abs(kernel1 - kernel2)
            im = axes[i, 2].imshow(diff, cmap='hot')
            axes[i, 2].set_title(f"Difference | {metric_name}: {metrics[metric_name]:.4f}")
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f"metric_verification_results/{metric_name}_{'sorted' if sort_by_metric else 'random'}.png", dpi=300)
        plt.close()

def create_metric_comparison_grid(pairs, n_samples=5):
    """
    Create a grid showing how different metrics rank the same set of kernel pairs.
    This helps visualize which metrics agree/disagree on kernel similarity.
    
    Args:
        pairs: List of (kernel1, kernel2, type1, type2) tuples
        n_samples: Number of pairs to include in the grid
    """
    # Compute metrics for all pairs
    pair_metrics = []
    for kernel1, kernel2, type1, type2 in tqdm(pairs, desc="Computing metrics for grid"):
        metrics = compute_all_metrics_for_pair(kernel1, kernel2)
        pair_metrics.append((kernel1, kernel2, type1, type2, metrics))
    
    # Randomly select pairs to visualize
    selected_pairs = random.sample(pair_metrics, min(n_samples, len(pair_metrics)))
    
    # Get all metric names
    metric_names = list(selected_pairs[0][4].keys())
    
    # Create a large figure
    fig, axes = plt.subplots(len(metric_names) + 1, n_samples, 
                            figsize=(4*n_samples, 3*(len(metric_names) + 1)))
    
    # First row shows the kernel pairs
    for i, (kernel1, kernel2, type1, type2, _) in enumerate(selected_pairs):
        # Show the two kernels side by side in the same subplot
        ax = axes[0, i]
        # Create a combined image with a divider
        combined = np.zeros((kernel1.shape[0], kernel1.shape[1]*2 + 1))
        combined[:, :kernel1.shape[1]] = kernel1
        combined[:, kernel1.shape[1]+1:] = kernel2
        combined[:, kernel1.shape[1]] = 1  # white divider line
        
        ax.imshow(combined, cmap='viridis')
        ax.set_title(f"{type1} vs {type2}")
        ax.axis('off')
    
    # Each subsequent row shows how a different metric ranks these pairs
    for row, metric_name in enumerate(metric_names, 1):
        # Sort the pairs by this metric
        if metric_name in ['SSIM', 'MS-SSIM']:
            sorted_pairs = sorted(selected_pairs, key=lambda x: x[4][metric_name], reverse=True)
        else:
            sorted_pairs = sorted(selected_pairs, key=lambda x: x[4][metric_name])
        
        # Map original pairs to their rank according to this metric
        pair_to_rank = {id(pair): idx for idx, pair in enumerate(sorted_pairs)}
        
        # Display the pairs in their original order but with their rank
        for col, pair in enumerate(selected_pairs):
            rank = pair_to_rank[id(pair)]
            metric_value = pair[4][metric_name]
            
            # Show the difference image
            diff = np.abs(pair[0] - pair[1])
            im = axes[row, col].imshow(diff, cmap='hot')
            axes[row, col].set_title(f"Rank: {rank+1}, Value: {metric_value:.4f}")
            axes[row, col].axis('off')
    
    # Add row labels for metrics
    for row, metric_name in enumerate(metric_names, 1):
        axes[row, 0].set_ylabel(metric_name, fontsize=12, rotation=90, labelpad=10)
    
    # Add a title for the first row
    axes[0, 0].set_ylabel("Kernel Pairs", fontsize=12, rotation=90, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(f"metric_verification_results/metric_comparison_grid.png", dpi=300)
    plt.close()

def main():
    # Generate kernel pairs
    pairs = generate_kernel_pairs(n_pairs=100, same_type_ratio=0.5)
    
    # Visualize random pairs
    visualize_kernel_pairs_with_metrics(pairs, n_samples=8)
    
    # Visualize pairs sorted by each metric
    for metric in ['MSE', 'L1', 'Huber', 'KL Divergence', 'Wasserstein', 'Gradient Diff', 'SSIM', 'MS-SSIM']:
        visualize_kernel_pairs_with_metrics(pairs, n_samples=8, sort_by_metric=metric)
    
    # Create comparison grid
    create_metric_comparison_grid(pairs, n_samples=6)
    
    print("Visualization complete! Check the 'metric_verification_results' directory.")

if __name__ == "__main__":
    main() 