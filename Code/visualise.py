import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy import stats

def format_time(seconds):
    """Helper function to format time in a human-readable way."""
    if seconds >= 3600:  # More than an hour
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"
    elif seconds >= 60:  # More than a minute
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        return f"{seconds:.2f}s"

def visualise_batch(batch_dir):
    # Load batch info
    with open(os.path.join(batch_dir, "batch_info.json"), 'r') as f:
        batch_info = json.load(f)
    
    # Determine if images were processed in grayscale
    is_grayscale = len(batch_info['kernel_stats']['shape']) <= 2
    
    # Create figure with adjusted size for title
    plt.figure(figsize=(15, 16))
    
    # Add batch information as title with more space at top
    plt.suptitle(f"Kernel Type: {batch_info['kernel_type']}\n"
                f"Images: {batch_info['original_images']} original, {batch_info['augmented_images']} augmented\n"
                f"Processed: {batch_info['timing']['start_time']}", 
                y=0.99)  # Moved title up slightly
    
    # First Row: Original Process
    # 1. Load and plot augmented validation image
    val_aug_image = Image.open(os.path.join(batch_dir, "augmented_1.png"))
    plt.subplot(3, 3, 1)
    plt.imshow(val_aug_image, cmap='gray' if is_grayscale else None)
    plt.title("Validation Image")
    plt.axis('off')
    
    # 2. Load and plot original kernel
    kernel = np.load(os.path.join(batch_dir, "kernel.npy"))
    if len(kernel.shape) == 1:
        kernel = kernel.reshape(32, 32)
    elif len(kernel.shape) > 2:
        kernel = kernel.squeeze()
    
    plt.subplot(3, 3, 2)
    plt.imshow(kernel, cmap='gray')
    plt.colorbar()
    plt.title(f"Original Kernel\n({batch_info['kernel_type']})")
    plt.axis('off')
    
    # 3. Load and plot convolved result
    convolved = np.load(os.path.join(batch_dir, "convolved.npy"))
    plt.subplot(3, 3, 3)
    plt.imshow(convolved, cmap='gray' if is_grayscale else None)
    plt.colorbar()
    plt.title("Original Convolution")
    plt.axis('off')
    
    # Second Row: Predicted Process
    # 4. Plot same validation image again
    plt.subplot(3, 3, 4)
    plt.imshow(val_aug_image, cmap='gray' if is_grayscale else None)
    plt.title("Validation Image")
    plt.axis('off')
    
    # 5. Load and plot predicted kernel
    model_output = np.load(os.path.join(batch_dir, "model_output.npy"))
    if len(model_output.shape) == 1:
        model_output = model_output.reshape(32, 32)
    elif len(model_output.shape) > 2:
        model_output = model_output.squeeze()
    
    plt.subplot(3, 3, 5)
    plt.imshow(model_output, cmap='gray')
    plt.colorbar()
    plt.title("Predicted Kernel: best model output on Validation Set")
    plt.axis('off')
    
    # 6. Load and plot predicted convolution
    predicted_conv = np.load(os.path.join(batch_dir, "predicted_convolution.npy"))
    plt.subplot(3, 3, 6)
    plt.imshow(predicted_conv, cmap='gray' if is_grayscale else None)
    plt.colorbar()
    plt.title("Predicted Convolution")
    plt.axis('off')
    
    # Third Row: Statistics
    plt.subplot(3, 3, 7)  # Span all three columns
    plt.axis('off')
    
    # Calculate various similarity metrics
    convolved = np.array(convolved)
    predicted_conv = np.array(predicted_conv)

    # Squeeze any singleton dimensions
    if convolved.ndim > 2:
        convolved = convolved.squeeze()
    if predicted_conv.ndim > 2:
        predicted_conv = predicted_conv.squeeze()

    # Calculate metrics
    ssim_score = ssim(convolved, predicted_conv, 
                     data_range=predicted_conv.max() - predicted_conv.min(),
                     full=False)
    ssim_score = (ssim_score + 1) / 2
    
    pearson_corr = stats.pearsonr(convolved.flatten(), predicted_conv.flatten())[0]
    spearman_corr = stats.spearmanr(convolved.flatten(), predicted_conv.flatten())[0]
    mse = np.mean((convolved - predicted_conv) ** 2)

    # Bottom row - split into three columns
    
    # 7. Processing Times (bottom left)
    plt.subplot(3, 3, 7)
    plt.axis('off')
    timing_text = (
        "\n"  # New line before first row
        f"Processing Times:\n"
        f"Total: {batch_info['timing']['processing_time']:.2f}s\n"
        f"Augmentation: {batch_info['timing']['augmentation_time']}\n"
        f"Convolution: {batch_info['timing']['convolution_time']}\n"
        f"Training: {batch_info['timing']['training_time']}\n"  # New line after first row
        f"MSE: {batch_info['kernel_comparison']['mse']:.4e}\n"
        f"MNC: {batch_info['kernel_comparison']['mnc']:.4f}"
    )
    plt.text(0.5, 0.5, timing_text, ha='center', va='center')
    plt.title("Timing Information")

    # 8. Kernel Statistics (bottom middle)
    plt.subplot(3, 3, 8)
    plt.axis('off')
    kernel_stats_text = (
        "\n"  # New line before first row
        f"Kernel Statistics:\n"
        f"Min: {batch_info['kernel_stats']['min']:.4f}\n"
        f"Max: {batch_info['kernel_stats']['max']:.4f}\n"
        f"Mean: {batch_info['kernel_stats']['mean']:.4f}\n"
        f"Std: {batch_info['kernel_stats']['std']:.4f}\n"  # New line after first row
        f"Sum: {batch_info['kernel_stats']['sum']:.4f}"
    )
    plt.text(0.5, 0.5, kernel_stats_text, ha='center', va='center')
    plt.title("Kernel Statistics")

    # 9. Convolution Comparison Metrics (bottom right)
    plt.subplot(3, 3, 9)
    plt.axis('off')
    comparison_text = (
        "\n"  # New line before first row
        f"Convolution Comparison:\n"
        f"MSE: {mse:.3e}\n"
        f"MNC: {pearson_corr:.3f}\n"
        f"Pearson: {pearson_corr:.3f}\n"
        f"Spearman: {spearman_corr:.3f}"
    )
    plt.text(0.5, 0.5, comparison_text, ha='center', va='center')
    plt.title("Comparison Metrics")

    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

def visualise_global_stats(visualization_dir):
    # Load global statistics
    stats_file = os.path.join(visualization_dir, "global_statistics.json")
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Global statistics file not found: {stats_file}")
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Create figure with a larger size and better spacing
    plt.figure(figsize=(20, 16))
    
    # Create a GridSpec to control subplot layout
    gs = plt.GridSpec(3, 3, height_ratios=[1, 1.2, 1.2])
    
    # Get validation metrics
    if "validation_metrics" not in stats or not stats["validation_metrics"]:
        raise ValueError("No validation metrics found in statistics file")
    
    val_metrics = stats["validation_metrics"]
    batches = [x["batch"] for x in val_metrics]
    mse_scores = [x["mse_loss"] for x in val_metrics]
    mnc_scores = [x["mnc"] for x in val_metrics]
    kernel_types = [x["kernel_type"] for x in val_metrics]
    
    # Get the loss function name
    loss_function_name = val_metrics[0].get("loss_function", "mse").upper()
    
    # Calculate kernel stats for metrics values
    kernel_stats = {}
    for k_type in set(kernel_types):
        type_mse = [x["mse_loss"] for x in val_metrics if x["kernel_type"] == k_type]
        type_mnc = [x["mnc"] for x in val_metrics if x["kernel_type"] == k_type]
        if type_mse and type_mnc:
            kernel_stats[k_type] = {
                'mse_min': min(type_mse),
                'mse_max': max(type_mse),
                'mse_mean': sum(type_mse) / len(type_mse),
                'mnc_min': min(type_mnc),
                'mnc_max': max(type_mnc),
                'mnc_mean': sum(type_mnc) / len(type_mnc),
                'count': len(type_mse)
            }
    
    # 1. Distribution of Kernel Types (top left)
    plt.subplot(gs[0, 0])
    kernel_types_dist = stats["kernel_types"]
    plt.bar(kernel_types_dist.keys(), kernel_types_dist.values())
    plt.title("Distribution of Kernel Types")
    plt.ylabel("Number of Batches")
    plt.xticks(rotation=45)
    
    # 2. Text information (top middle)
    plt.subplot(gs[0, 1])
    plt.axis('off')
    info_text = (
        f"Pipeline Timeline:\n"
        f"Start: {stats['pipeline_info']['start_time']}\n"
        f"Last Update: {stats['pipeline_info']['last_update']}\n\n"
        f"Processing Times:\n"
        f"Total: {format_time(stats['pipeline_info']['total_processing_time'])}\n"
        f"Average Batch: {format_time(stats['pipeline_info']['average_batch_time'])}\n"
        f"Augmentation: {format_time(stats['pipeline_info']['augmentation_time'])}\n"
        f"Convolution: {format_time(stats['pipeline_info']['convolution_time'])}\n"
        f"Training: {format_time(stats['pipeline_info']['training_time'])}"
    )
    plt.text(0.05, 0.95, info_text, va='top', fontsize=10)
    
    # 3. Metrics Statistics (top right)
    plt.subplot(gs[0, 2])
    plt.axis('off')
    stats_text = (
        f"Images Processed:\n"
        f"Total Batches: {stats['total_batches']}\n"
        f"Original Images: {stats['total_original_images']}\n"
        f"Augmented Images: {stats['total_augmented_images']}\n\n"
    )
    
    # Add metrics statistics
    if kernel_stats:
        stats_text += f"Metrics Statistics per Kernel Type:\n"
        sorted_kernels = sorted(kernel_stats.items(), key=lambda x: x[1]['mse_mean'])
        for k_type, k_stats in sorted_kernels:
            stats_text += f"{k_type}:\n"
            stats_text += f"  MSE: {k_stats['mse_mean']:.4e} (min: {k_stats['mse_min']:.4e}, max: {k_stats['mse_max']:.4e})\n"
            stats_text += f"  MNC: {k_stats['mnc_mean']:.4f} (min: {k_stats['mnc_min']:.4f}, max: {k_stats['mnc_max']:.4f})\n"
    
    plt.text(0.05, 0.95, stats_text, va='top', fontsize=10)
    
    # 4. MSE Evolution plot (middle row)
    plt.subplot(gs[1, :])
    
    # Plot MSE scores per kernel type
    for k_type in set(kernel_types):
        mask = [kt == k_type for kt in kernel_types]
        plt.scatter(
            [b for b, m in zip(batches, mask) if m],
            [s for s, m in zip(mse_scores, mask) if m],
            label=f"{k_type}",
            alpha=0.6,
            marker='o'
        )
    
    # Add trend line
    z = np.polyfit(batches, mse_scores, 1)
    p = np.poly1d(z)
    plt.plot(batches, p(batches), 'k--', alpha=0.3, label='Trend')
    
    plt.xlabel('Batch Number')
    plt.ylabel('MSE Score')
    plt.title('MSE Score Evolution per Kernel Type')
    plt.legend(loc='upper right')
    plt.yscale('log')  # Use log scale for better visualization
    
    # 5. MNC Evolution plot (bottom row)
    plt.subplot(gs[2, :])
    
    # Plot MNC scores per kernel type
    for k_type in set(kernel_types):
        mask = [kt == k_type for kt in kernel_types]
        plt.scatter(
            [b for b, m in zip(batches, mask) if m],
            [s for s, m in zip(mnc_scores, mask) if m],
            label=f"{k_type}",
            alpha=0.6,
            marker='x'
        )
    
    # Add trend line
    z = np.polyfit(batches, mnc_scores, 1)
    p = np.poly1d(z)
    plt.plot(batches, p(batches), 'r--', alpha=0.3, label='Trend')
    
    plt.xlabel('Batch Number')
    plt.ylabel('MNC Score')
    plt.title('MNC Score Evolution per Kernel Type')
    plt.legend(loc='lower right')
    
    # Add main title
    plt.suptitle(
        f"Global Training Statistics\n"
        f"Using {len(stats['kernel_types'])} Unique Kernel{'s' if len(stats['kernel_types']) > 1 else ''}\n"
        f"Loss Function: {loss_function_name}", 
        y=0.95, fontsize=16
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    
    return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description="Visualise training results")
    parser.add_argument("--visualization-dir", type=str, default="visualization",
                       help="Directory containing visualization data")
    parser.add_argument("--batch", type=int, help="Specific batch to visualise")
    parser.add_argument("--global-stats", action="store_true",
                       help="Show global statistics visualization")
    parser.add_argument("--save", action="store_true",
                       help="Save the visualization instead of displaying it")
    args = parser.parse_args()
    
    if args.global_stats:
        fig = visualise_global_stats(args.visualization_dir)
        if args.save:
            output_path = os.path.join(args.visualization_dir, "global_statistics.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Global statistics visualization saved to {output_path}")
        else:
            plt.show()
        return

    if args.batch is not None:
        batch_dir = os.path.join(args.visualization_dir, f"batch_{args.batch}")
        if not os.path.exists(batch_dir):
            print(f"Batch directory not found: {batch_dir}")
            return
        visualise_batch(batch_dir)
        if args.save:
            output_path = os.path.join(batch_dir, "visualization.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Batch visualization saved to {output_path}")
        else:
            plt.show()
    else:
        print("Please specify a batch number to visualise or use --global-stats")

if __name__ == "__main__":
    main() 