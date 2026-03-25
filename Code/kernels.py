import numpy as np
import random
from astropy.modeling.functional_models import Gaussian2D, Moffat2D, AiryDisk2D

###############################################################################
# Available kernel types
###############################################################################
KERNEL_TYPES = [
    "gaussian", 
    "moffat", 
    "airy", 
    "double_gaussian", 
    "elliptical_gaussian",
    "ring", 
    "donut", 
    "cross",
    "top_hat",
    "shell"
]

###############################################################################
# Utility functions
###############################################################################
def compute_kernel_distribution(num_kernels, kernel_types):
    """
    Returns a list of counts indicating how many kernels of each type 
    should be generated to achieve as uniform a distribution as possible.
    """
    n_types = len(kernel_types)
    base = num_kernels // n_types
    remainder = num_kernels % n_types
    
    distribution = [base] * n_types
    for i in range(remainder):
        distribution[i] += 1
    
    return distribution

###############################################################################
# Individual kernel helpers
###############################################################################
def top_hat_2d(x, y, x_0, y_0, radius):
    """
    Returns a simple top-hat function (filled disk) with given radius.
    """
    r = np.sqrt((x - x_0)**2 + (y - y_0)**2)
    return (r <= radius).astype(float)

###############################################################################
# Main kernel generator
###############################################################################
def generate_kernel(kernel_type="gaussian", size=27):
    """
    Generates a 2D kernel of the specified type.
    
    Args:
        kernel_type: Type of kernel to generate
        size: Size of the kernel (both width and height)
    
    Returns:
        A normalized kernel of the specified size
    """
    
    # Use the provided kernel size
    x = np.arange(0, size)
    y = np.arange(0, size)
    x, y = np.meshgrid(x, y)
    
    # Allow the "center" to drift slightly within ~1 pixel
    x_0 = size // 2 + np.random.uniform(-1.0, 1.0)
    y_0 = size // 2 + np.random.uniform(-1.0, 1.0)

    # Initialize kernel
    kernel = None

    if kernel_type == "gaussian":
        # Scale standard deviations relative to kernel size
        x_stddev = np.random.uniform(size/32, size/4)
        y_stddev = np.random.uniform(size/32, size/4)
        theta = np.random.uniform(0, 2 * np.pi)

        model = Gaussian2D(
            amplitude=1.0,
            x_mean=x_0,
            y_mean=y_0,
            x_stddev=x_stddev,
            y_stddev=y_stddev,
            theta=theta
        )
        kernel = model(x, y)

    elif kernel_type == "moffat":
        # Scale gamma relative to kernel size
        gamma = np.random.uniform(size/32, size/5.3)
        alpha = np.random.uniform(1.0, 4.0)

        model = Moffat2D(
            amplitude=1.0,
            x_0=x_0,
            y_0=y_0,
            gamma=gamma,
            alpha=alpha
        )
        kernel = model(x, y)

    elif kernel_type == "airy":
        radius = np.random.uniform(size/32, size/3.2)
        model = AiryDisk2D(
            amplitude=1.0,
            x_0=x_0,
            y_0=y_0,
            radius=radius
        )
        kernel = model(x, y)
        
        # Enhance the rings slightly
        kernel = np.power(kernel, 0.5)
        kernel += np.max(kernel) * 0.01

    elif kernel_type == "double_gaussian":
        # Two Gaussians, different widths
        x_stddev1 = np.random.uniform(size/32, size/10.7)
        x_stddev2 = np.random.uniform(size/10.7, size/5.3)

        model1 = Gaussian2D(
            amplitude=0.6,
            x_mean=x_0,
            y_mean=y_0,
            x_stddev=x_stddev1,
            y_stddev=x_stddev1
        )
        model2 = Gaussian2D(
            amplitude=0.4,
            x_mean=x_0,
            y_mean=y_0,
            x_stddev=x_stddev2,
            y_stddev=x_stddev2
        )
        kernel = model1(x, y) + model2(x, y)

    elif kernel_type == "elliptical_gaussian":
        # Strongly elliptical Gaussian
        x_stddev = np.random.uniform(size/32, size/8)
        y_stddev = x_stddev * np.random.uniform(1.5, 3.0)
        theta = np.random.uniform(0, 2*np.pi)

        model = Gaussian2D(
            amplitude=1.0,
            x_mean=x_0,
            y_mean=y_0,
            x_stddev=x_stddev,
            y_stddev=y_stddev,
            theta=theta
        )
        kernel = model(x, y)

    elif kernel_type == "ring":
        # A ring shape
        radius = np.random.uniform(size/8, size/2.7)
        width = np.random.uniform(size/32, size/10.7)
        r = np.sqrt((x - x_0)**2 + (y - y_0)**2)
        kernel = np.exp(-((r - radius)**2) / (2 * width**2))

    elif kernel_type == "donut":
        # Donut shape: ring with a central depression
        radius = np.random.uniform(size/8, size/3.2)
        width = np.random.uniform(size/32, size/10.7)
        r = np.sqrt((x - x_0)**2 + (y - y_0)**2)
        ring = np.exp(-((r - radius)**2) / (2 * width**2))

        # Negative gaussian at center
        central_amp = np.random.uniform(-0.8, -0.3)
        central_size = np.random.uniform(size/33.75, size/16)
        central = Gaussian2D(
            amplitude=central_amp,
            x_mean=x_0,
            y_mean=y_0,
            x_stddev=central_size,
            y_stddev=central_size
        )(x, y)

        kernel = ring + central
        kernel = np.maximum(kernel, 0)  # clip negative

    elif kernel_type == "cross":
        # Cross pattern. We combine two perpendicular "bars" in a rotated frame
        width = np.random.uniform(size/64, size/16)
        angle = np.random.uniform(0, np.pi)
        
        # Rotate coords
        x_rot = (x - x_0) * np.cos(angle) + (y - y_0) * np.sin(angle)
        y_rot = -(x - x_0) * np.sin(angle) + (y - y_0) * np.cos(angle)

        bar_x = np.exp(-x_rot**2 / (2 * width**2))
        bar_y = np.exp(-y_rot**2 / (2 * width**2))
        kernel = (bar_x + bar_y) / 2.0

    elif kernel_type == "top_hat":
        # Simple disk
        radius = np.random.uniform(size/16, size/3.2)
        kernel = top_hat_2d(x, y, x_0, y_0, radius)

    elif kernel_type == "shell":
        # A shell or "thin ring" shape
        # This is like a radial Gaussian peak at some radius
        radius = np.random.uniform(size/8, size/2.7)
        shell_width = np.random.uniform(size/64, size/16)
        r = np.sqrt((x - x_0)**2 + (y - y_0)**2)
        kernel = np.exp(-((r - radius)**2) / (2 * shell_width**2))

    else:
        # Fallback: just do a small Gaussian if we ever get an unknown type
        model = Gaussian2D(
            amplitude=1.0,
            x_mean=x_0,
            y_mean=y_0,
            x_stddev=size/16,
            y_stddev=size/16
        )
        kernel = model(x, y)

    # Ensure non-negative, then normalize
    kernel = np.maximum(kernel, 0)
    sum_val = np.sum(kernel)
    if sum_val < 1e-12:
        # If it's basically zero, fill with a small uniform patch to avoid NaNs
        kernel = np.ones_like(kernel) / kernel.size
    else:
        kernel /= sum_val

    return kernel

###############################################################################
# Functions to generate multiple kernels
###############################################################################
def generate_distributed_kernels(num_kernels, kernel_size=27):
    """
    Generates kernels in (kernel, type) pairs, with a nearly uniform 
    distribution across KERNEL_TYPES.
    
    Args:
        num_kernels: Number of kernels to generate
        kernel_size: Size of each kernel (width and height)
    """
    distribution = compute_kernel_distribution(num_kernels, KERNEL_TYPES)
    kernels = []
    for count, ktype in zip(distribution, KERNEL_TYPES):
        for _ in range(count):
            kernel = generate_kernel(ktype, size=kernel_size)
            kernels.append((kernel, ktype))
    return kernels


def generate_cyclic_kernels(num_kernels, kernel_size=21):
    """
    Generates kernels in (kernel, type) pairs by cycling 
    through KERNEL_TYPES in order.
    
    Args:
        num_kernels: Number of kernels to generate
        kernel_size: Size of each kernel (width and height)
    """
    kernels = []
    for i in range(num_kernels):
        kernel_type = KERNEL_TYPES[i % len(KERNEL_TYPES)]
        kernel = generate_kernel(kernel_type, size=kernel_size)
        kernels.append((kernel, kernel_type))
    return kernels

def load_kernels(num_kernels=10, kernel_size=27):
    """
    Convenience function that calls generate_cyclic_kernels 
    by default. Switch to generate_distributed_kernels if you prefer.
    
    Args:
        num_kernels: Number of kernels to generate
        kernel_size: Size of each kernel (width and height)
    """
    return generate_cyclic_kernels(num_kernels, kernel_size=kernel_size)
