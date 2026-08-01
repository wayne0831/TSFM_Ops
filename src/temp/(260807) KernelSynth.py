import functools
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel, DotProduct, ExpSineSquared, RationalQuadratic, WhiteKernel
)

# 1. Define global parameters
LENGTH = 512  # Number of time steps in the generated series

# 2. Build the Kernel Bank (representing various frequencies, trends, and noise)
KERNEL_BANK = [
    ExpSineSquared(periodicity=24 / LENGTH),       # Hourly pattern
    ExpSineSquared(periodicity=24 * 7 / LENGTH),   # Weekly pattern
    ExpSineSquared(periodicity=365 / LENGTH),      # Yearly pattern
    RBF(length_scale=0.1),                         # Short-term smooth changes
    RBF(length_scale=1.0),                         # Medium-term smooth changes
    RBF(length_scale=10.0),                        # Long-term smooth trends
    RationalQuadratic(alpha=1.0),                  # Multi-scale variations
    DotProduct(sigma_0=1.0),                       # Linear trends
    WhiteKernel(noise_level=0.05),                 # Small random noise
    ConstantKernel(constant_value=1.0)             # Scale adjustments
]

def random_binary_map(a, b):
    """Randomly combines two kernels using either addition or multiplication."""
    binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
    return np.random.choice(binary_maps)(a, b)

def sample_from_gp_prior(kernel, X):
    """Fits a GP regressor with the composite kernel to sample a random function."""
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    # Draw one random sample from the unconditioned GP prior distribution
    y_samples = gp.sample_y(X[:, np.newaxis], n_samples=1, random_state=None)
    return y_samples.flatten()

def generate_kernel_synth_series(max_kernels=5):
    """Generates a single synthetic time series using random kernel composition."""
    while True:
        X = np.linspace(0, 1, LENGTH)
        
        # Randomly choose how many kernels to combine (1 to max_kernels)
        num_kernels = np.random.randint(1, max_kernels + 1)
        selected_kernels = np.random.choice(KERNEL_BANK, size=num_kernels, replace=True)
        
        # Sequentially combine the sampled kernels using the random math operators
        composite_kernel = functools.reduce(random_binary_map, selected_kernels)
        
        try:
            # Generate the time series sample
            ts = sample_from_gp_prior(kernel=composite_kernel, X=X)
            return ts, composite_kernel
        except np.linalg.LinAlgError:
            # Skip invalid matrices (e.g., non-positive definite results) and retry
            continue

# --- Example Usage ---
if __name__ == "__main__":
    synthetic_ts, selected_structure = generate_kernel_synth_series(max_kernels=4)
    print(f"Generated a series of length: {len(synthetic_ts)}")
    print(f"Composite Kernel Structure used:\n{selected_structure}")