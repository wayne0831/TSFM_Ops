import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 6가지 기본 커널 함수 정의
# ---------------------------------------------------------
def kernel_constant(t, c=1.0):
    return np.full((len(t), len(t)), c)

def kernel_white_noise(t, sigma_n=0.5):
    return np.eye(len(t)) * (sigma_n ** 2)

def kernel_linear(t, sigma=1.0):
    # 정규화된 시간 축 사용 (0 ~ 1)
    t_norm = (t - np.mean(t)) / np.std(t)
    return (sigma ** 2) + np.outer(t_norm, t_norm)

def kernel_rbf(t, length_scale=20.0):
    dist_sq = (t[:, None] - t[None, :]) ** 2
    return np.exp(-dist_sq / (2.0 * (length_scale ** 2)))

def kernel_rational_quadratic(t, alpha=0.5, length_scale=20.0):
    dist_sq = (t[:, None] - t[None, :]) ** 2
    return (1.0 + dist_sq / (2.0 * alpha * (length_scale ** 2))) ** (-alpha)

def kernel_periodic(t, period=40.0, length_scale=1.0):
    diff = np.abs(t[:, None] - t[None, :])
    sin_term = np.sin(np.pi * diff / period)
    return np.exp(-2.0 * (sin_term ** 2) / (length_scale ** 2))

# ---------------------------------------------------------
# 2. GP Prior 샘플링 함수
# ---------------------------------------------------------
def sample_from_gp(cov_matrix, num_samples=3, jitter=1e-5):
    length = cov_matrix.shape[0]
    # 수치적 안정성을 위해 대각선에 작은 jitter 추가
    K = cov_matrix + np.eye(length) * jitter
    mean = np.zeros(length)
    return np.random.multivariate_normal(mean, K, size=num_samples)

# ---------------------------------------------------------
# 3. 6개 커널별 시계열 시각화
# ---------------------------------------------------------
np.random.seed(42)
T = 200
t = np.linspace(0, T - 1, T)

kernels = [
    ("1. Constant Kernel", kernel_constant(t)),
    ("2. White Noise Kernel", kernel_white_noise(t, sigma_n=0.3)),
    ("3. Linear Kernel", kernel_linear(t, sigma=1.0)),
    ("4. RBF Kernel", kernel_rbf(t, length_scale=15.0)),
    ("5. Rational Quadratic Kernel", kernel_rational_quadratic(t, alpha=0.2, length_scale=15.0)),
    ("6. Periodic Kernel", kernel_periodic(t, period=40.0))
]

fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
axes = axes.flatten()

for idx, (title, cov) in enumerate(kernels):
    ax = axes[idx]
    samples = sample_from_gp(cov, num_samples=3)
    for s_idx, sample in enumerate(samples):
        ax.plot(t, sample, label=f"Sample {s_idx + 1}" if idx == 0 else "")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

axes[0].legend(loc="upper right")
plt.tight_layout()
plt.show()