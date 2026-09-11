import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. 기저 커널 함수 정의
# -------------------------------------------------------------
def rbf_kernel(t1, t2, l=15.0, sigma=1.0):
    diff = t1[:, None] - t2[None, :]
    return (sigma**2) * np.exp(-(diff**2) / (2.0 * l**2))

def periodic_kernel(t1, t2, p=30.0, l=1.0, sigma=1.0):
    diff = np.abs(t1[:, None] - t2[None, :])
    return (sigma**2) * np.exp(-2.0 * (np.sin(np.pi * diff / p)**2) / (l**2))

def linear_kernel(t1, t2, c=0.0, sigma_b=0.0, sigma_v=1.0):
    return (sigma_b**2) + (sigma_v**2) * ((t1[:, None] - c) * (t2[None, :] - c))

# -------------------------------------------------------------
# 2. GP 샘플링 함수 (K -> L -> y = L*u)
# -------------------------------------------------------------
def sample_gp(cov_matrix, num_samples=3, jitter=1e-6):
    N = cov_matrix.shape[0]
    K_stable = cov_matrix + np.eye(N) * jitter
    L = np.linalg.cholesky(K_stable)  # 숄레스키 분해
    u = np.random.normal(0, 1, size=(N, num_samples))  # 표준 정규 잡음
    return L @ u  # 상관관계 주입

# -------------------------------------------------------------
# 3. 데이터 및 커널 생성
# -------------------------------------------------------------
np.random.seed(42)
N = 150
t = np.linspace(0, 150, N)

# 4대 대표 커널 구성
K_rbf = rbf_kernel(t, t, l=15.0)
K_per = periodic_kernel(t, t, p=25.0, l=1.2)
K_lin = linear_kernel(t, t, c=0.0, sigma_v=0.02)
K_mult = K_lin * K_per  # LIN * PER (진폭 변조 복합 커널)

kernels = [
    ("RBF (Smooth Local Variation)", K_rbf),
    ("Periodic (Repeating Structure)", K_per),
    ("Linear (Global Trend)", K_lin),
    ("Linear x Periodic (Growing Amplitude)", K_mult)
]

# -------------------------------------------------------------
# 4. 시각화 (좌측: 공분산 행렬, 우측: 샘플링된 시계열)
# -------------------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(12, 14), gridspec_kw={'width_ratios': [1, 1.8]})

for i, (name, K) in enumerate(kernels):
    # (좌) 공분산 행렬 히트맵
    im = axes[i, 0].imshow(K, cmap='viridis', origin='upper')
    axes[i, 0].set_title(f"Covariance Matrix: {name.split('(')[0]}", fontsize=10, fontweight='bold')
    axes[i, 0].set_xlabel("Time Index $t'$")
    axes[i, 0].set_ylabel("Time Index $t$")
    fig.colorbar(im, ax=axes[i, 0], fraction=0.046, pad=0.04)

    # (우) GP 사전분포로부터 샘플링된 함수들
    samples = sample_gp(K, num_samples=3)
    for s_idx in range(3):
        axes[i, 1].plot(t, samples[:, s_idx], label=f"Sample {s_idx+1}", alpha=0.85, linewidth=1.8)
    
    axes[i, 1].set_title(f"Generated Time Series Samples: {name}", fontsize=10, fontweight='bold')
    axes[i, 1].set_xlabel("Time ($t$)")
    axes[i, 1].set_ylabel("Function Value $y(t)$")
    axes[i, 1].grid(True, linestyle='--', alpha=0.5)
    if i == 0:
        axes[i, 1].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()