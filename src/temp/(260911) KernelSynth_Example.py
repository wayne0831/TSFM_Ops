# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(101)
# L = 400
# t = np.linspace(0, 10, L)

# # 1. 기저 컴포넌트 생성 (Linear, Periodic, White Noise, RQ-Shock)
# lin = 0.6 * t
# per = 2.0 * np.sin(2 * np.pi * t / 1.5)
# wn = np.random.normal(0, 0.4, L)
# shock = np.zeros(L)
# shock[220:] = 3.5  # Mean shift / Structural Break

# # 2. 커널 개수(N=1~5)별 합성 시계열 구성
# # N=1: 단일 주기 커널 (비현실적 순수 사인파)
# y_n1 = per

# # N=2: Periodic + WhiteNoise (단순 계절성)
# y_n2 = per + wn

# # N=3: (Linear * Periodic) + WhiteNoise (진폭 변조 복합 현실 시계열 - 강력 추천)
# amp_mod = (0.2 + 0.15 * t)
# y_n3 = (amp_mod * per) + lin + wn

# # N=3 (Alternative): Linear + Shock(RQ) + WhiteNoise (구조적 단절 현실 시계열)
# y_n3_alt = lin + shock + wn

# # N=4: Linear + Periodic + Shock + WhiteNoise (다중 복합 신호)
# y_n4 = lin + per + shock + wn

# # N=5: 비현실적 과적합 (Linear * Periodic * High-freq Sine + Heavy Shock + Noise)
# chaotic_per = np.sin(2 * np.pi * t / 0.3)
# y_n5 = (0.3 * t * per * chaotic_per) + shock + np.random.normal(0, 1.2, L)

# # 3. 5단계 파형 비교 시각화
# fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)

# configs = [
#     (y_n1, "N=1: Single Kernel [Per] (Pure Sine - Unrealistic)", "tab:gray"),
#     (y_n2, "N=2: 2 Kernels [Per + WN] (Simple Seasonal with Noise)", "tab:blue"),
#     (y_n3, "N=3: 3 Kernels [(Lin * Per) + WN] (Realistic: Modulated Seasonality & Trend)", "tab:green"),
#     (y_n4, "N=4: 4 Kernels [Lin + Per + Shock + WN] (Complex Industrial Signal)", "tab:orange"),
#     (y_n5, "N=5: 5 Kernels Over-composed (Chaotic Beatings & High Entropy - Unrealistic)", "tab:red")
# ]

# for idx, (sig, title, color) in enumerate(configs):
#     axes[idx].plot(sig, color=color, lw=1.4)
#     axes[idx].set_title(title, fontsize=11, fontweight='bold', loc='left')
#     axes[idx].grid(True, linestyle='--', alpha=0.5)
#     axes[idx].margins(x=0.01)

# axes[4].set_xlabel("Time Index (t)", fontsize=11)
# plt.tight_layout()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(42)
# length = 300
# t = np.linspace(0, 30, length)
# jitter = 1e-5

# # 1. 거리 제곱 행렬 계산: (t_i - t_j)^2
# dist_sq = (t[:, None] - t[None, :]) ** 2

# # 2. RBF 커널 샘플링 함수 (l 값 제어)
# def sample_rbf(l_val):
#     cov = np.exp(-dist_sq / (2.0 * (l_val ** 2))) + np.eye(length) * jitter
#     return np.random.multivariate_normal(np.zeros(length), cov)

# # 3. RQ 커널 샘플링 함수 (alpha 값 제어, 기본 l=1.0 고정)
# def sample_rq(alpha_val, l_val=1.0):
#     cov = (1.0 + dist_sq / (2.0 * alpha_val * (l_val ** 2))) ** (-alpha_val) + np.eye(length) * jitter
#     return np.random.multivariate_normal(np.zeros(length), cov)

# # 4. 파라미터 조건별 샘플 생성
# rbf_samples = [
#     (0.1, sample_rbf(0.1), "tab:blue"),
#     (1.0, sample_rbf(1.0), "tab:green"),
#     (10.0, sample_rbf(10.0), "tab:red")
# ]

# rq_samples = [
#     (0.1, sample_rq(0.1), "tab:purple"),
#     (1.0, sample_rq(1.0), "tab:orange"),
#     (10.0, sample_rq(10.0), "tab:brown")
# ]

# # 5. 2행 3열 서브플롯 시각화
# fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)

# # [Row 1] RBF: l에 따른 파형 비교
# for idx, (l_val, series, color) in enumerate(rbf_samples):
#     ax = axes[0, idx]
#     ax.plot(t, series, color=color, lw=1.5)
#     ax.set_title(f"RBF Kernel: $l = {l_val}$", fontsize=12, fontweight='bold')
#     ax.grid(True, linestyle='--', alpha=0.5)
#     if idx == 0:
#         ax.set_ylabel("RBF Waveforms", fontsize=11, fontweight='bold')

# # [Row 2] RQ: alpha에 따른 파형 비교
# for idx, (alpha_val, series, color) in enumerate(rq_samples):
#     ax = axes[1, idx]
#     ax.plot(t, series, color=color, lw=1.5)
#     ax.set_title(f"RQ Kernel: $\\alpha = {alpha_val}$ ($l=1.0$)", fontsize=12, fontweight='bold')
#     ax.grid(True, linestyle='--', alpha=0.5)
#     ax.set_xlabel("Time Index ($t$)", fontsize=10)
#     if idx == 0:
#         ax.set_ylabel("RQ Waveforms", fontsize=11, fontweight='bold')

# plt.suptitle("Impact of Hyperparameters on GP Prior Realizations (RBF vs. RQ)", fontsize=14, fontweight='bold', y=0.98)
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 1. 시뮬레이션 환경 설정
np.random.seed(42)
length = 400
t = np.linspace(0, 20, length)
l_scale = 1.0
sigma_sq = 1.0
jitter = 1e-6

# 시점 간 유클리드 거리 제곱 행렬: (t_i - t_j)^2
dist_sq = (t[:, None] - t[None, :]) ** 2

# 2. 비교 대상 alpha 파라미터 리스트
alpha_candidates = [0.05, 0.5, 2.0, 10.0, 100.0]
colors = ['#8e44ad', '#2980b9', '#16a085', '#d35400', '#c0392b']

# 3. 5행 1열 시계열 파형 플롯 렌더링
fig, axes = plt.subplots(len(alpha_candidates), 1, figsize=(14, 13), sharex=True)

for idx, alpha in enumerate(alpha_candidates):
    # RQ 공분산 행렬 연산
    cov = sigma_sq * ((1.0 + dist_sq / (2.0 * alpha * (l_scale ** 2))) ** (-alpha))
    cov += np.eye(length) * jitter  # 수치적 안정성 확보
    
    # 해당 커널에서 3개의 가상 시계열 샘플 추출
    samples = np.random.multivariate_normal(mean=np.zeros(length), cov=cov, size=3)
    
    ax = axes[idx]
    # 메인 시계열 렌더링 (굵은 선)
    ax.plot(t, samples[0], color=colors[idx], lw=1.8, label=f"Trajectory 1 (Main)")
    # 배경 시계열 렌더링 (투명도 부여로 파형 다양성 제시)
    ax.plot(t, samples[1], color=colors[idx], lw=1.0, alpha=0.45, linestyle='--')
    ax.plot(t, samples[2], color=colors[idx], lw=1.0, alpha=0.45, linestyle=':')
    
    # 서브플롯 타이틀 및 메타 정보 표기
    if alpha < 1.0:
        desc = "Heavy-tailed Multi-scale (Sharp Spikes & Local Shocks)"
    elif alpha <= 2.0:
        desc = "Intermediate Multi-scale (Macro Drift + Local Variations)"
    else:
        desc = "Smooth Convergence to RBF (Pure Non-linear Trend)"
        
    ax.set_title(f"RQ Kernel ($\ell={l_scale}$, $\sigma^2={sigma_sq}$) | $\\alpha = {alpha}$ ➔ {desc}", 
                 fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.margins(x=0.01)

axes[-1].set_xlabel("Time Axis ($t$)", fontsize=11, fontweight='bold')
plt.suptitle("Transition of Time Series Waveforms Generated by RQ Kernel across Scale Mixture Parameter ($\\alpha$)", 
             fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()