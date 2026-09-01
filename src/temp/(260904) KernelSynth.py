import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple

# =====================================================================
# 1. Table 2: 기본 커널(Basis Kernel) 정의
# =====================================================================
class Kernel:
    """커널 베이스 클래스 (덧셈 및 곱셈 연산자 오버로딩 지원)"""
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __add__(self, other):
        return CombinedKernel(self, other, op="+")

    def __mul__(self, other):
        return CombinedKernel(self, other, op="*")


class CombinedKernel(Kernel):
    """두 커널을 덧셈(+) 또는 곱셈(*)으로 합성한 복합 커널"""
    def __init__(self, k1: Kernel, k2: Kernel, op: str):
        self.k1 = k1
        self.k2 = k2
        self.op = op

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self.op == "+":
            return self.k1(x1, x2) + self.k2(x1, x2)
        elif self.op == "*":
            return self.k1(x1, x2) * self.k2(x1, x2)
        raise ValueError(f"Unknown operation: {self.op}")


class ConstantKernel(Kernel):
    """Constant Kernel: K(x, x') = C (C=1)"""
    def __init__(self, c: float = 1.0):
        self.c = c

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.full((len(x1), len(x2)), self.c)


class WhiteNoiseKernel(Kernel):
    """White Noise Kernel: K(x, x') = sigma_n * I(x == x')"""
    def __init__(self, sigma_n: float):
        self.sigma_n = sigma_n

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # 두 시점의 차이가 거의 0인 대각 성분에 노이즈 분산 부여
        diff = np.abs(x1[:, None] - x2[None, :])
        return np.where(diff < 1e-6, self.sigma_n, 0.0)


class LinearKernel(Kernel):
    """Linear Kernel: K(x, x') = sigma^2 + x * x'"""
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return (self.sigma ** 2) + np.outer(x1, x2)


class RBFKernel(Kernel):
    """RBF (Gaussian) Kernel: K(x, x') = exp(-||x - x'||^2 / (2 * l^2))"""
    def __init__(self, length_scale: float):
        self.l = length_scale

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        dist_sq = (x1[:, None] - x2[None, :]) ** 2
        return np.exp(-dist_sq / (2 * (self.l ** 2)))


class RationalQuadraticKernel(Kernel):
    """Rational Quadratic Kernel: K(x, x') = (1 + ||x - x'||^2 / (2 * alpha))^(-c)"""
    def __init__(self, alpha: float, c: float = 1.0):
        self.alpha = alpha
        self.c = c

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        dist_sq = (x1[:, None] - x2[None, :]) ** 2
        return (1.0 + dist_sq / (2.0 * self.alpha)) ** (-self.c)


class PeriodicKernel(Kernel):
    """Periodic Kernel: K(x, x') = exp(-2 * sin^2(pi * ||x - x'|| / p))"""
    def __init__(self, period: float):
        self.p = period

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        diff = np.abs(x1[:, None] - x2[None, :])
        sin_val = np.sin(np.pi * diff / self.p)
        return np.exp(-2.0 * (sin_val ** 2))


# =====================================================================
# 2. Kernel Bank 생성 (Table 2 하이퍼파라미터 풀)
# =====================================================================
def build_kernel_bank() -> List[Kernel]:
    kernel_bank: List[Kernel] = []

    # 1. Constant (C=1)
    kernel_bank.append(ConstantKernel(c=1.0))

    # 2. White Noise (sigma_n in {0.1, 1.0})
    for s_n in [0.1, 1.0]:
        kernel_bank.append(WhiteNoiseKernel(sigma_n=s_n))

    # 3. Linear (sigma in {0, 1, 10})
    for s in [0.0, 1.0, 10.0]:
        kernel_bank.append(LinearKernel(sigma=s))

    # 4. RBF (length_scale in {0.1, 1.0, 10.0})
    for l in [0.1, 1.0, 10.0]:
        kernel_bank.append(RBFKernel(length_scale=l))

    # 5. Rational Quadratic (alpha in {0.1, 1.0, 10.0})
    for alpha in [0.1, 1.0, 10.0]:
        kernel_bank.append(RationalQuadraticKernel(alpha=alpha))

    # 6. Periodic (periods for various frequencies)
    periods = [
        24, 48, 96, 168, 336, 672,
        7, 14, 30, 60, 365, 730,
        4, 26, 52, 6, 12, 40, 10
    ]
    for p in periods:
        kernel_bank.append(PeriodicKernel(period=p))

    return kernel_bank


# =====================================================================
# 3. Algorithm 2: KernelSynth 시계열 생성 함수
# =====================================================================
def kernel_synth(
    kernel_bank: List[Kernel],
    max_kernels: int = 5,
    length: int = 1024,
    jitter: float = 1e-6
) -> np.ndarray:
    """
    Gaussian Process Prior 기반 합성 시계열 생성 (Algorithm 2)
    """
    # 1. 조합할 커널 개수 j ~ U{1, J}
    j = np.random.randint(1, max_kernels + 1)

    # 2. 커널 뱅크에서 j개의 기본 커널 복원 추출
    selected_indices = np.random.choice(len(kernel_bank), size=j, replace=True)
    selected_kernels = [kernel_bank[idx] for idx in selected_indices]

    # 3. 첫 번째 커널로 초기화
    composed_kernel = selected_kernels[0]

    # 4~7. 무작위 이항 연산자(+, *)를 이용해 커널 합성
    for i in range(1, j):
        op = np.random.choice(["+", "*"])
        if op == "+":
            composed_kernel = composed_kernel + selected_kernels[i]
        else:
            composed_kernel = composed_kernel * selected_kernels[i]

    # 시간 인덱스 t (0 ~ length-1)
    t = np.linspace(0, length - 1, length)

    # 공분산 행렬 계산
    cov_matrix = composed_kernel(t, t)

    # 수치적 안정성을 위해 대각 행렬에 약간의 Jitter 추가
    cov_matrix += np.eye(length) * jitter

    # 8. GP Prior: N(0, K*)에서 샘플링 (평균 = 0)
    mean_vector = np.zeros(length)
    synthetic_series = np.random.multivariate_normal(mean_vector, cov_matrix)

    return synthetic_series


# =====================================================================
# 4. 실행 및 시각화 테스트
# =====================================================================
if __name__ == "__main__":
    # 재현성을 위한 시드 설정
    np.random.seed(42)

    # 커널 뱅크 빌드
    bank = build_kernel_bank()
    print(f"Total basis kernels in bank: {len(bank)}")

    # 4개의 합성 시계열 샘플 생성 및 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True)
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        # 길이 512의 합성 시계열 생성 (논문 기본값은 1024)
        ts = kernel_synth(bank, max_kernels=5, length=512)
        ax.plot(ts, color="tab:blue", lw=1.2)
        ax.set_title(f"KernelSynth Sample #{idx + 1}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()