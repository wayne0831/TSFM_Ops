import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

# =====================================================================
# 1. 메타데이터(이름, 하이퍼파라미터, 수식 표기)를 추적하는 기본 커널 정의
# =====================================================================
class Kernel:
    """
    모든 커널의 추상 베이스 클래스 (Base Class)
    - name: 커널 종류 이름 (예: 'Linear', 'Periodic')
    - params: 해당 커널에 적용된 하이퍼파라미터 딕셔너리 (예: {'sigma': 1.0})
    - expr: 플롯 타이틀 및 시각화용 축약 수식 문자열 (예: 'Lin(σ=1)')
    """
    def __init__(self, name: str, params: Dict[str, Any], expr: str):
        self.name = name
        self.params = params
        self.expr = expr

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        두 시점 벡터 x1(N,), x2(N,) 사이의 공분산 행렬 (N, N)을 계산하는 추상 메서드
        """
        raise NotImplementedError

    def __add__(self, other: "Kernel") -> "CombinedKernel":
        """
        덧셈 연산자(+) 오버로딩: 두 커널을 더하여 독립 패턴의 중첩(Superposition)을 생성
        """
        return CombinedKernel(self, other, op="+")

    def __mul__(self, other: "Kernel") -> "CombinedKernel":
        """
        곱셈 연산자(*) 오버로딩: 두 커널을 곱하여 패턴 간의 상호작용 및 진폭 변조(Modulation)를 생성
        """
        return CombinedKernel(self, other, op="*")


class CombinedKernel(Kernel):
    """
    두 커널을 덧셈(+) 또는 곱셈(*)으로 결합한 복합 커널 클래스 (트리 구조)
    """
    def __init__(self, k1: Kernel, k2: Kernel, op: str):
        self.k1 = k1
        self.k2 = k2
        self.op = op
        # 하위 커널들의 수식을 괄호로 묶어 결합된 수식 문자열을 재귀적으로 생성
        expr = f"({k1.expr}{op}{k2.expr})"
        super().__init__(name="Combined", params={}, expr=expr)

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # 덧셈 결합: 두 공분산 행렬의 원소별 합 (Cov_A + Cov_B)
        if self.op == "+":
            return self.k1(x1, x2) + self.k2(x1, x2)
        # 곱셈 결합: 두 공분산 행렬의 원소별 곱 (Hadamard Product, Cov_A * Cov_B)
        elif self.op == "*":
            return self.k1(x1, x2) * self.k2(x1, x2)
        raise ValueError(f"Unknown operation: {self.op}")


class ConstantKernel(Kernel):
    """
    Constant Kernel: K(x, x') = C
    - 시계열의 전체적인 기준 오프셋/상수 레벨을 결정
    """
    def __init__(self, c: float = 1.0):
        super().__init__(name="Constant", params={"c": c}, expr=f"Const({c:g})")
        self.c = c

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # 모든 시점 쌍 간의 공분산 값을 고정 상수 C로 채움 (N x N 행렬)
        return np.full((len(x1), len(x2)), self.c)


class WhiteNoiseKernel(Kernel):
    """
    White Noise Kernel: K(x, x') = sigma_n * I(x == x')
    - 시점 간 상관관계가 없는 독립적인 불규칙 잡음(Noise) 모델링
    """
    def __init__(self, sigma_n: float):
        super().__init__(name="WhiteNoise", params={"sigma_n": sigma_n}, expr=f"WN(σ={sigma_n:g})")
        self.sigma_n = sigma_n

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # x1과 x2 시점 간의 절대 차이를 브로드캐스팅으로 계산 (N x N)
        diff = np.abs(x1[:, None] - x2[None, :])
        # 대각 성분(동일 시점, diff ≈ 0)에만 노이즈 분산(sigma_n)을 부여하고 나머지는 0
        return np.where(diff < 1e-6, self.sigma_n, 0.0)


class LinearKernel(Kernel):
    """
    Linear Kernel: K(x, x') = sigma^2 + x * x'
    - 일정한 기울기로 지속 상승/하강하는 선형 추세(Trend) 모델링
    """
    def __init__(self, sigma: float):
        super().__init__(name="Linear", params={"sigma": sigma}, expr=f"Lin(σ={sigma:g})")
        self.sigma = sigma

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # 두 시점 벡터의 외적(Outer Product)을 통해 x * x' 행렬을 생성하고 오프셋 sigma^2을 가산
        return (self.sigma ** 2) + np.outer(x1, x2)


class RBFKernel(Kernel):
    """
    RBF (Radial Basis Function / Squared Exponential) Kernel:
    K(x, x') = exp(-||x - x'||^2 / (2 * l^2))
    - 미분 가능하며 매우 매끄럽고 완만한 국소 변동성(Local Variation) 모델링
    """
    def __init__(self, length_scale: float):
        super().__init__(name="RBF", params={"length_scale": length_scale}, expr=f"RBF(l={length_scale:g})")
        self.l = length_scale

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # 시점 간의 유클리드 거리 제곱 행렬 계산: (t_i - t_j)^2
        dist_sq = (x1[:, None] - x2[None, :]) ** 2
        # 길이 스케일(l)에 따른 지수 감쇠 적용
        return np.exp(-dist_sq / (2.0 * (self.l ** 2)))


class RationalQuadraticKernel(Kernel):
    """
    Rational Quadratic Kernel:
    K(x, x') = (1 + ||x - x'||^2 / (2 * alpha * l^2))^(-alpha)
    - 서로 다른 여러 길이 스케일(Multi-scale)의 RBF 커널이 혼합된 형태의 부드러운 곡선 모델링
    """
    def __init__(self, alpha: float, c: float = 1.0):
        super().__init__(name="RationalQuadratic", params={"alpha": alpha, "c": c}, expr=f"RQ(α={alpha:g})")
        self.alpha = alpha
        self.c = c

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        dist_sq = (x1[:, None] - x2[None, :]) ** 2
        return (1.0 + dist_sq / (2.0 * self.alpha)) ** (-self.c)


class PeriodicKernel(Kernel):
    """
    Periodic Kernel:
    K(x, x') = exp(-2 * sin^2(pi * ||x - x'|| / p))
    - 엄격하게 주기 p마다 동일한 파형이 반복되는 계절성(Seasonality) 모델링
    """
    def __init__(self, period: float):
        super().__init__(name="Periodic", params={"period": period}, expr=f"Per(p={period:g})")
        self.p = period

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        diff = np.abs(x1[:, None] - x2[None, :])
        # 주기 p를 기반으로 한 삼각함수 파동 변환
        sin_val = np.sin(np.pi * diff / self.p)
        return np.exp(-2.0 * (sin_val ** 2))


# =====================================================================
# 2. Kernel Bank 생성 함수 (Chronos Table 2에 정의된 총 31개 기본 커널)
# =====================================================================
def build_kernel_bank() -> List[Kernel]:
    """
    논문 Table 2의 커널 종류 및 하이퍼파라미터 조합을 1:1 인스턴스로 생성하여 반환 (총 31개)
    """
    bank: List[Kernel] = []
    
    # 1. Constant 커널 (1개: C=1.0)
    bank.append(ConstantKernel(c=1.0))
    
    # 2. White Noise 커널 (2개: sigma_n in {0.1, 1.0})
    for s_n in [0.1, 1.0]:
        bank.append(WhiteNoiseKernel(sigma_n=s_n))
        
    # 3. Linear 커널 (3개: sigma in {0.0, 1.0, 10.0})
    for s in [0.0, 1.0, 10.0]:
        bank.append(LinearKernel(sigma=s))
        
    # 4. RBF 커널 (3개: length_scale in {0.1, 1.0, 10.0})
    for l in [0.1, 1.0, 10.0]:
        bank.append(RBFKernel(length_scale=l))
        
    # 5. Rational Quadratic 커널 (3개: alpha in {0.1, 1.0, 10.0})
    for alpha in [0.1, 1.0, 10.0]:
        bank.append(RationalQuadraticKernel(alpha=alpha))
        
    # 6. Periodic 커널 (19개: 일, 주, 월, 분기, 연 단위 주기를 나타내는 19개 주기 p)
    periods = [24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730, 4, 26, 52, 6, 12, 40, 10]
    for p in periods:
        bank.append(PeriodicKernel(period=p))
        
    # 총 1 + 2 + 3 + 3 + 3 + 19 = 31개
    return bank


# =====================================================================
# 3. Algorithm 2: 메타데이터 추적형 KernelSynth 시계열 생성 함수
# =====================================================================
def kernel_synth(
    kernel_bank: List[Kernel],
    sample_id: int,
    max_kernels: int = 5,
    length: int = 512,
    jitter: float = 1e-5
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    가우시안 프로세스 사전분포(GP Prior)로부터 합성 시계열 1개와 조합 메타데이터를 함께 생성
    - sample_id: 생성할 샘플의 식별 번호 (1 ~ 25)
    - max_kernels: 수식 조합 시 결합할 수 있는 최대 커널 개수 (기본값 J=5)
    - length: 생성할 시계열 타임스텝 길이 (L=512)
    - jitter: Cholesky 분해 시 수치적 양의 준정부호성을 보장하기 위한 대각 안정화 상수
    """
    # -------------------------------------------------------------
    # Step 1: 조합할 커널 개수 j ~ U{1, J} 무작위 추출
    # -------------------------------------------------------------
    j = np.random.randint(1, max_kernels + 1)

    # -------------------------------------------------------------
    # Step 2: 31개 커널 뱅크에서 j개의 기본 커널 복원 추출 (Sampling with replacement)
    # -------------------------------------------------------------
    selected_indices = np.random.choice(len(kernel_bank), size=j, replace=True)
    selected_kernels = [kernel_bank[idx] for idx in selected_indices]

    # -------------------------------------------------------------
    # Step 3: 첫 번째 커널로 수식 및 메타데이터 초기화
    # -------------------------------------------------------------
    composed_kernel = selected_kernels[0]
    kernels_used = [selected_kernels[0].name]
    hyperparameters = [selected_kernels[0].params]
    operations = []

    # -------------------------------------------------------------
    # Step 4~7: 이항 연산자(+, *)를 무작위 선택하여 커널을 순차 합성
    # -------------------------------------------------------------
    for i in range(1, j):
        # +, * 중 하나를 50% 확률로 무작위 선택
        op = np.random.choice(["+", "*"])
        operations.append(op)
        kernels_used.append(selected_kernels[i].name)
        hyperparameters.append(selected_kernels[i].params)

        # 연산자 오버로딩에 의해 CombinedKernel 객체 트리 생성
        if op == "+":
            composed_kernel = composed_kernel + selected_kernels[i]
        else:
            composed_kernel = composed_kernel * selected_kernels[i]

    # -------------------------------------------------------------
    # Step 8: GP Prior: N(0, K*)에서 길이 length의 시계열 샘플링
    # -------------------------------------------------------------
    # 시간 축 인덱스 벡터 t = [0, 1, 2, ..., length-1]
    t = np.linspace(0, length - 1, length)

    # 조합된 복합 커널 함수를 사용해 (length x length) 공분산 행렬 K* 계산
    cov_matrix = composed_kernel(t, t)

    # 부동소수점 오차로 인한 고유값 왜곡을 방지하기 위해 대각선에 미세한 Jitter 추가
    cov_matrix += np.eye(length) * jitter

    # 평균이 0인 다변량 정규분포 N(0, cov_matrix)에서 1개의 난수 벡터 샘플링
    mean_vector = np.zeros(length)
    synthetic_series = np.random.multivariate_normal(mean_vector, cov_matrix)

    # -------------------------------------------------------------
    # Step 9: DataFrame 기록용 메타데이터 딕셔너리 구성
    # -------------------------------------------------------------
    meta = {
        "Sample_ID": sample_id,
        "Num_Kernels": j,
        "Kernel_Expression": composed_kernel.expr,
        "Kernels_Used": kernels_used,
        "Operations": operations if operations else ["None"],
        "Hyperparameters": hyperparameters
    }
    return synthetic_series, meta


# =====================================================================
# 4. 실행 및 시각화 테스트 (25개 샘플 생성, DataFrame 구축, 5x5 Grid Plot)
# =====================================================================
if __name__ == "__main__":
    # 재현성을 위한 난수 생성기 시드 고정
    np.random.seed(42)

    # 31개 기본 커널 뱅크 인스턴스 빌드
    bank = build_kernel_bank()

    samples_data = []      # 생성된 시계열 수치 배열을 저장할 리스트
    metadata_records = []  # 커널 조합 메타데이터를 저장할 리스트
    num_samples = 49       # 8x8 그리드 생성을 위한 총 샘플 수

    # 64회 반복하며 가상 시계열 및 메타데이터 추출
    for i in range(1, num_samples + 1):
        ts, meta = kernel_synth(bank, sample_id=i, max_kernels=5, length=512)
        samples_data.append(ts)
        metadata_records.append(meta)

    # -----------------------------------------------------------------
    # (1) 조합 메타데이터를 Pandas DataFrame으로 변환 및 검증
    # -----------------------------------------------------------------
    df_kernels = pd.DataFrame(metadata_records)

    print("=== KernelSynth Generation Metadata DataFrame ===")
    # 주요 메타 컬럼(샘플ID, 커널수, 최종수식, 적용연산자) 상위 5개 출력
    print(df_kernels[["Sample_ID", "Num_Kernels", "Kernel_Expression", "Operations"]].head(100))
    print("\n전체 컬럼 목록:", list(df_kernels.columns))

    # -----------------------------------------------------------------
    # (2) 5x5 서브플롯 그리드 생성 및 Title에 커널 조합 수식 매핑
    # -----------------------------------------------------------------
    # 가로 20인치, 세로 12인치의 여유 있는 캔버스 생성 (X축 공유)
    fig, axes = plt.subplots(7, 7, figsize=(20, 12), sharex=True)
    axes = axes.flatten()  # 2차원 축 배열(5, 5)을 1차원(25,)으로 펼쳐 인덱싱 편의성 확보

    for idx in range(num_samples):
        ax = axes[idx]
        ts = samples_data[idx]
        expr = metadata_records[idx]["Kernel_Expression"]

        # 시계열 라인 플롯 렌더링
        ax.plot(ts, color="tab:blue", lw=1.1)

        # 서브플롯 타이틀에 샘플 번호와 축약된 커널 조합 수식을 함께 명시
        # 폭이 좁은 5x5 환경에서 텍스트 겹침을 방지하기 위해 폰트 크기 7.5pt 지정
        ax.set_title(f"#{idx+1}: {expr}", fontsize=7.5, fontweight="bold", pad=4)
        #ax.set_title(f"{expr}", fontsize=7.5, fontweight="bold", pad=4)

        # 가독성을 높이기 위한 반투명 보조선 및 눈금 폰트 조정
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)

    # 서브플롯 간 여백 자동 조정 후 화면 출력
    plt.tight_layout()
    plt.show()