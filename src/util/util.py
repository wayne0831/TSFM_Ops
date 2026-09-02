import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from PyEMD import EMD
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. FFT 기반 동적 주기 추정기
def estimate_period_fft(series, default_period=24):
    """
    FFT를 활용하여 시계열 데이터에서 가장 지배적인 주기를 동적으로 추정합니다.
    """
    n = len(series)
    t = np.arange(n)
    
    # 1. FFT 전처리: 선형 추세 및 평균(DC 성분) 제거
    p = np.polyfit(t, series, 1)
    detrended = series - np.polyval(p, t)
    detrended = detrended - np.mean(detrended)
    
    # 2. FFT 수행 및 진폭 계산
    fft_vals = np.fft.rfft(detrended)
    frequencies = np.fft.rfftfreq(n)
    magnitudes = np.abs(fft_vals)
    
    # 3. 0Hz(상수항)를 제외하고 진폭이 가장 큰 인덱스 탐색
    if len(magnitudes) > 1:
        dominant_idx = np.argmax(magnitudes[1:]) + 1 
        dominant_freq = frequencies[dominant_idx]
        
        if dominant_freq > 0:
            period = int(np.round(1.0 / dominant_freq))
            if 2 <= period <= n // 2:
                return period
                
    return default_period

# =====================================================================
# 2. STL-EMD 기반 시계열 3차원 강도(Strength) 산출기
# =====================================================================
def calculate_time_series_strength(series: np.ndarray):
    """
    1. 고전적 STL 분해: y = T + S + R_stl -> F_T_STL, F_S_STL 산출
    2. STL-EMD 하이브리드 분해: y = T + S + I + R_pure -> F_T_STL_EMD, F_S_STL_EMD, F_I_STL_EMD 산출
    """
    estimated_p = estimate_period_fft(series)
    
    # 주기가 데이터 길이의 절반을 넘지 않도록 조정
    if len(series) < 2 * estimated_p:
        estimated_p = max(2, len(series) // 2 - 1)
        
    try:
        # [1단계] 고전적 STL 분해 수행 (y = T + S + R_stl)
        res = STL(series, period=estimated_p, robust=True).fit()
        T, S, R_stl = res.trend, res.seasonal, res.resid
        
        var_R_stl = np.var(R_stl)
        var_TR_stl = np.var(T + R_stl)
        var_SR_stl = np.var(S + R_stl)
        
        # STL 기반 피처 연산
        F_T_STL = max(0.0, 1.0 - (var_R_stl / var_TR_stl if var_TR_stl > 0 else 1.0))
        F_S_STL = max(0.0, 1.0 - (var_R_stl / var_SR_stl if var_SR_stl > 0 else 1.0))
        F_R_STL = max(0.0, 1.0 - (1 / var_R_stl if var_R_stl > 0 else 1.0))

        # [2단계] EMD 분해 기동 (R_stl을 I + R_pure로 분리)
        emd = EMD()
        imfs = emd.emd(R_stl)
        
        # IMF 성분 중 고주파/노이즈(주로 IMF1)를 제외한 나머지를 개입(I) 성분으로 집계
        if len(imfs) > 2:
            I = np.sum(imfs[1:], axis=0) 
        elif len(imfs) == 2:
            I = imfs[1]
        else:
            I = np.zeros_like(R_stl)
            
        # 순수 가우시안 확률 노이즈 정제 (y = T + S + I + R_pure)
        R_pure = R_stl - I
        
        # 최종 정제된 R_pure 기반의 강도 재연산
        var_R_pure = np.var(R_pure)
        var_TR_pure = np.var(T + R_pure) 
        var_SR_pure = np.var(S + R_pure)
        var_IR_pure = np.var(I + R_pure) 
        
        # STL-EMD 기반 피처 연산
        F_T_STL_EMD = max(0.0, 1.0 - (var_R_pure / var_TR_pure if var_TR_pure > 0 else 1.0))
        F_S_STL_EMD = max(0.0, 1.0 - (var_R_pure / var_SR_pure if var_SR_pure > 0 else 1.0))
        F_I_STL_EMD = max(0.0, 1.0 - (var_R_pure / var_IR_pure if var_IR_pure > 0 else 1.0))
            
    except Exception:
        # 수리적 예외 발생 시 0.0 반환
        F_T_STL, F_S_STL, F_R_STL = 0.0, 0.0, 0.0
        F_T_STL_EMD, F_S_STL_EMD, F_I_STL_EMD = 0.0, 0.0, 0.0
        
    return F_T_STL, F_S_STL, F_R_STL, F_T_STL_EMD, F_S_STL_EMD, F_I_STL_EMD


# =====================================================================
# 3. KernelSynth 기본 커널 및 클래스 정의 (연구자님 코드 기반 축약)
# =====================================================================
class Kernel:
    def __init__(self, name: str, params: Dict[str, Any], expr: str):
        self.name, self.params, self.expr = name, params, expr
    def __call__(self, x1, x2): raise NotImplementedError
    def __add__(self, other): return CombinedKernel(self, other, op="+")
    def __mul__(self, other): return CombinedKernel(self, other, op="*")

class CombinedKernel(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel, op: str):
        self.k1, self.k2, self.op = k1, k2, op
        super().__init__(name="Combined", params={}, expr=f"({k1.expr}{op}{k2.expr})")
    def __call__(self, x1, x2):
        if self.op == "+": return self.k1(x1, x2) + self.k2(x1, x2)
        elif self.op == "*": return self.k1(x1, x2) * self.k2(x1, x2)

class ConstantKernel(Kernel):
    def __init__(self, c=1.0):
        super().__init__("Constant", {"c": c}, f"Const({c:g})")
        self.c = c
    def __call__(self, x1, x2): return np.full((len(x1), len(x2)), self.c)

class WhiteNoiseKernel(Kernel):
    def __init__(self, sigma_n):
        super().__init__("WhiteNoise", {"sigma_n": sigma_n}, f"WN(σ={sigma_n:g})")
        self.sigma_n = sigma_n
    def __call__(self, x1, x2): 
        return np.where(np.abs(x1[:, None] - x2[None, :]) < 1e-6, self.sigma_n, 0.0)

class LinearKernel(Kernel):
    def __init__(self, sigma):
        super().__init__("Linear", {"sigma": sigma}, f"Lin(σ={sigma:g})")
        self.sigma = sigma
    def __call__(self, x1, x2): return (self.sigma**2) + np.outer(x1, x2)

class RBFKernel(Kernel):
    def __init__(self, length_scale):
        super().__init__("RBF", {"length_scale": length_scale}, f"RBF(l={length_scale:g})")
        self.l = length_scale
    def __call__(self, x1, x2): 
        return np.exp(-((x1[:, None] - x2[None, :])**2) / (2.0 * (self.l**2)))

class RationalQuadraticKernel(Kernel):
    def __init__(self, alpha, c=1.0):
        super().__init__("RationalQuadratic", {"alpha": alpha, "c": c}, f"RQ(α={alpha:g})")
        self.alpha, self.c = alpha, c
    def __call__(self, x1, x2): 
        return (1.0 + ((x1[:, None] - x2[None, :])**2) / (2.0 * self.alpha))**(-self.c)

class PeriodicKernel(Kernel):
    def __init__(self, period):
        super().__init__("Periodic", {"period": period}, f"Per(p={period:g})")
        self.p = period
    def __call__(self, x1, x2): 
        diff = np.abs(x1[:, None] - x2[None, :])
        return np.exp(-2.0 * (np.sin(np.pi * diff / self.p)**2))

def build_kernel_bank() -> List[Kernel]:
    bank = [ConstantKernel(c=1.0)]
    for s_n in [0.1, 1.0]: bank.append(WhiteNoiseKernel(sigma_n=s_n))
    for s in [0.0, 1.0, 10.0]: bank.append(LinearKernel(sigma=s))
    for l in [0.1, 1.0, 10.0]: bank.append(RBFKernel(length_scale=l))
    for alpha in [0.1, 1.0, 10.0]: bank.append(RationalQuadraticKernel(alpha=alpha))
    periods = [24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730, 4, 26, 52, 6, 12, 40, 10]
    for p in periods: bank.append(PeriodicKernel(period=p))
    return bank

# =====================================================================
# 4. 사전 규칙 기반 정답 라벨링 (Rule-based Sector Mapping)
# =====================================================================
def assign_sector_label(kernels_used: List[str]) -> Tuple[str, str]:
    """커널 구성 요소에 따라 S1~S4 정답(Ground Truth) 섹터를 추론합니다."""
    has_per = any("Periodic" in k for k in kernels_used)
    has_lin = any("Linear" in k for k in kernels_used)
    
    if has_per and has_lin:
        return "S1", "Composite"
    elif has_per and not has_lin:
        return "S2", "Seasonal"
    elif not has_per and has_lin:
        return "S4", "Trending"
    else:
        return "S3", "Stationary"

# =====================================================================
# 5. KernelSynth 데이터 생성 로직
# =====================================================================
def kernel_synth_generate(kernel_bank, max_kernels=5, length=512, jitter=1e-5):
    j = np.random.randint(1, max_kernels + 1)
    selected_kernels = [kernel_bank[idx] for idx in np.random.choice(len(kernel_bank), size=j, replace=True)]
    
    composed_kernel = selected_kernels[0]
    kernels_used = [selected_kernels[0].name]
    operations = []

    for i in range(1, j):
        op = np.random.choice(["+", "*"])
        operations.append(op)
        kernels_used.append(selected_kernels[i].name)
        if op == "+": composed_kernel = composed_kernel + selected_kernels[i]
        else: composed_kernel = composed_kernel * selected_kernels[i]

    t = np.linspace(0, length - 1, length)
    cov_matrix = composed_kernel(t, t) + np.eye(length) * jitter
    synthetic_series = np.random.multivariate_normal(np.zeros(length), cov_matrix)
    
    sector, pattern = assign_sector_label(kernels_used)
    
    meta = {
        "True_Sector": sector,
        "Pattern": pattern,
        "Num_Kernels": j,
        "Kernel_Expression": composed_kernel.expr,
        "Operations": operations if operations else ["None"],
        "Kernels_Used": kernels_used
    }
    return synthetic_series, meta

# =====================================================================
# 6. 메인 실행 블록: 대량 생성 및 STL-EMD 적용, DataFrame 적재
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)
    bank = build_kernel_bank()
    
    NUM_SAMPLES = 5000
    LENGTH      = 512
    records = []
    
    print(f"💡 KernelSynth 데이터 {NUM_SAMPLES}개 생성 및 STL-EMD 3차원 분해 시작...")
    
    for i in range(NUM_SAMPLES):
        # 1. 시계열 데이터 및 메타데이터 생성
        ts, meta = kernel_synth_generate(bank, max_kernels=5, length=LENGTH)
        
        # Min-Max 스케일링 (분해 안정성 확보)
        ts_scaled = (ts - np.min(ts)) / (np.max(ts) - np.min(ts) + 1e-9)
        
        # 2. STL-EMD 강도 추출
        ft_stl, fs_stl, fr_stl, ft_emd, fs_emd, fi_emd = calculate_time_series_strength(ts_scaled)
        
        # 3. 단일 레코드 조합
        record = {
            "True_Sector": meta["True_Sector"],
            "Pattern": meta["Pattern"],
            "F_T_STL": ft_stl,
            "F_S_STL": fs_stl,
            "F_R_STL": fr_stl,
            "F_T_STL_EMD": ft_emd,
            "F_S_STL_EMD": fs_emd,
            "F_I_STL_EMD": fi_emd,
            "Num_Kernels": meta["Num_Kernels"],
            "Kernel_Expression": meta["Kernel_Expression"],
            "Operations": str(meta["Operations"]) # 리스트를 문자열로 저장
        }
        records.append(record)
        
        if (i+1) % 10 == 0:
            print(f"  - {i+1}/{NUM_SAMPLES} 샘플 완료...")

    # 4. 최종 Pandas DataFrame 구성
    df_results = pd.DataFrame(records)
    
    # 5. 지정하신 컬럼 순서대로 정렬
    columns_order = [
        "True_Sector", "Pattern", 
        "F_T_STL", "F_S_STL", "F_R_STL", 
        "F_T_STL_EMD", "F_S_STL_EMD", "F_I_STL_EMD", 
        "Num_Kernels", "Kernel_Expression", "Operations"
    ]
    df_results = df_results[columns_order]

    print("\n✅ 최종 결과 데이터프레임 (상위 5개):")
    display(df_results.head())