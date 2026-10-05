# KernelSynth-based synthetic time series generation pipeline

import os
import time
import warnings
import itertools
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from PyEMD import EMD

# 프로젝트 내부 모듈 import
from src.config import *
from src.util.data_analysis import calculate_time_series_strength, estimate_period_fft

warnings.filterwarnings('ignore')

# =====================================================================
# 1. 4대 기저 커널 클래스 정의 (Constant 제거, RQ 일반화)
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

class LinearKernel(Kernel):
    def __init__(self, sigma: float):
        super().__init__("LIN", {"sigma": sigma}, f"LIN(sigma={sigma:g})")
        self.sigma = sigma
    def __call__(self, x1, x2):
        return (self.sigma ** 2) + np.outer(x1, x2)

class PeriodicKernel(Kernel):
    def __init__(self, period: float):
        super().__init__("PER", {"period": period}, f"PER(period={period:g})")
        self.p = period
    def __call__(self, x1, x2):
        diff = np.abs(x1[:, None] - x2[None, :])
        return np.exp(-2.0 * (np.sin(np.pi * diff / self.p) ** 2))

# class RationalQuadraticKernel(Kernel):
#     """
#     RQ 커널 수식: (1 + diff^2 / (2 * alpha * l^2))^(-alpha)
#     alpha >= 10 일 때 RBF 커널로 수렴하는 거동을 완벽히 모사
#     """
#     def __init__(self, alpha: float, length_scale: float = 1.0):
#         super().__init__("RQ", {"alpha": alpha, "l": length_scale}, f"RQ(alpha={alpha:g},l={length_scale:g})")
#         self.alpha = alpha
#         self.l = length_scale
#     def __call__(self, x1, x2):
#         dist_sq = (x1[:, None] - x2[None, :]) ** 2
#         return (1.0 + dist_sq / (2.0 * self.alpha * (self.l ** 2))) ** (-self.alpha)

class WhiteNoiseKernel(Kernel):
    def __init__(self, sigma_n: float):
        super().__init__("WN", {"sigma_n": sigma_n}, f"WN(sigma={sigma_n:g})")
        self.sigma_n = sigma_n
    def __call__(self, x1, x2):
        diff = np.abs(x1[:, None] - x2[None, :])
        return np.where(diff < 1e-6, self.sigma_n, 0.0)

# =====================================================================
# 2. 파라미터화된 4대 커널 뱅크 구축 (길이 종속적 주기 필터링)
# =====================================================================
def build_kernel_bank(length: int) -> Dict[str, List[Kernel]]:
    #bank = {"LIN": [], "PER": [], "RQ": [], "WN": []}
    bank = {"LIN": [], "PER": [], "WN": []}

    
    # 1. Linear (추세 기울기 분산 제어)
    for s in [0.0, 1.0, 10.0]:
        bank["LIN"].append(LinearKernel(sigma=s))
        
    # 2. Periodic (최소 2사이클 보장: p <= length // 2)
    all_candidate_periods = [4, 6, 7, 10, 12, 14, 24, 26, 30, 48, 52, 60, 96, 168, 240, 336]
    max_valid_period = length // 2
    valid_periods = [p for p in all_candidate_periods if 2 <= p <= max_valid_period]
    if not valid_periods:
        valid_periods = [max(2, length // 4)]
        
    for p in valid_periods:
        bank["PER"].append(PeriodicKernel(period=p))
        
    # 3. Rational Quadratic (쇼크 영역: alpha<=1.0 / RBF 평활 영역: alpha>=10.0)
    # for alpha in [0.1, 0.5, 1.0, 10.0, 50.0]:
    #     for l in [0.5, 1.0, 5.0, 10.0]:
    #         bank["RQ"].append(RationalQuadraticKernel(alpha=alpha, length_scale=l))
            
    # 4. White Noise (잔차 노이즈 강도)
    for s_n in [0.1, 0.5, 1.0]:
        bank["WN"].append(WhiteNoiseKernel(sigma_n=s_n))
        
    return bank

# =====================================================================
# 3. 연산자 전수 조합이 반영된 40대 구조적 커널 케이스 정의
# =====================================================================
def get_all_kernel_cases() -> List[Dict[str, Any]]:
    """
    4대 기저 커널 {LIN, PER, RQ, WN}과 연산자 {+, *}의 모든 가능한 조합 생성
    - N=1: 4개
    - N=2: 6쌍 * 2^1 = 12개
    - N=3: 4트리플 * 2^2 = 16개 (['*', '*'], ['+', '*'] 등 모두 포함)
    - N=4: 1쿼드 * 2^3 = 8개 (['*', '*', '*'] 등 모두 포함)
    총 40개 케이스
    """
    cases = []
    case_id = 1
    base_kernels = ["LIN", "PER", "RQ", "WN"]
    
    # N = 1, 2, 3, 4 순회
    for n in range(1, 5):
        # n개 커널의 조합 (순서 무관 조합)
        for k_tuple in itertools.combinations(base_kernels, n):
            keys = list(k_tuple)
            
            if n == 1:
                cases.append({
                    "Case_ID": case_id,
                    "Num_Kernels": 1,
                    "Keys": keys,
                    "Ops": []
                })
                case_id += 1
            else:
                # n-1개 자리에 올 수 있는 모든 +, * 연산자 카테시안 곱 (2^(n-1)개)
                all_ops_combos = list(itertools.product(["+", "*"], repeat=n - 1))
                for ops in all_ops_combos:
                    cases.append({
                        "Case_ID": case_id,
                        "Num_Kernels": n,
                        "Keys": keys,
                        "Ops": list(ops)
                    })
                    case_id += 1
                    
    return cases

# =====================================================================
# 4. 섹터 및 패턴 자동 할당 함수
# =====================================================================
def assign_sector_label(kernels_used: List[str]) -> Tuple[str, str]:
    has_per = any("PER" in k for k in kernels_used)
    has_lin = any("LIN" in k for k in kernels_used)
    
    if has_per and has_lin:
        return "S1", "Composite"
    elif has_per and not has_lin:
        return "S2", "Seasonal"
    elif not has_per and has_lin:
        return "S4", "Trending"
    else:
        return "S3", "Stationary"

# =====================================================================
# 5. 케이스 기반 시계열 GP 샘플러
# =====================================================================
def sample_from_case(case: Dict[str, Any], bank: Dict[str, List[Kernel]], length: int, jitter: float = 1e-5):
    # 해당 커널 키에 맞춰 뱅크에서 무작위 파라미터 인스턴스 1개씩 추출
    selected = [np.random.choice(bank[k]) for k in case["Keys"]]
    
    composed = selected[0]
    for i in range(1, len(selected)):
        op = case["Ops"][i - 1]
        composed = (composed + selected[i]) if op == "+" else (composed * selected[i])
        
    t = np.linspace(0, length - 1, length)
    cov_matrix = composed(t, t) + np.eye(length) * jitter
    series = np.random.multivariate_normal(np.zeros(length), cov_matrix)
    
    return series, composed.expr

