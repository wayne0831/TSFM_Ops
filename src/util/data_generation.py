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

class RationalQuadraticKernel(Kernel):
    """
    RQ 커널 수식: (1 + diff^2 / (2 * alpha * l^2))^(-alpha)
    alpha >= 10 일 때 RBF 커널로 수렴하는 거동을 완벽히 모사
    """
    def __init__(self, alpha: float, length_scale: float = 1.0):
        super().__init__("RQ", {"alpha": alpha, "l": length_scale}, f"RQ(alpha={alpha:g},l={length_scale:g})")
        self.alpha = alpha
        self.l = length_scale
    def __call__(self, x1, x2):
        dist_sq = (x1[:, None] - x2[None, :]) ** 2
        return (1.0 + dist_sq / (2.0 * self.alpha * (self.l ** 2))) ** (-self.alpha)

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
    bank = {"LIN": [], "PER": [], "RQ": [], "WN": []}
    
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
    for alpha in [0.1, 0.5, 1.0, 10.0, 50.0]:
        for l in [0.5, 1.0, 5.0, 10.0]:
            bank["RQ"].append(RationalQuadraticKernel(alpha=alpha, length_scale=l))
            
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

# =====================================================================
# 6. 메인 실행 파이프라인
# =====================================================================
if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(42)

    # PARAMS 파싱
    num_samples_per_case = int(PARAMS[DATA_GEN_METHOD]["NUM_SAMPLES"])
    raw_lengths = str(PARAMS[DATA_GEN_METHOD]["LENGTH"])
    lengths = [int(l.strip()) for l in raw_lengths.split(",") if l.strip()]

    meta_path = RES_PATH['data_generation'][DATA_GEN_METHOD]['METADATA']
    data_path = RES_PATH['data_generation'][DATA_GEN_METHOD]['DATA']
    
    base_meta_dir = os.path.dirname(meta_path)
    base_data_dir = os.path.dirname(data_path)
    base_meta_name = os.path.splitext(os.path.basename(meta_path))[0]
    base_data_name = os.path.splitext(os.path.basename(data_path))[0]

    os.makedirs(base_meta_dir, exist_ok=True)
    os.makedirs(base_data_dir, exist_ok=True)

    # 연산자가 빠짐없이 전수 반영된 40개 케이스 로드
    cases = get_all_kernel_cases()
    
    print(f"🚀 [KernelSynth] 4대 커널 기반 전수 연산자({len(cases)}개 케이스) 데이터 생성 시작")
    print(f"   - 타겟 길이 목록: {lengths}")
    print(f"   - 케이스당 샘플 수: {num_samples_per_case}개 (길이별 총 {len(cases) * num_samples_per_case}개 샘플)")

    columns_order = [
        "Case_ID", "True_Sector", "Pattern", "LENGTH",
        "F_T_STL", "F_S_STL", "F_R_STL",
        "F_T_STL_EMD", "F_S_STL_EMD", "F_I_STL_EMD",
        "Num_Kernels", "Kernel_Expression", "Operations", "Kernels_Used"
    ]

    # 길이별 순회 루프
    for length in lengths:
        # 동적 커널 뱅크 빌드 (p <= length // 2 보장)
        bank = build_kernel_bank(length=length)
        available_periods = [k.p for k in bank["PER"]]
        print(f"\n▶ Current Sequence Length: {length} (허용된 유효 주기 {len(available_periods)}개: {available_periods})")
        
        len_records = []
        ts_list_for_len = []
        
        for case in cases:
            c_id = case["Case_ID"]
            k_keys = case["Keys"]
            ops = case["Ops"]
            
            # Ground Truth 섹터 및 패턴 도출
            sec, pat = assign_sector_label(kernels_used=k_keys)
            
            for s_idx in range(num_samples_per_case):
                # 1. GP 시계열 샘플링
                ts, expr = sample_from_case(case, bank, length=length)
                
                # 2. Min-Max 정규화
                ts_scaled = (ts - np.min(ts)) / (np.max(ts) - np.min(ts) + 1e-9)
                ts_scaled_f32 = ts_scaled.astype(np.float32)
                ts_list_for_len.append(ts_scaled_f32)
                
                # 3. STL-EMD 3차원 분해 지표 산출
                ft_stl, fs_stl, fr_stl, ft_emd, fs_emd, fi_emd = calculate_time_series_strength(ts_scaled)
                
                # 4. 메타데이터 레코드 적재
                record = {
                    "Case_ID": c_id,
                    "True_Sector": sec,
                    "Pattern": pat,
                    "LENGTH": length,
                    "Num_Kernels": case["Num_Kernels"],
                    "Kernels_Used": str(k_keys),
                    "Operations": str(ops) if ops else "['None']",
                    "Kernel_Expression": expr,
                    "F_T_STL": ft_stl,
                    "F_S_STL": fs_stl,
                    "F_R_STL": fr_stl,
                    "F_T_STL_EMD": ft_emd,
                    "F_S_STL_EMD": fs_emd,
                    "F_I_STL_EMD": fi_emd
                }
                len_records.append(record)
        
        # -------------------------------------------------------------
        # 해당 길이 전용 .npz 시계열 배열 저장
        # -------------------------------------------------------------
        ts_array = np.vstack(ts_list_for_len)
        cur_npz_path = os.path.join(base_data_dir, f"{base_data_name}_len{length}.npz")
        np.savez_compressed(cur_npz_path, time_series=ts_array)
        print(f"   💾 [NPZ 저장] {cur_npz_path} (Shape: {ts_array.shape})")

        # -------------------------------------------------------------
        # 해당 길이 전용 .csv 메타데이터 저장
        # -------------------------------------------------------------
        df_len = pd.DataFrame(len_records)[columns_order]
        cur_csv_path = os.path.join(base_meta_dir, f"{base_meta_name}_len{length}.csv")
        df_len.to_csv(cur_csv_path, index=False)
        print(f"   📄 [CSV 저장] {cur_csv_path} (Rows: {len(df_len)})")
        print(f"   📊 [섹터별 분포] {dict(df_len['True_Sector'].value_counts())}")

    elapsed = (time.time() - start_time) / 60
    print(f"\n✅ 40개 전수 케이스 x {len(lengths)}개 길이 생성이 모두 완료되었습니다.")
    print(f"⏱️ 총 소요 시간: {elapsed:.2f}분")