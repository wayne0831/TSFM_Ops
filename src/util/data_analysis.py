import os
import time
import warnings
import itertools
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from PyEMD import EMD
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

# 프로젝트 내부 모듈 import (사용자 환경)
from src.config import *
warnings.filterwarnings('ignore')

# =====================================================================
# FFT 기반 동적 주기 추정기
# =====================================================================
def estimate_period_fft(series: np.ndarray, default_period: int = 24) -> int:
    """
    FFT(고속 푸리에 변환)를 기반으로 시계열 데이터의 가장 지배적인 주기(Dominant Period)를 동적으로 추정
    
    선형 추세 및 DC(평균) 성분에 의한 저주파 스펙트럼 왜곡을 사전에 제거한 후, 
    0Hz를 제외한 최대 진폭 주파수의 역수를 취해 정수형 주기를 산출
    
    Args:
        series (np.ndarray): 분석 대상 1차원 시계열 데이터
        default_period (int): 유효 주기를 탐색하지 못했을 때 반환할 기본 대체 주기 (기본값: 24)
        
    Returns:
        int: 동적으로 추정된 지배적 주기 (2 <= period <= n // 2 보장)
    """

    n = len(series)
    t = np.arange(n)

    # 1차 다항식(직선) 피팅을 통해 시계열 전반의 선형 경향성 추정
    p = np.polyfit(t, series, 1)

    # 원본 신호에서 선형 추세를 차감하여 기저선을 평탄화 (스펙트럼 누출 및 왜곡 방지)
    detrended = series - np.polyval(p, t)

    # 평균을 0으로 맞추어 주파수 0Hz(DC Component)에 에너지가 집중되는 현상 차단
    detrended = detrended - np.mean(detrended)

    # 실수 입력 신호에 최적화된 rfft 수행
    fft_vals = np.fft.rfft(detrended)

    # 시계열 길이 n에 대응하는 주파수 빈(Frequency Bins) 계산
    frequencies = np.fft.rfftfreq(n)

    # 복소수 계수의 절댓값을 계산하여 각 주파수별 진폭(Magnitude) 산출
    magnitudes = np.abs(fft_vals)

    # 진폭 에너지가 가장 강하게 솟구친 피크 인덱스 추출
    if len(magnitudes) > 1:
        dominant_idx = np.argmax(magnitudes[1:]) + 1 
        dominant_freq = frequencies[dominant_idx]
        if dominant_freq > 0:
            # 주파수와 주기의 반비례 관계(T = 1 / f)에 따라 정수형 주기로 환산
            period = int(np.round(1.0 / dominant_freq))

            # 물리적 유효 주기 조건 검증:
            # 1) 최소 주기 2: 나이퀴스트 표본화 정리에 따른 최소 진동 간격
            # 2) 최대 주기 n // 2: 데이터 길이 내에서 최소 2회 이상의 사이클 완결 보장
            if 2 <= period <= n // 2:
                return period

    # 주기성이 없는 순수 무작위 잡음이거나 경계 조건을 벗어난 경우 기본값 반환[cite: 3]
    return default_period

# =====================================================================
# Wu & Huang (2004) 백색잡음 유의성 검정 기반 동적 EMD 잔차 분해 모듈
# =====================================================================
def decompose_residual_emd(R_stl: np.ndarray, alpha_level: float = 1.645) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wu & Huang (2004) 백색잡음 통계적 유의성 검정을 적용하여 R_stl을 I_t와 R_pure로 동적 분해합니다.
    
    [핵심 수리신호처리 공리]
    1. IMF_1 앵커링: 최고주파 모드인 IMF_1의 (E_1, T_1)을 기준으로 잡음 기준선의 높낮이(Offset C)를 
       적응형 보정하여 전처리(Min-Max, Z-score)에 영향받지 않는 스케일 불변성을 확보합니다.
    2. 95% 단측 신뢰 상한선: 시계열 길이 N과 각 모드의 주기 T_k에 기반한 표본 오차 확산 폭을 적용합니다.
    3. Residue(r_n)의 I_t 귀속: EMD의 최종 잔여항은 진동이 소멸된 초저주파 기저선이므로,
       Mean Shift 등의 계단형 개입(Step Intervention) 신호로 보아 I_t에 통합합니다.
    """
    N = len(R_stl)
    emd = EMD()
    imfs = emd.emd(R_stl)
    
    # 분해 모드가 1개 이하인 경우 (예외 처리)
    if len(imfs) <= 1:
        return np.zeros_like(R_stl), R_stl.copy()
        
    # PyEMD의 마지막 성분은 진동 극값이 소멸된 최종 Residue r_n(t)
    oscillatory_imfs = imfs[:-1]
    residue = imfs[-1]
    num_osc = len(oscillatory_imfs)
    
    if num_osc == 1:
        # 진동 모드가 IMF_1 하나뿐인 경우: IMF_1 -> 순수 잡음, Residue -> 개입 성분
        return residue, oscillatory_imfs[0]
        
    # 1. 각 진동 모드의 분산 에너지(E_k) 및 평균 주기(T_k) 산출
    E_k = np.zeros(num_osc)
    T_k = np.zeros(num_osc)
    
    for k in range(num_osc):
        E_k[k] = np.mean(oscillatory_imfs[k] ** 2)
        peaks, _ = find_peaks(oscillatory_imfs[k])
        num_peaks = len(peaks)
        T_k[k] = N / max(num_peaks, 1)  # 주기 추정 (Zero-division 방지)
        
    ln_E = np.log(np.maximum(E_k, 1e-12))
    ln_T = np.log(np.maximum(T_k, 1e-12))
    
    # 2. Step 2: IMF_1 기준 적응형 스케일 보정 (Anchoring Offset C)
    offset_C = ln_E[0] + ln_T[0]
    
    # 3. 95% 신뢰구간 상한선(Upper Limit) 산출 및 동적 모드 판정
    k_signal = []
    k_noise = [0]  # 최고주파 IMF_1은 물리적 기저 잡음으로 앵커링 (Fallback 보장)
    
    for k in range(1, num_osc):
        # Wu & Huang (2004) 해석적 확산 함수
        spread = alpha_level * np.sqrt(2.0 * T_k[k] / N)
        upper_bound = -ln_T[k] + offset_C + spread
        
        if ln_E[k] > upper_bound:
            k_signal.append(k)  # 상한선을 뚫고 올라온 결정론적 충격/개입 모드
        else:
            k_noise.append(k)   # 백색잡음 가설 범위 내의 확률적 잡음 모드
            
    # 4. 성분 동적 재합성
    # I_t: 상한선 초과 유의미 모드들의 합 + 최종 기저 레벨 이동(Residue)
    I_t = np.sum([oscillatory_imfs[k] for k in k_signal], axis=0) if len(k_signal) > 0 else np.zeros(N)
    I_t = I_t + residue  # Mean Shift 계단형 단절 성분 흡수
    
    # R_pure: 백색잡음 기준선 이하의 고주파 모드들의 합 (순수 백색잡음)
    R_pure = np.sum([oscillatory_imfs[k] for k in k_noise], axis=0)
    
    return I_t, R_pure

# =====================================================================
# STL-EMD 기반 3차원 시계열 강도 산출기 (고도화 확정본)
# =====================================================================
def calculate_time_series_strength(series: np.ndarray):
    """
    1. 고전 STL 분해: y = T + S + R_stl -> F_T_STL, F_S_STL, F_R_STL 산출
    2. STL-EMD 동적 분해: y = T + S + I + R_pure -> F_T_STL_EMD, F_S_STL_EMD, F_I_STL_EMD 산출
    """
    estimated_p = estimate_period_fft(series)
    if len(series) < 2 * estimated_p:
        estimated_p = max(2, len(series) // 2 - 1)
        
    try:
        # 고전적 STL 분해 수행
        res = STL(series, period=estimated_p, robust=True).fit()
        T, S, R_stl = res.trend, res.seasonal, res.resid
        
        var_R_stl = np.var(R_stl)
        var_TR_stl = np.var(T + R_stl)
        var_SR_stl = np.var(S + R_stl)
        
        F_T_STL = max(0.0, 1.0 - (var_R_stl / var_TR_stl if var_TR_stl > 0 else 1.0))
        F_S_STL = max(0.0, 1.0 - (var_R_stl / var_SR_stl if var_SR_stl > 0 else 1.0))
        F_R_STL = max(0.0, 1.0 - (1.0 / var_R_stl if var_R_stl > 0 else 1.0))

        # EMD 기반 동적 잔차 도출
        I, R_pure = decompose_residual_emd(R_stl, alpha_level=1.645)
        
        # Hyndman 분산 분할 공리 기반 3차원 정밀 강도 산출
        var_R_pure = np.var(R_pure)

        # # 260930: R_pure 대신 R_stl (I + R_pure) 적용
        # var_TR_pure = np.var(T + I + R_pure)  # 분모에 I_t를 포함하여 분산 보존 (T + R_stl과 동일)
        # var_SR_pure = np.var(S + I + R_pure)  # 분모에 I_t를 포함하여 분산 보존 (S + R_stl과 동일)

        #261001: R_pure 적용
        var_TR_pure = np.var(T + R_pure)
        var_SR_pure = np.var(S + R_pure)  
        var_IR_pure = np.var(I + R_pure)     
        
        F_T_STL_EMD = max(0.0, 1.0 - (var_R_pure / var_TR_pure if var_TR_pure > 0 else 1.0))
        F_S_STL_EMD = max(0.0, 1.0 - (var_R_pure / var_SR_pure if var_SR_pure > 0 else 1.0))
        F_I_STL_EMD = max(0.0, 1.0 - (var_R_pure / var_IR_pure if var_IR_pure > 0 else 1.0))
            
    except Exception:
        F_T_STL, F_S_STL, F_R_STL = 0.0, 0.0, 0.0
        F_T_STL_EMD, F_S_STL_EMD, F_I_STL_EMD = 0.0, 0.0, 0.0
        
    return F_T_STL, F_S_STL, F_R_STL, F_T_STL_EMD, F_S_STL_EMD, F_I_STL_EMD


# =============================================================================
# 시계열 강도 임곗값 알고리즘
# =============================================================================

def get_otsu_threshold(data: np.ndarray, num_bins: int = 101) -> float:
    """
    1) Otsu 알고리즘: 집단 간 분산(Between-Class Variance)을 최대화하는 최적 경계점 탐색
    
    [수리적 원리]
    지표 데이터를 임곗값 T를 기준으로 두 집단(C0: 미달, C1: 초과)으로 분할했을 때,
    전체 분산 중 집단 간 분산 sigma_B^2(T) = w0(T) * w1(T) * [mu0(T) - mu1(T)]^2 가
    최대가 되는 지점을 전역 탐색(Grid Search)하여 결정 경계로 확정합니다.
    """
    # 결측치(NaN) 제거
    data = data[~np.isnan(data)]
    if len(data) < 2 or np.ptp(data) == 0:
        return float(np.mean(data))

    # [0.0, 1.0] 정규화 범위 내에서 히스토그램 빈(Bin) 생성
    hist, bin_edges = np.histogram(data, bins=num_bins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = data.size

    current_max = 0.0
    threshold = bin_centers[0]
    w_bg = 0.0          # 배경(C0, Low Component) 누적 가중치
    sum_bg = 0.0        # 배경 누적 모멘트 합
    sum_total = np.dot(hist, bin_centers)  # 전체 데이터의 기대값 총합

    # 각 구간을 경계 후보로 순회하며 집단 간 분산 계산
    for i in range(num_bins):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg  # 전경(C1, High Component) 가중치
        if w_fg == 0:
            break

        sum_bg += hist[i] * bin_centers[i]
        m_bg = sum_bg / w_bg                    # C0 집단 평균 (mu0)
        m_fg = (sum_total - sum_bg) / w_fg      # C1 집단 평균 (mu1)

        # 집단 간 분산(Between-Class Variance) 연산
        var_between = w_bg * w_fg * ((m_bg - m_fg) ** 2)

        # 최대 분산 갱신 시 최적 임곗값 업데이트
        if var_between > current_max:
            current_max = var_between
            threshold = bin_centers[i]

    return float(threshold)


def get_gmm_threshold(data: np.ndarray) -> float:
    """
    2) 2-컴포넌트 Gaussian Mixture Model: 베이지안 사후확률이 같아지는 결정 경계 도출
    
    [수리적 원리]
    지표 분포를 Low(C1)와 High(C2) 2개의 가우시안 확률밀도의 혼합(Mixture)으로 모델링합니다:
    p(x) = w1 * N(x | m1, s1^2) + w2 * N(x | m2, s2^2)
    두 컴포넌트의 사후확률이 50:50으로 교차하는 지점(w1*N1 = w2*N2)을 양변에 로그를 취해
    정리한 2차 방정식 A*x^2 + B*x + C = 0 의 실근 중 두 평균 사이(m1 <= x <= m2)의 해를 구합니다.
    """
    data = data[~np.isnan(data)]
    # 표본 수가 극소량이거나 분산이 거의 없는 단조 데이터일 경우 단순 평균 반환
    if len(data) < 10 or np.ptp(data) < 0.03:
        return float(np.mean(data))

    # 1D 데이터를 GMM 입력 규격(-1, 1)으로 변환 후 피팅
    X = data.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)

    # 파라미터 추출: 평균(means), 분산(covariances), 혼합 가중치(weights)
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    # 평균값을 기준으로 컴포넌트 정렬 (idx[0]: Low, idx[1]: High)
    idx = np.argsort(means)
    m1, m2 = float(means[idx[0]]), float(means[idx[1]])
    s1, s2 = max(float(np.sqrt(variances[idx[0]])), 1e-6), max(float(np.sqrt(variances[idx[1]])), 1e-6)
    w1, w2 = float(weights[idx[0]]), float(weights[idx[1]])

    # 2차 방정식 계수 유도: A*x^2 + B*x + C = 0
    # log(w1 / (s1 * sqrt(2pi))) - (x - m1)^2 / (2*s1^2) = log(w2 / (s2 * sqrt(2pi))) - (x - m2)^2 / (2*s2^2)
    A = 1.0 / (2.0 * s1**2) - 1.0 / (2.0 * s2**2)
    B = m2 / (s2**2) - m1 / (s1**2)
    C = (
        m1**2 / (2.0 * s1**2)
        - m2**2 / (2.0 * s2**2)
        - np.log((w1 * s2) / (w2 * s1 + 1e-9) + 1e-9)
    )

    # 두 가우시안의 분산이 일치하여 A가 0에 수렴하는 경우 (1차 방정식 B*x + C = 0 으로 퇴화)
    if abs(A) < 1e-6:
        if abs(B) > 1e-6:
            return float(np.clip(-C / B, 0.0, 1.0))
        return float((m1 + m2) / 2.0)

    # 판별식(Discriminant) 계산
    det = B**2 - 4.0 * A * C
    if det >= 0:
        roots = [
            (-B + np.sqrt(det)) / (2.0 * A),
            (-B - np.sqrt(det)) / (2.0 * A)
        ]
        # 두 가우시안 중심(m1, m2) 사이에 위치하는 실근을 유효 경계로 선정
        valid = [r for r in roots if m1 <= r <= m2]
        if valid:
            return float(valid[0])

        # 구간 내 실근이 없을 경우 평균과 가장 인접한 실근을 클리핑하여 반환
        closest_root = min(roots, key=lambda r: min(abs(r - m1), abs(r - m2)))
        return float(np.clip(closest_root, 0.0, 1.0))

    # 허근 발생 시 표준편차 역수를 가중치로 둔 보수적 중간값 반환
    return float(np.average([m1, m2], weights=[s2, s1]))


def get_kde_valley_threshold(data: np.ndarray) -> float:
    """
    3) KDE Valley 알고리즘: 확률밀도함수의 두 봉우리 사이 국소 극소점(골짜기) 탐색
    
    [수리적 원리]
    Silverman 대역폭을 적용한 가우시안 커널 밀도 추정(KDE)으로 연속 밀도 함수 f(x)를 형성합니다.
    - 이봉형(Bimodal): f'(x) = 0 이고 f''(x) > 0 인 두 주요 극대점 사이의 국소 극소점(Valley)을 탐색합니다.
    - 단봉형(Unimodal): 꼬리로 진입하며 곡률이 변하는 변곡점 f''(x) = 0 지점을 포착합니다.
    """
    data = data[~np.isnan(data)]
    if len(data) < 5 or np.ptp(data) == 0:
        return float(np.mean(data))

    kde = gaussian_kde(data, bw_method="silverman")
    x_grid = np.linspace(0.0, 1.0, 201)  # 200개 격자 탐색
    density = kde(x_grid)

    # 국소 극대점(피크) 탐색
    peaks, _ = find_peaks(density, distance=20)

    # 1) 다봉형/이봉형인 경우: 가장 높은 두 봉우리 사이의 골짜기 최저점 탐색
    if len(peaks) >= 2:
        top2 = sorted(peaks, key=lambda idx: density[idx], reverse=True)[:2]
        p_left, p_right = sorted(top2)
        valley_idx = p_left + np.argmin(density[p_left:p_right])
        return float(x_grid[valley_idx])

    # 2) 단봉형인 경우: 2차 도함수의 부호가 바뀌는 변곡점(Inflection Point) 탐색
    peak_idx = np.argmax(density)
    d2 = np.gradient(np.gradient(density))
    inflection_pts = np.where(np.diff(np.sign(d2)))[0]

    # 주 봉우리 우측 내리막 꼬리로 진입하는 첫 번째 변곡점 선택
    candidates = [i for i in inflection_pts if i > peak_idx]
    if candidates:
        return float(x_grid[candidates[0]])

    # 변곡점 검출 실패 시 중앙값(Median) 반환
    return float(np.median(data))


def get_kneedle_threshold(data: np.ndarray) -> float:
    """
    4) Kneedle (Elbow / Knee) 알고리즘: 정렬 누적 곡선 대비 최대 수직 편차점 탐색
    
    [수리적 원리]
    지표값들을 오름차순 정렬한 누적 곡선 y(t)와 원점-종점을 잇는 대각 기준선 y = t 사이의
    수직 편차 거리 D(t) = |y(t) - t| 가 최대가 되는 무릎점(Knee/Elbow Point)을 찾아
    지표의 급격한 상승이 시작되는 전이 경계를 검출합니다.
    """
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return float(data[0]) if len(data) == 1 else 0.5

    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    t_seq = np.linspace(0.0, 1.0, n)  # 정규화된 샘플 순위 (0 ~ 1)
    denom = np.ptp(sorted_vals)
    if denom == 0:
        return float(sorted_vals[0])

    # 지표값 [0, 1] 정규화 곡선 y(t)
    y_norm = (sorted_vals - sorted_vals[0]) / (denom + 1e-9)

    # 대각 기준선 대비 절대 편차 곡선 계산
    diff = np.abs(y_norm - t_seq)
    knee_idx = np.argmax(diff)

    return float(sorted_vals[knee_idx])