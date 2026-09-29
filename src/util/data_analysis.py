# calculate time series strength using STL and EMD methods with Wu & Huang (2004) Dynamic Partitioning

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

# 프로젝트 내부 모듈 import (사용자 환경)
from src.config import *
from src.util.data_analysis import calculate_time_series_strength, estimate_period_fft

warnings.filterwarnings('ignore')

# =====================================================================
# 1. FFT 기반 동적 주기 추정기
# =====================================================================
def estimate_period_fft(series: np.ndarray, default_period: int = 24) -> int:
    n = len(series)
    t = np.arange(n)
    p = np.polyfit(t, series, 1)
    detrended = series - np.polyval(p, t)
    detrended = detrended - np.mean(detrended)
    
    fft_vals = np.fft.rfft(detrended)
    frequencies = np.fft.rfftfreq(n)
    magnitudes = np.abs(fft_vals)
    
    if len(magnitudes) > 1:
        dominant_idx = np.argmax(magnitudes[1:]) + 1 
        dominant_freq = frequencies[dominant_idx]
        if dominant_freq > 0:
            period = int(np.round(1.0 / dominant_freq))
            if 2 <= period <= n // 2:
                return period
    return default_period

# =====================================================================
# 2. Wu & Huang (2004) 백색잡음 유의성 검정 기반 동적 EMD 잔차 분해 모듈
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
# 3. STL-EMD 기반 3차원 시계열 강도 산출기 (고도화 확정본)
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
        var_TR_pure = np.var(T + I + R_pure)  # 분모에 I_t를 포함하여 분산 보존 (T + R_stl과 동일)
        var_SR_pure = np.var(S + I + R_pure)  # 분모에 I_t를 포함하여 분산 보존 (S + R_stl과 동일)
        var_IR_pure = np.var(I + R_pure)      # R_stl의 총분산과 동일
        
        F_T_STL_EMD = max(0.0, 1.0 - (var_R_pure / var_TR_pure if var_TR_pure > 0 else 1.0))
        F_S_STL_EMD = max(0.0, 1.0 - (var_R_pure / var_SR_pure if var_SR_pure > 0 else 1.0))
        F_I_STL_EMD = max(0.0, 1.0 - (var_R_pure / var_IR_pure if var_IR_pure > 0 else 1.0))
            
    except Exception:
        F_T_STL, F_S_STL, F_R_STL = 0.0, 0.0, 0.0
        F_T_STL_EMD, F_S_STL_EMD, F_I_STL_EMD = 0.0, 0.0, 0.0
        
    return F_T_STL, F_S_STL, F_R_STL, F_T_STL_EMD, F_S_STL_EMD, F_I_STL_EMD