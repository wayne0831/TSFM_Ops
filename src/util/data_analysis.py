# calculate time series strength using STL and EMD methods

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from PyEMD import EMD
from typing import List, Dict, Any, Tuple
import warnings
import os

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
# 2. STL-EMD 기반 시계열 강도 산출기
# =====================================================================
def calculate_time_series_strength(series: np.ndarray):
    estimated_p = estimate_period_fft(series)
    if len(series) < 2 * estimated_p:
        estimated_p = max(2, len(series) // 2 - 1)
        
    try:
        res = STL(series, period=estimated_p, robust=True).fit()
        T, S, R_stl = res.trend, res.seasonal, res.resid
        
        var_R_stl = np.var(R_stl)
        var_TR_stl = np.var(T + R_stl)
        var_SR_stl = np.var(S + R_stl)
        
        F_T_STL = max(0.0, 1.0 - (var_R_stl / var_TR_stl if var_TR_stl > 0 else 1.0))
        F_S_STL = max(0.0, 1.0 - (var_R_stl / var_SR_stl if var_SR_stl > 0 else 1.0))
        F_R_STL = max(0.0, 1.0 - (1.0 / var_R_stl if var_R_stl > 0 else 1.0))

        emd = EMD()
        imfs = emd.emd(R_stl)
        
        if len(imfs) > 2:
            I = np.sum(imfs[1:], axis=0) 
        elif len(imfs) == 2:
            I = imfs[1]
        else:
            I = np.zeros_like(R_stl)
            
        R_pure = R_stl - I
        
        var_R_pure = np.var(R_pure)
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