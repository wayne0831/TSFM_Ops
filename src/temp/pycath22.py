import numpy as np
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL

def extract_maximal_meta_features(x, seasonal_period=24):
    """
    입력 시계열 윈도우 x (1D numpy array)에서 
    박사 학위 분류 체계(축 1, 2, 3)를 완벽하게 대변하는
    16차원의 고도화된 하이브리드 라우터 입력 벡터 f를 추출합니다.
    
    Args:
        x: 1차원 시계열 데이터 (numpy array)
        seasonal_period: 시계열의 고유 주기 (기본값: 24)
    """
    meta_features = []
    L = len(x)
    mean_x = np.mean(x)
    std_x = np.std(x) + 1e-9
    var_x = np.var(x) + 1e-9
    
    # =========================================================================
    # [AXIS 1] 구조적/물리적 성분 차원 (Structural & Physical Component)
    # =========================================================================
    # 1-1. 선형 트렌드 기울기 (Linear Slope)
    t_axis = np.arange(L)
    slope, _ = np.polyfit(t_axis, x, 1)
    meta_features.append(slope)
    
    # STL 분해 기반의 트렌드 및 계절성 강도 연산
    try:
        stl_result = STL(x, period=seasonal_period, robust=True).fit()
        trend_comp = stl_result.trend
        seasonal_comp = stl_result.seasonal
        resid_comp = stl_result.resid
        
        var_r = np.var(resid_comp)
        var_tr = np.var(trend_comp + resid_comp)
        var_sr = np.var(seasonal_comp + resid_comp)
        
        # 1-2. 트렌드 강도 (Trend Strength)
        trend_strength = max(0.0, 1.0 - (var_r / var_tr)) if var_tr > 1e-9 else 0.0
        # 1-3. 계절성 강도 (Seasonality Strength)
        seasonality_strength = max(0.0, 1.0 - (var_r / var_sr)) if var_sr > 1e-9 else 0.0
    except Exception:
        trend_strength, seasonality_strength = 0.0, 0.0
        
    meta_features.extend([trend_strength, seasonality_strength])
    
    # 1-4. 단기 관성 지표 (Autocorrelation at Lag 1)
    autocorr_lag1 = np.sum((x[:-1] - mean_x) * (x[1:] - mean_x)) / (L * var_x)
    meta_features.append(autocorr_lag1)
    
    # 1-5. 주기적 정합성 지표 (Autocorrelation at Lag Period)
    if L > seasonal_period:
        autocorr_period = np.sum((x[:-seasonal_period] - mean_x) * (x[seasonal_period:] - mean_x)) / (L * var_x)
    else:
        autocorr_period = 0.0
    meta_features.append(autocorr_period)
    
    # =========================================================================
    # [AXIS 2] 확률론적 상태 변화 차원 (Statistical Stationarity & Drift)
    # =========================================================================
    # 2-1 & 2-2. ADF 단위근 검정 (통계량 및 P-value) -> 안정성 확인
    try:
        adf_stat, adf_p, _, _, _, _ = adfuller(x, maxlag=5, autolag=None)
    except Exception:
        adf_stat, adf_p = 0.0, 1.0 
    meta_features.extend([adf_stat, adf_p])
    
    # 2-3. KPSS 검정 통계량 (KPSS Statistic) -> 추세 정상성 교차 확인
    try:
        # 정적으로 가동하기 위해 유의 수준 회귀 모드인 'ct'(trend stationary) 적용
        kpss_stat, _, _, _ = kpss(x, regression='ct', nlags='auto')
    except Exception:
        kpss_stat = 0.0
    meta_features.append(kpss_stat)
    
    # 2-4. 시간 경과에 따른 데이터 평균 이동 (Mean Drift Ratio)
    half = L // 2
    mean_first = np.mean(x[:half])
    mean_second = np.mean(x[half:])
    mean_drift = (mean_second - mean_first) / std_x
    meta_features.append(mean_drift)
    
    # 2-5. 시간 경과에 따른 구조적 변동성 드리프트 (Variance Drift Ratio)
    var_first = np.var(x[:half]) + 1e-9
    var_second = np.var(x[half:]) + 1e-9
    var_drift_ratio = var_second / var_first
    meta_features.append(var_drift_ratio)
    
    # 2-6 & 2-7. 고차 적률 지표 (Kurtosis & Skewness) -> 아웃라이어 및 비대칭 충격
    kurt = kurtosis(x, fisher=True)
    skw = skew(x)
    meta_features.extend([kurt, skw])
    
    # =========================================================================
    # [AXIS 3] 정보이론적 복잡도 차원 (Information-Theoretic Complexity)
    # =========================================================================
    # 3-1. 신호 불확실성 엔트로피 (Shannon Entropy)
    hist, _ = np.histogram(x, bins=10, density=True)
    hist = hist[hist > 0]
    shannon_entropy = -np.sum(hist * np.log(hist + 1e-9))
    meta_features.append(shannon_entropy)
    
    # 3-2. 순서 동역학 복잡도 (Permutation Entropy - Order 3) -> 비선형 카오스 정량화
    # 시간축 패턴 조합의 무작위성을 반영하는 고급 비선형 지표
    try:
        sub_windows = np.array([x[i:i+3] for i in range(L-2)])
        perms = np.argsort(sub_windows, axis=1)
        perm_ids = perms[:, 0] * 100 + perms[:, 1] * 10 + perms[:, 2]
        _, counts = np.unique(perm_ids, return_counts=True)
        probs = counts / len(perm_ids)
        permutation_entropy = -np.sum(probs * np.log2(probs + 1e-9))
    except Exception:
        permutation_entropy = 0.0
    meta_features.append(permutation_entropy)
    
    # 3-3. 고주파 잡음 밀도 (Zero Crossing Rate)
    normalized_x = x - mean_x
    zero_crossings = np.where(np.diff(np.sign(normalized_x)))[0]
    zcr = len(zero_crossings) / (L - 1)
    meta_features.append(zcr)
    
    # 3-4. 미세 백색 진동의 세기 (Fluctuation Intensity)
    # 1차 차분 신호의 표준편차를 통해 노이즈의 거친 텍스처 정량화
    fluctuation_intensity = np.std(np.diff(x)) / std_x
    meta_features.append(fluctuation_intensity)

    # 16차원 최종 메타 특징 벡터 f 매핑 완성
    f = np.array(meta_features, dtype=np.float32)
    return f

# 복잡한 OOD 시나리오를 가정한 프로토타입 시뮬레이션
if __name__ == "__main__":
    L_standard = 520
    period = 24
    t = np.linspace(0, 100, L_standard)
    
    # 시나리오 설정: 
    # 확실한 결정론적 트렌드 + 주기성 성분이 기저에 있으나, 
    # 후반부에 변동성 폭발(Concept Drift)이 일어나며, 전체적으로 거친 미세 노이즈가 침투한 복잡한 리얼 월드형 시계열
    synthetic_series = 0.03 * t + np.sin(2 * np.pi * t / period)
    synthetic_series[:260] += np.random.normal(0, 0.1, 260)   # 전반부: 잔잔함
    synthetic_series[260:] += np.random.normal(0, 2.0, 260)   # 후반부: 이분산성 폭발 구간
    synthetic_series[400] = 30.0                              # 거대한 돌발 스파이크 주입
    
    f_vector = extract_maximal_meta_features(synthetic_series, seasonal_period=period)
    
    print("===================================================================")
    print(f"★ 박사학위용 16차원 고도화 신경-통계 라우터 입력 벡터 f 추출 성공")
    print("===================================================================")
    # A1 (Axis 1): 구조적/물리적 성분 차원
    # A2 (Axis 2): 확률론적 상태 변화 차원 (Statistical Stationarity & Drift)
    # A3 (Axis 3): 정보이론적 복잡도 차원 (Information-Theoretic Complexity)
    axis_labels = [
        "A1-Trend 기울기 (Linear Slope)", "A1-트렌드 강도 (Trend Strength)", 
        "A1-계절성 강도 (Seasonality Strength)", "A1-단기연속성 (Autocorr Lag-1)", 
        "A1-주기상관도 (Autocorr Lag-Period)", "A2-ADF 통계량 (ADF Statistic)", 
        "A2-ADF 유의성 (ADF P-value)", "A2-KPSS 통계량 (KPSS Statistic)", 
        "A2-평균이동추세 (Mean Drift Ratio)", "A2-분산변동추세 (Var Drift Ratio)", 
        "A2-아웃라이어 첨도 (Kurtosis)", "A2-분포비대칭 왜도 (Skewness)", 
        "A3-기저불확실성 (Shannon Entropy)", "A3-패턴카오스 (Permutation Entropy)", 
        "A3-잡음교차밀도 (Zero Crossing Rate)", "A3-미세진동강도 (Fluctuation Intensity)"
    ]
    
    for idx, (label, value) in enumerate(zip(axis_labels, f_vector)):
        print(f" Dim {idx+1:02d} | {label:<35} : {value:.4f}")