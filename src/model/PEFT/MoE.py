import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.config import *
from src.model.PEFT.LoRA import (
    enable_lora,
    get_lora_state,
    reset_lora_parameters,
    save_lora,
    set_lora_state,
)
from src.model.PEFT.train_forecast import (
    create_sliding_windows,
    forecast,
    train_peft_adapter,
)
from src.util.data_analysis import (
    calculate_strength_stats,
    calculate_time_series_strength,
    get_gmm_threshold,
    get_kde_valley_threshold,
    get_kneedle_threshold,
    get_median_threshold,
    get_otsu_threshold,
    get_static_threshold,
)


def _normalize_decomp_method(decomp_method: str) -> str:
    method = decomp_method.strip().upper().replace('-', '_')
    if method not in ('STL', 'STL_EMD'):
        raise ValueError(f"지원하지 않는 시계열 분해 방식입니다: {decomp_method} ('STL' 또는 'STL_EMD'만 지원)")
    return method


def get_sector_keys(decomp_method: str) -> List[str]:
    """
    시계열 분해 방식(STL 또는 STL_EMD)에 따라 정의되는 섹터 키 목록을 반환합니다.
      - T (Trend): HT (F_T >= theta_T) / LT (F_T < theta_T)
      - S (Seasonality): HS (F_S >= theta_S) / LS (F_S < theta_S)
      - I (Intervention): HI (F_I >= theta_I) / LI (F_I < theta_I)
      - 'STL' (2D: F_T, F_S -> 4개 섹터):
          ['LT_LS', 'HT_LS', 'LT_HS', 'HT_HS']
      - 'STL_EMD' (3D: F_T, F_S, F_I -> 8개 섹터):
          ['LT_LS_LI', 'HT_LS_LI', 'LT_HS_LI', 'LT_LS_HI', 'HT_HS_LI', 'HT_LS_HI', 'LT_HS_HI', 'HT_HS_HI']
    """
    method = _normalize_decomp_method(decomp_method)
    if method == 'STL':
        return ['LT_LS', 'HT_LS', 'LT_HS', 'HT_HS']
    else:
        return [
            'LT_LS_LI',
            'HT_LS_LI',
            'LT_HS_LI',
            'LT_LS_HI',
            'HT_HS_LI',
            'HT_LS_HI',
            'LT_HS_HI',
            'HT_HS_HI',
        ]


def get_thresholds(
    decomp_method: str,
    strength_matrix: Optional[np.ndarray] = None,
    threshold_method: str = "median",
) -> Tuple[float, ...]:
    """
    분해 방식('STL' 또는 'STL_EMD')과 임계치 산출 방식(기본값: 'median' = Training set Q2)에 따라
    결정 경계 임계치 (theta_T, theta_S) 또는 (theta_T, theta_S, theta_I)를 반환합니다.
    """
    method = _normalize_decomp_method(decomp_method)
    thr_mode = str(threshold_method).strip().lower()

    if strength_matrix is not None and len(strength_matrix) > 0 and thr_mode != "static":
        fn_map = {
            "median": get_median_threshold,
            "q2": get_median_threshold,
            "otsu": get_otsu_threshold,
            "gmm": get_gmm_threshold,
            "kde_valley": get_kde_valley_threshold,
            "kneedle": get_kneedle_threshold,
        }
        thr_fn = fn_map.get(thr_mode, get_median_threshold)
        dim = 2 if method == 'STL' else 3
        computed = [round(float(thr_fn(strength_matrix[:, d])), 4) for d in range(dim)]
        return tuple(computed)

    theta_T, theta_S, theta_I = get_static_threshold()
    if method == 'STL':
        return (theta_T, theta_S)
    else:
        return (theta_T, theta_S, theta_I)


def extract_strength_vector(series: np.ndarray, decomp_method: str) -> np.ndarray:
    """
    입력 시계열 시퀀스(series)에서 calculate_time_series_strength를 호출하여
    분해 방식('STL' 또는 'STL_EMD')에 맞는 물리적 강도 벡터를 계산합니다.
      - 'STL': [F_T_STL, F_S_STL]
      - 'STL_EMD': [F_T_STL_EMD, F_S_STL_EMD, F_I_STL_EMD]
    """
    method = _normalize_decomp_method(decomp_method)
    F_T_stl, F_S_stl, _, F_T_emd, F_S_emd, F_I_emd = calculate_time_series_strength(series)
    if method == 'STL':
        return np.array([F_T_stl, F_S_stl], dtype=np.float32)
    else:
        return np.array([F_T_emd, F_S_emd, F_I_emd], dtype=np.float32)


def summarize_strength_matrix(strength_matrix: np.ndarray, decomp_method: str) -> Dict[str, Optional[Dict[str, float]]]:
    """
    시퀀스별 강도 행렬(Shape: [N, D])로부터 F_T, F_S, F_I 각각의
    통계량(min, q1, q2, q3, max, mean) 딕셔너리를 산출합니다.
    """
    method = _normalize_decomp_method(decomp_method)
    if strength_matrix is None or len(strength_matrix) == 0:
        empty_stats = {'min': 0.0, 'q1': 0.0, 'q2': 0.0, 'q3': 0.0, 'max': 0.0, 'mean': 0.0}
        return {
            'F_T': empty_stats,
            'F_S': empty_stats,
            'F_I': empty_stats if method == 'STL_EMD' else None,
        }

    stats_dict: Dict[str, Optional[Dict[str, float]]] = {
        'F_T': calculate_strength_stats(strength_matrix[:, 0]),
        'F_S': calculate_strength_stats(strength_matrix[:, 1]),
        'F_I': calculate_strength_stats(strength_matrix[:, 2]) if (method == 'STL_EMD' and strength_matrix.shape[1] >= 3) else None,
    }
    return stats_dict


def assign_deterministic_sector(strength_vec: np.ndarray, thresholds: Tuple[float, ...]) -> str:
    """
    물리적 강도 벡터(F_T, F_S[, F_I])와 임계치(theta_T, theta_S[, theta_I])를 비교하여
    결정론적 섹터 이름(예: 'HT_LS', 'HT_LS_HI' 등)을 반환합니다.
    """
    axis_names = ['T', 'S', 'I']
    tags = []
    for idx, (val, thr) in enumerate(zip(strength_vec, thresholds)):
        prefix = 'H' if val >= thr else 'L'
        tags.append(f"{prefix}{axis_names[idx]}")
    return "_".join(tags)


# =====================================================================
# 1단계: 전문가 구축 (Stage 1: Expert Construction)
# =====================================================================
def build_moe_expert_pool(
    model,
    adapters: Dict,
    tr_data: np.ndarray,
    cl: int,
    hl: int,
    decomp_method: str = "STL_EMD",
    epochs: int = 100,
    bs: int = 64,
    lr: float = 1e-4,
    stride: int = None,
    chk_dir: str = None,
    chk_prefix: str = None,
    threshold_method: str = "median",
) -> Tuple[Dict[str, Dict], Dict[str, int], Tuple[float, ...], Dict[str, Optional[Dict[str, float]]]]:
    """
    [1단계: 전문가 구축 (Expert Construction)]
      1. 학습 데이터 슬라이딩 윈도우(X_train, Y_train) 생성 (stride = PARAMS['PEFT']['common']['stride'])
      2. 입력 시계열 시퀀스마다 STL(4개 섹터) 또는 STL_EMD(8개 섹터) 분해를 통해 물리적 강도(F_T, F_S[, F_I]) 계산
      3. Training set 기준 F_T, F_S[, F_I]의 q2(중앙값, get_median_threshold)를 임계치(theta_T, theta_S[, theta_I])로 정의하여
         4개(STL) 또는 8개(STL_EMD) 상호 배타적 Sector로 데이터 분할
      4. 각 Sector 데이터만을 사용해 독립적인 LoRA 어댑터를 학습하여 Expert Pool 구축

    반환값:
      - expert_pool: 학습된 섹터별 LoRA 가중치 딕셔너리
      - sector_counts: 학습 데이터의 섹터별 시퀀스 개수
      - thresholds: Training set 기준으로 도출된 (theta_T, theta_S[, theta_I]) 임계치 튜플
      - train_strength_stats: Training set 시퀀스들의 F_T, F_S, F_I 통계량(min, q1, q2, q3, max, mean)
    """
    method = _normalize_decomp_method(decomp_method)
    if stride is None:
        stride = int(str(PARAMS['PEFT']['common']['stride']).split(',')[0].strip())
    X_train, Y_train = create_sliding_windows(tr_data, cl=cl, hl=hl, stride=stride)
    n_windows = len(X_train)
    dim = 2 if method == 'STL' else 3

    sector_keys = get_sector_keys(method)

    print(f"  [Stage 1: {method} Oracle] Calculating strengths for {n_windows} training sequences (threshold_method={threshold_method})...")
    train_strengths = np.zeros((n_windows, dim), dtype=np.float32)
    for i, seq in enumerate(X_train):
        train_strengths[i] = extract_strength_vector(seq, decomp_method=method)

    # Training set 시퀀스별 F_T, F_S, F_I 통계량(min, q1, q2, q3, max, mean) 산출
    train_strength_stats = summarize_strength_matrix(train_strengths, decomp_method=method)

    # Training set 기준 q2(중앙값)를 임계치(threshold)로 설정
    thresholds = get_thresholds(
        decomp_method=method,
        strength_matrix=train_strengths,
        threshold_method=threshold_method,
    )

    window_sectors = [
        assign_deterministic_sector(train_strengths[i], thresholds)
        for i in range(n_windows)
    ]
    window_sectors_arr = np.array(window_sectors)

    sector_counts: Dict[str, int] = {}
    for s_key in sector_keys:
        sector_counts[s_key] = int(np.sum(window_sectors_arr == s_key))

    print("\n" + "-" * 60)
    print(f"📈 [Stage 1: Training Strength Statistics ({method})]")
    for axis_key in ('F_T', 'F_S', 'F_I'):
        if train_strength_stats.get(axis_key) is not None:
            st = train_strength_stats[axis_key]
            print(
                f"  - {axis_key}: min={st['min']:.4f}, q1={st['q1']:.4f}, "
                f"q2(median)={st['q2']:.4f}, q3={st['q3']:.4f}, max={st['max']:.4f}, mean={st['mean']:.4f}"
            )
    print(
        f"📊 [Stage 1: Expert Construction] Sector Distribution "
        f"(decomp={method}, threshold_method={threshold_method}, sectors={len(sector_keys)}, theta={thresholds})"
    )
    for s_key in sector_keys:
        cnt = sector_counts[s_key]
        ratio = (cnt / n_windows) * 100 if n_windows > 0 else 0.0
        print(f"  - Sector [{s_key:<8}]: {cnt:>6} sequences ({ratio:>5.1f}%)")
    print("-" * 60)

    # 각 섹터별 독립 LoRA 어댑터 학습
    expert_pool: Dict[str, Dict] = {}
    enable_lora(adapters, enabled=True)
    for s_key in sector_keys:
        mask = window_sectors_arr == s_key
        cnt = int(np.sum(mask))

        if cnt == 0:
            print(f"  ⚠️ Sector [{s_key}] has 0 samples -> Skipping training for this sector.")
            continue

        X_sec = X_train[mask]
        Y_sec = Y_train[mask]
        eff_bs = min(bs, len(X_sec))

        print(f"\n🔧 Training Expert for Sector [{s_key}] (samples={cnt}, epochs={epochs}, bs={eff_bs})...")
        # 다른 섹터와 그래디언트 간섭이 없도록 LoRA 파라미터(A, B) 초기화 후 독립 학습
        reset_lora_parameters(adapters)
        train_peft_adapter(
            model=model,
            adapters=adapters,
            X=X_sec,
            Y=Y_sec,
            epochs=epochs,
            bs=eff_bs,
            lr=lr,
            seed=0,
            log_every=max(1, epochs // 3),
        )

        # 학습 완료된 해당 섹터 전문가 가중치 저장
        expert_pool[s_key] = get_lora_state(adapters)

        if chk_dir and chk_prefix:
            os.makedirs(chk_dir, exist_ok=True)
            sec_chk_path = os.path.join(chk_dir, f"{chk_prefix}_sector[{s_key}].pt")
            save_lora(adapters, sec_chk_path)

    return expert_pool, sector_counts, thresholds, train_strength_stats


# =====================================================================
# 2단계: 전문가 할당 (Stage 2: Expert Gating)
# =====================================================================
def gate_test_windows(
    ctx_arr: np.ndarray,
    decomp_method: str = "STL_EMD",
    thresholds: Optional[Tuple[float, ...]] = None,
) -> Tuple[np.ndarray, List[str], Tuple[float, ...]]:
    """
    [2단계: 전문가 할당 (Expert Gating)]
    추론 시 유입된 각 테스트 컨텍스트 시퀀스에 실시간으로 STL 또는 STL_EMD 분해를 적용하여
    물리적 강도 지표(F_T, F_S[, F_I])를 추출하고, 1단계(Training set)에서 정의된
    임계치(theta_T, theta_S[, theta_I])와 비교하여 결정론적 섹터(예: 'HT_LS', 'HT_LS_HI')를 판별합니다.

    반환값:
      - strengths_arr: Shape (N_test_windows, D) ('STL'이면 D=2, 'STL_EMD'이면 D=3)
      - assigned_sectors: 길이 N_test_windows의 결정론적 섹터 이름 리스트
      - thresholds: 적용된 임계치 튜플
    """
    method = _normalize_decomp_method(decomp_method)
    if thresholds is None:
        thresholds = get_thresholds(method, threshold_method="static")
    dim = 2 if method == 'STL' else 3

    strengths_arr = np.zeros((len(ctx_arr), dim), dtype=np.float32)
    assigned_sectors: List[str] = []

    for i, ctx in enumerate(ctx_arr):
        s_vec = extract_strength_vector(ctx, decomp_method=method)
        strengths_arr[i] = s_vec
        assigned_sectors.append(assign_deterministic_sector(s_vec, thresholds))

    return strengths_arr, assigned_sectors, thresholds


# =====================================================================
# 3단계: 전문가 활용 (Stage 3: Expert Utilization)
# =====================================================================
def compute_moe_sector_weights(
    strength_vec: np.ndarray,
    assigned_sector: str,
    trained_sectors: List[str],
    inference_method: str = "single",
) -> Dict[str, float]:
    """
    [3단계: 추론 방식(inference_method)에 따른 섹터 전문가 가중치 산출]
      - 'single' (단일 전문가 호출):
          입력 시퀀스가 속한 결정론적 섹터(assigned_sector)의 전문가가 학습되어 있으면 단독 호출 (가중치 1.0).
          만약 해당 섹터의 전문가가 없으면(Inactive), 모든 전문가 가중치를 0.0으로 두어
          원래의 순수 TSFM(TimesFM 백본)을 호출해 예측을 수행합니다.
      - 'weighted' (물리적 강도 비례 가중 앙상블):
          2단계에서 계산된 F_T, F_S(, F_I) 크기에 비례하여 각 섹터 전문가의 가중치를 산출해 가중 합산
    """
    if not trained_sectors:
        return {}

    mode = inference_method.strip().lower()
    eps = 1e-8
    f_clipped = np.clip(strength_vec, 1e-4, 1.0 - 1e-4)

    # 1) 입력 시퀀스가 속한 sector의 전문가만 단독 호출하여 추론하는 경우
    if mode == "single":
        if assigned_sector in trained_sectors:
            return {s: (1.0 if s == assigned_sector else 0.0) for s in trained_sectors}

        # 해당 섹터에 학습된 expert가 없는 경우(Inactive):
        # 대체 전문가를 쓰지 않고, 순수 TSFM 백본을 호출할 수 있도록 전문가 가중치를 모두 0.0으로 반환
        return {s: 0.0 for s in trained_sectors}

    # 2) F_T, F_S(, F_I) 값의 크기에 비례하여 sector별 전문가를 weighted ensemble하는 경우 ('weighted')
    raw_weights: Dict[str, float] = {}
    for s_key in trained_sectors:
        w = 1.0
        tags = s_key.split('_')
        for idx, tag in enumerate(tags):
            f_val = float(f_clipped[idx])
            w *= f_val if tag.startswith('H') else (1.0 - f_val)
        raw_weights[s_key] = w

    total_w = sum(raw_weights.values())
    if total_w <= eps:
        uniform_w = 1.0 / len(trained_sectors)
        return {s: uniform_w for s in trained_sectors}

    return {s: val / total_w for s, val in raw_weights.items()}


def forecast_moe_test_series(
    model,
    adapters: Dict,
    expert_pool: Dict[str, Dict],
    data: np.ndarray,
    cl: int,
    hl: int,
    decomp_method: str = "STL_EMD",
    inference_method: str = "single",
    thresholds: Optional[Tuple[float, ...]] = None,
    tsfm_name: str = TSFM_METHOD,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, Optional[Dict[str, float]]], pd.DataFrame]:
    """
    [2단계 + 3단계 통합 추론 함수]
      1. 테스트 시계열(data)에서 hl 간격으로 컨텍스트 시퀀스 추출
      2. [2단계: 전문가 할당] 각 시퀀스별 실시간 F_T, F_S(, F_I) 산출 및 결정론적 섹터 판별
      3. [3단계: 전문가 활용] inference_method('single' 또는 'weighted')에 따라
         할당된 섹터의 전문가를 호출(Stitching)하여 최종 예측치 산출
         (단, 'single' 모드에서 할당된 섹터에 전문가가 없으면 원래의 순수 TSFM을 호출하여 예측 수행)
      4. 각 테스트 컨텍스트마다 F_T, F_S, F_I 및 호출한 expert 정보를 DataFrame으로 기록하여 반환

    반환값:
      - predictions: 전체 테스트 구간 예측값 (1D np.ndarray)
      - actuals: 전체 테스트 구간 실제값 (1D np.ndarray)
      - test_sector_counts: 테스트 윈도우들의 섹터별 할당 개수
      - test_strength_stats: 테스트 윈도우들의 F_T, F_S, F_I 통계량(min, q1, q2, q3, max, mean)
      - test_context_df: 테스트 컨텍스트별 F_T, F_S, F_I, 할당 섹터, 호출된 expert 기록 DataFrame
    """
    method = _normalize_decomp_method(decomp_method)
    cl, hl = int(cl), int(hl)
    ctx_list = []
    rem_lens = []
    actuals = []
    ctx_ranges = []

    i = cl
    pred_offset = 0
    while i < len(data):
        rem_len = min(hl, len(data) - i)
        ctx_list.append(data[i - cl : i])
        actuals.append(data[i : i + rem_len])
        rem_lens.append(rem_len)
        ctx_ranges.append((i - cl, i, pred_offset, pred_offset + rem_len))
        i += hl
        pred_offset += rem_len

    ctx_arr = np.stack(ctx_list).astype(np.float32)  # Shape: (N_windows, cl)
    n_windows = len(ctx_arr)

    # 2단계: 실시간 특징 지표(F_T, F_S[, F_I]) 추출 및 섹터 판별 (Training set의 thresholds 적용)
    strengths_arr, assigned_sectors, applied_thresholds = gate_test_windows(
        ctx_arr,
        decomp_method=method,
        thresholds=thresholds,
    )
    test_strength_stats = summarize_strength_matrix(strengths_arr, decomp_method=method)

    sector_keys = get_sector_keys(method)
    test_sector_counts = {s_key: assigned_sectors.count(s_key) for s_key in sector_keys}
    print(f"  [Stage 2: Gating Summary ({method})] Applied Thresholds: {applied_thresholds}")
    print(f"  [Stage 2: Gating Summary ({method})] Test Windows Sector Counts: {test_sector_counts}")
    for axis_key in ('F_T', 'F_S', 'F_I'):
        if test_strength_stats.get(axis_key) is not None:
            st = test_strength_stats[axis_key]
            print(
                f"    - Test {axis_key}: min={st['min']:.4f}, q1={st['q1']:.4f}, "
                f"q2={st['q2']:.4f}, q3={st['q3']:.4f}, max={st['max']:.4f}, mean={st['mean']:.4f}"
            )

    trained_sectors = list(expert_pool.keys())
    mode = inference_method.strip().lower()

    # 3단계: 각 시퀀스별 전문가 호출 가중치 행렬 계산
    weight_matrix = np.zeros((n_windows, len(trained_sectors)), dtype=np.float32)
    called_experts: List[str] = []
    expert_weights_str_list: List[str] = []

    for win_idx in range(n_windows):
        w_dict = compute_moe_sector_weights(
            strength_vec=strengths_arr[win_idx],
            assigned_sector=assigned_sectors[win_idx],
            trained_sectors=trained_sectors,
            inference_method=inference_method,
        )
        for s_idx, s_key in enumerate(trained_sectors):
            weight_matrix[win_idx, s_idx] = w_dict.get(s_key, 0.0)

        total_win_w = float(np.sum(weight_matrix[win_idx])) if len(trained_sectors) > 0 else 0.0
        if total_win_w <= 1e-6:
            # 할당된 섹터에 학습된 expert가 없는 경우 -> 원래 있던 순수 TSFM 호출
            called_experts.append(tsfm_name)
            expert_weights_str_list.append(str({tsfm_name: 1.0}))
        elif mode == "single":
            called_experts.append(assigned_sectors[win_idx])
            expert_weights_str_list.append(str({assigned_sectors[win_idx]: 1.0}))
        else:
            rounded_w = {k: round(float(v), 4) for k, v in w_dict.items()}
            top_expert = max(w_dict.items(), key=lambda item: item[1])[0]
            called_experts.append(f"weighted(top={top_expert})")
            expert_weights_str_list.append(str(rounded_w))

    # 3단계: 온디맨드 플러그인(Plug-in) 결합 및 추론 수행
    ensemble_pred_windows = np.zeros((n_windows, hl), dtype=np.float32)

    # (A) 학습된 전문가가 할당된 시퀀스 추론
    for s_idx, s_key in enumerate(trained_sectors):
        sec_weights = weight_matrix[:, s_idx]  # Shape: (N_windows,)
        active_win_mask = sec_weights > 1e-6
        if not np.any(active_win_mask):
            continue

        # 해당 섹터 전문가의 LoRA 가중치(A, B)를 백본에 주입(Stitching) 및 활성화
        set_lora_state(adapters, expert_pool[s_key])

        # 해당 섹터가 호출된 전체 시퀀스들에 대해 추론 수행
        active_ctx = ctx_arr[active_win_mask]
        sec_preds = forecast(model, active_ctx, hl=hl)  # Shape: (N_active, hl)

        ensemble_pred_windows[active_win_mask] += sec_weights[active_win_mask, None] * sec_preds

    # (B) 할당된 영역에 대한 expert가 없는 시퀀스(또는 학습된 전문가가 전혀 없는 경우) -> 원래 TSFM 호출
    row_weight_sums = np.sum(weight_matrix, axis=1) if len(trained_sectors) > 0 else np.zeros(n_windows, dtype=np.float32)
    tsfm_fallback_mask = row_weight_sums <= 1e-6
    if np.any(tsfm_fallback_mask):
        n_fallback = int(np.sum(tsfm_fallback_mask))
        print(
            f"  ℹ️ [Stage 3: TSFM Fallback] {n_fallback} test window(s) assigned to inactive sector(s) "
            f"-> Calling base {tsfm_name} backbone for prediction."
        )
        enable_lora(adapters, enabled=False)
        fallback_ctx = ctx_arr[tsfm_fallback_mask]
        fallback_preds = forecast(model, fallback_ctx, hl=hl)
        ensemble_pred_windows[tsfm_fallback_mask] = fallback_preds
        enable_lora(adapters, enabled=True)

    # 테스트 컨텍스트별 F_T, F_S, F_I 및 호출한 expert 기록 DataFrame 생성
    context_records: List[Dict[str, Any]] = []
    for win_idx in range(n_windows):
        c_start, c_end, p_start, p_end = ctx_ranges[win_idx]
        f_t_val = round(float(strengths_arr[win_idx, 0]), 4)
        f_s_val = round(float(strengths_arr[win_idx, 1]), 4)
        f_i_val = round(float(strengths_arr[win_idx, 2]), 4) if (method == 'STL_EMD' and strengths_arr.shape[1] >= 3) else np.nan

        th_t = round(float(applied_thresholds[0]), 4)
        th_s = round(float(applied_thresholds[1]), 4)
        th_i = round(float(applied_thresholds[2]), 4) if (method == 'STL_EMD' and len(applied_thresholds) >= 3) else np.nan

        context_records.append(
            {
                'window_idx': win_idx,
                'context_start_idx': c_start,
                'context_end_idx': c_end,
                'pred_start_idx': p_start,
                'pred_end_idx': p_end,
                'decomp_method': method,
                'inference_method': inference_method,
                'F_T': f_t_val,
                'F_S': f_s_val,
                'F_I': f_i_val,
                'theta_T': th_t,
                'theta_S': th_s,
                'theta_I': th_i,
                'assigned_sector': assigned_sectors[win_idx],
                'called_expert': called_experts[win_idx],
                'expert_weights': expert_weights_str_list[win_idx],
            }
        )
    test_context_df = pd.DataFrame(context_records)

    predictions = [ensemble_pred_windows[idx, :rem_len] for idx, rem_len in enumerate(rem_lens)]
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(actuals, axis=0),
        test_sector_counts,
        test_strength_stats,
        test_context_df,
    )
