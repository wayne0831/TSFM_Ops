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


def extract_all_strength_matrices(seq_arr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    입력 시퀀스 배열(Shape: [N, cl])에 대해 calculate_time_series_strength를 단 1회씩만 호출하여
    'STL'용 강도 행렬(Shape: [N, 2])과 'STL_EMD'용 강도 행렬(Shape: [N, 3])을 동시에 산출합니다.
    """
    n = len(seq_arr)
    stl_mat = np.zeros((n, 2), dtype=np.float32)
    emd_mat = np.zeros((n, 3), dtype=np.float32)

    for i, seq in enumerate(seq_arr):
        F_T_stl, F_S_stl, _, F_T_emd, F_S_emd, F_I_emd = calculate_time_series_strength(seq)
        stl_mat[i, 0] = F_T_stl
        stl_mat[i, 1] = F_S_stl
        emd_mat[i, 0] = F_T_emd
        emd_mat[i, 1] = F_S_emd
        emd_mat[i, 2] = F_I_emd

    return {"STL": stl_mat, "STL_EMD": emd_mat}


def build_test_context_windows(
    data: np.ndarray,
    cl: int,
    hl: int,
) -> Tuple[np.ndarray, List[np.ndarray], List[int], List[Tuple[int, int, int, int]]]:
    """
    1D 테스트 시계열(data, 앞부분 cl 길이의 컨텍스트 포함)에서 hl 간격으로
    컨텍스트 배열(ctx_arr), 실제값 리스트(actuals), 유효 길이 리스트(rem_lens), 인덱스 범위(ctx_ranges)를 생성합니다.
    """
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

    ctx_arr = np.stack(ctx_list).astype(np.float32)
    return ctx_arr, actuals, rem_lens, ctx_ranges


def get_or_train_shared_expert(
    model,
    adapters: Dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    epochs: int = 100,
    bs: int = 64,
    lr: float = 1e-4,
    shared_chk_path: Optional[str] = None,
    fallback_chk_path: Optional[str] = None,
) -> Optional[Dict]:
    """
    섹터 구분 없이 전체 training set(X_train, Y_train)으로 파인튜닝된 Shared Expert LoRA를 확보합니다.
    동일 조건의 체크포인트(shared_chk_path)가 이미 존재하면 재학습 없이 즉시 로드해 재사용합니다.
    """
    n_windows = len(X_train)
    if n_windows == 0:
        return None

    enable_lora(adapters, enabled=True)
    if shared_chk_path and os.path.isfile(shared_chk_path):
        try:
            loaded_state = torch.load(shared_chk_path, map_location="cpu")
            if isinstance(loaded_state, dict) and set(loaded_state.keys()) == set(adapters.keys()):
                print(f"\n♻️ [Stage 1: Shared Expert] Reusing existing full-train LoRA checkpoint: {shared_chk_path}")
                return {k: (v[0].clone(), v[1].clone()) for k, v in loaded_state.items()}
        except Exception as e:
            print(f"  ⚠️ Failed to load shared checkpoint ({shared_chk_path}): {e} -> Retraining Shared Expert.")

    eff_shared_bs = min(bs, n_windows)
    print(f"\n🌐 Training Shared Expert on full training set (samples={n_windows}, epochs={epochs}, bs={eff_shared_bs})...")
    reset_lora_parameters(adapters)
    train_peft_adapter(
        model=model,
        adapters=adapters,
        X=X_train,
        Y=Y_train,
        epochs=epochs,
        bs=eff_shared_bs,
        lr=lr,
        seed=0,
        log_every=max(1, epochs // 3),
    )
    shared_expert = get_lora_state(adapters)

    save_target = shared_chk_path or fallback_chk_path
    if save_target:
        os.makedirs(os.path.dirname(save_target), exist_ok=True)
        save_lora(adapters, save_target)
        print(f"✅ Shared Expert checkpoint saved to: {save_target}")

    return shared_expert


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
    train_shared_expert: bool = False,
    shared_chk_path: Optional[str] = None,
    precomputed_windows: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    precomputed_train_strengths: Optional[np.ndarray] = None,
    shared_expert_override: Optional[Dict] = None,
) -> Tuple[Dict[str, Dict], Dict[str, int], Tuple[float, ...], Dict[str, Optional[Dict[str, float]]], Optional[Dict]]:
    """
    [1단계: 전문가 구축 (Expert Construction)]
      1. 학습 데이터 슬라이딩 윈도우(X_train, Y_train) 생성 (precomputed_windows 전달 시 재사용)
      2. 입력 시계열 시퀀스마다 STL(4개 섹터) 또는 STL_EMD(8개 섹터) 물리적 강도(F_T, F_S[, F_I]) 계산
         (precomputed_train_strengths 전달 시 중복 분해 없이 즉시 재사용)
      3. Training set 기준 F_T, F_S[, F_I]의 q2(중앙값, get_median_threshold)를 임계치(theta_T, theta_S[, theta_I])로 정의하여
         4개(STL) 또는 8개(STL_EMD) 상호 배타적 Sector로 데이터 분할
      4. 각 Sector 데이터만을 사용해 독립적인 Domain Expert LoRA 어댑터를 학습하여 Expert Pool 구축
      5. train_shared_expert=True인 경우 Shared Expert LoRA 어댑터 확보 (shared_expert_override 또는 체크포인트 재사용)
    """
    method = _normalize_decomp_method(decomp_method)
    if stride is None:
        stride = int(str(PARAMS['PEFT']['common']['stride']).split(',')[0].strip())
    if precomputed_windows is not None:
        X_train, Y_train = precomputed_windows
    else:
        X_train, Y_train = create_sliding_windows(tr_data, cl=cl, hl=hl, stride=stride)
    n_windows = len(X_train)
    dim = 2 if method == 'STL' else 3

    sector_keys = get_sector_keys(method)

    if precomputed_train_strengths is not None:
        train_strengths = precomputed_train_strengths
        print(
            f"  [Stage 1: {method} Oracle] Reusing precomputed strengths for {n_windows} training sequences "
            f"(threshold_method={threshold_method})..."
        )
    else:
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

    # Shared Expert 확보 (외부에서 이미 준비된 경우 재사용, 아니면 체크포인트 로드 또는 1회 학습)
    shared_expert: Optional[Dict] = None
    if shared_expert_override is not None:
        shared_expert = shared_expert_override
    elif train_shared_expert and n_windows > 0:
        fallback_shared_path = os.path.join(chk_dir, f"{chk_prefix}_shared.pt") if (chk_dir and chk_prefix) else None
        shared_expert = get_or_train_shared_expert(
            model=model,
            adapters=adapters,
            X_train=X_train,
            Y_train=Y_train,
            epochs=epochs,
            bs=bs,
            lr=lr,
            shared_chk_path=shared_chk_path,
            fallback_chk_path=fallback_shared_path,
        )

    return expert_pool, sector_counts, thresholds, train_strength_stats, shared_expert


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
    has_shared_expert: bool = False,
) -> Tuple[Dict[str, float], float]:
    """
    [3단계: 추론 방식(inference_method)에 따른 Domain Expert 및 Shared Expert 가중치 산출]
      - 'single' (Domain Expert 단독 호출):
          입력 시퀀스가 속한 결정론적 섹터(assigned_sector)의 전문가가 학습되어 있으면 단독 호출 (가중치 1.0).
          만약 해당 섹터의 전문가가 없으면(Inactive), 모든 전문가 가중치를 0.0으로 두어
          원래의 순수 TSFM(TimesFM 백본)을 호출해 예측을 수행합니다.
      - 'ensemble' (Shared Expert + Domain Expert 0.5 : 0.5 결합):
          전체 training set으로 학습된 Shared Expert(0.5)와 속한 섹터의 Domain Expert(0.5) 출력을 1/2씩 결합합니다.
          만약 속한 섹터의 Domain Expert가 학습되지 않은 경우(Inactive), Shared Expert 단독(1.0)으로 예측합니다.

    반환값:
      - sector_weights: 각 학습된 섹터(Domain Expert)별 가중치 딕셔너리
      - shared_weight: Shared Expert에 부여되는 가중치 (0.0, 0.5 또는 1.0)
    """
    mode = inference_method.strip().lower()
    sector_weights = {s: 0.0 for s in trained_sectors}

    # 1) 입력 시퀀스가 속한 sector의 Domain Expert만 단독 호출 ('single')
    if mode == "single":
        if assigned_sector in trained_sectors:
            sector_weights[assigned_sector] = 1.0
        return sector_weights, 0.0

    # 2) Shared Expert(전체 학습셋) + Domain Expert(속한 섹터) 0.5:0.5 앙상블 ('ensemble')
    if mode == "ensemble":
        if assigned_sector in trained_sectors:
            if has_shared_expert:
                sector_weights[assigned_sector] = 0.5
                return sector_weights, 0.5
            else:
                sector_weights[assigned_sector] = 1.0
                return sector_weights, 0.0
        else:
            # 속한 섹터의 Domain Expert가 없는 경우(Inactive) -> Shared Expert 단독 호출
            if has_shared_expert:
                return sector_weights, 1.0
            return sector_weights, 0.0

    raise ValueError(f"지원하지 않는 inference_method입니다: {inference_method} ('single' 또는 'ensemble'만 지원)")


def forecast_moe_test_series_multi(
    model,
    adapters: Dict,
    expert_pool: Dict[str, Dict],
    data: np.ndarray,
    cl: int,
    hl: int,
    decomp_method: str = "STL_EMD",
    inference_methods: Optional[List[str]] = None,
    thresholds: Optional[Tuple[float, ...]] = None,
    tsfm_name: str = TSFM_METHOD,
    shared_expert: Optional[Dict] = None,
    precomputed_test_windows: Optional[
        Tuple[np.ndarray, List[np.ndarray], List[int], List[Tuple[int, int, int, int]]]
    ] = None,
    precomputed_test_strengths: Optional[np.ndarray] = None,
    precomputed_shared_preds: Optional[np.ndarray] = None,
) -> Tuple[
    Dict[str, np.ndarray],
    np.ndarray,
    Dict[str, int],
    Dict[str, Optional[Dict[str, float]]],
    Dict[str, pd.DataFrame],
]:
    """
    [2단계 + 3단계 One-Pass 통합 추론 함수]
    'single'과 'ensemble' 추론을 한 번의 패스로 동시 수행하여 중복 연산(STL/STL-EMD 분해 및 Domain/Shared Expert 순전파)을 제거합니다.
      1. 테스트 시계열(data)에서 hl 간격으로 컨텍스트 시퀀스 추출 (precomputed_test_windows 전달 시 재사용)
      2. [2단계: 전문가 할당] 각 시퀀스별 실시간 F_T, F_S(, F_I) 산출 및 결정론적 섹터 판별 (precomputed_test_strengths 전달 시 재사용)
      3. [3단계: 전문가 순전파]
         - 각 학습된 섹터 Domain Expert 순전파 (섹터당 1회) -> domain_pred_windows
         - 'ensemble' 요청 시 Shared Expert 순전파 (precomputed_shared_preds 전달 시 재사용, 아니면 1회) -> shared_pred_windows
         - 미학습(Inactive) 섹터가 있고 백본 폴백이 필요한 경우 순수 TSFM 순전파 (1회) -> tsfm_fallback_windows
      4. 요청된 각 inference_method별로 예측치 및 test_context_df 조립:
         - 'single': domain_pred (미학습 섹터는 순수 TSFM 백본)
         - 'ensemble': 0.5 * domain_pred + 0.5 * shared_pred (미학습 섹터는 shared_pred 단독)
    """
    if inference_methods is None:
        inference_methods = ["single", "ensemble"]
    modes = [m.strip().lower() for m in inference_methods if m.strip()]
    for m in modes:
        if m not in ("single", "ensemble"):
            raise ValueError(f"지원하지 않는 inference_method입니다: {m} ('single' 또는 'ensemble'만 지원)")

    method = _normalize_decomp_method(decomp_method)
    cl, hl = int(cl), int(hl)

    if precomputed_test_windows is not None:
        ctx_arr, actuals, rem_lens, ctx_ranges = precomputed_test_windows
    else:
        ctx_arr, actuals, rem_lens, ctx_ranges = build_test_context_windows(data, cl=cl, hl=hl)
    n_windows = len(ctx_arr)

    # 2단계: 실시간 특징 지표(F_T, F_S[, F_I]) 추출 및 섹터 판별 (사전 계산된 행렬 전달 시 즉시 재사용)
    if precomputed_test_strengths is not None:
        strengths_arr = precomputed_test_strengths
        applied_thresholds = thresholds if thresholds is not None else get_thresholds(method, threshold_method="static")
        assigned_sectors = [
            assign_deterministic_sector(strengths_arr[idx], applied_thresholds)
            for idx in range(n_windows)
        ]
    else:
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
    assigned_arr = np.array(assigned_sectors)
    has_shared = (shared_expert is not None) or (precomputed_shared_preds is not None)

    # 3단계 (A): 각 섹터별 Domain Expert 예측 (섹터당 1회만 수행하여 single / ensemble 모두 재사용)
    domain_pred_windows = np.zeros((n_windows, hl), dtype=np.float32)
    has_domain_pred = np.zeros(n_windows, dtype=bool)
    for s_key in trained_sectors:
        sec_mask = assigned_arr == s_key
        if not np.any(sec_mask):
            continue
        set_lora_state(adapters, expert_pool[s_key])
        domain_pred_windows[sec_mask] = forecast(model, ctx_arr[sec_mask], hl=hl)
        has_domain_pred[sec_mask] = True

    # 3단계 (B): Shared Expert 예측 ('ensemble' 요청 시 precomputed_shared_preds 재사용 또는 1회 수행)
    shared_pred_windows: Optional[np.ndarray] = None
    if "ensemble" in modes and has_shared:
        if precomputed_shared_preds is not None:
            shared_pred_windows = precomputed_shared_preds
        else:
            set_lora_state(adapters, shared_expert)
            shared_pred_windows = forecast(model, ctx_arr, hl=hl)

    # 3단계 (C): 순수 TSFM 백본 폴백 예측 (미학습 섹터가 있고 폴백이 필요한 경우에만 1회 수행)
    inactive_mask = ~has_domain_pred
    need_tsfm_fallback = np.any(inactive_mask) and (
        "single" in modes or ("ensemble" in modes and not has_shared)
    )
    tsfm_fallback_windows = np.zeros((n_windows, hl), dtype=np.float32)
    if need_tsfm_fallback:
        n_fallback = int(np.sum(inactive_mask))
        print(
            f"  ℹ️ [Stage 3: TSFM Fallback] {n_fallback} test window(s) assigned to inactive sector(s) "
            f"-> Calling base {tsfm_name} backbone for prediction."
        )
        enable_lora(adapters, enabled=False)
        tsfm_fallback_windows[inactive_mask] = forecast(model, ctx_arr[inactive_mask], hl=hl)
        enable_lora(adapters, enabled=True)

    # 4단계: 요청된 각 inference_method('single', 'ensemble')별로 최종 예측치 및 컨텍스트 로그 조립
    preds_by_method: Dict[str, np.ndarray] = {}
    context_dfs_by_method: Dict[str, pd.DataFrame] = {}

    th_t = round(float(applied_thresholds[0]), 4)
    th_s = round(float(applied_thresholds[1]), 4)
    th_i = round(float(applied_thresholds[2]), 4) if (method == 'STL_EMD' and len(applied_thresholds) >= 3) else np.nan

    for inf_m in modes:
        if inf_m == "single":
            pred_windows = np.where(has_domain_pred[:, None], domain_pred_windows, tsfm_fallback_windows)
        else:
            # 'ensemble': Shared Expert(0.5) + Domain Expert(0.5), 미학습 섹터는 Shared Expert(1.0)
            if shared_pred_windows is not None:
                pred_windows = np.where(
                    has_domain_pred[:, None],
                    0.5 * domain_pred_windows + 0.5 * shared_pred_windows,
                    shared_pred_windows,
                )
            else:
                pred_windows = np.where(has_domain_pred[:, None], domain_pred_windows, tsfm_fallback_windows)

        predictions = [pred_windows[idx, :rem_len] for idx, rem_len in enumerate(rem_lens)]
        preds_by_method[inf_m] = np.concatenate(predictions, axis=0)

        context_records: List[Dict[str, Any]] = []
        for win_idx in range(n_windows):
            c_start, c_end, p_start, p_end = ctx_ranges[win_idx]
            f_t_val = round(float(strengths_arr[win_idx, 0]), 4)
            f_s_val = round(float(strengths_arr[win_idx, 1]), 4)
            f_i_val = round(float(strengths_arr[win_idx, 2]), 4) if (method == 'STL_EMD' and strengths_arr.shape[1] >= 3) else np.nan
            sec_name = assigned_sectors[win_idx]
            has_dom = bool(has_domain_pred[win_idx])

            if inf_m == "single":
                if has_dom:
                    called_exp = sec_name
                    exp_w_str = str({sec_name: 1.0})
                else:
                    called_exp = tsfm_name
                    exp_w_str = str({tsfm_name: 1.0})
            else:
                if has_dom and has_shared:
                    called_exp = f"ensemble(SHARED+{sec_name})"
                    exp_w_str = str({'SHARED': 0.5, sec_name: 0.5})
                elif has_shared:
                    called_exp = "ensemble(SHARED_only)"
                    exp_w_str = str({'SHARED': 1.0})
                elif has_dom:
                    called_exp = sec_name
                    exp_w_str = str({sec_name: 1.0})
                else:
                    called_exp = tsfm_name
                    exp_w_str = str({tsfm_name: 1.0})

            context_records.append(
                {
                    'window_idx': win_idx,
                    'context_start_idx': c_start,
                    'context_end_idx': c_end,
                    'pred_start_idx': p_start,
                    'pred_end_idx': p_end,
                    'decomp_method': method,
                    'inference_method': inf_m,
                    'F_T': f_t_val,
                    'F_S': f_s_val,
                    'F_I': f_i_val,
                    'theta_T': th_t,
                    'theta_S': th_s,
                    'theta_I': th_i,
                    'assigned_sector': sec_name,
                    'called_expert': called_exp,
                    'expert_weights': exp_w_str,
                }
            )
        context_dfs_by_method[inf_m] = pd.DataFrame(context_records)

    actuals_concat = np.concatenate(actuals, axis=0)
    return (
        preds_by_method,
        actuals_concat,
        test_sector_counts,
        test_strength_stats,
        context_dfs_by_method,
    )


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
    shared_expert: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, Optional[Dict[str, float]]], pd.DataFrame]:
    """
    단일 inference_method('single' 또는 'ensemble')에 대한 편의 래퍼 함수입니다.
    """
    inf_m = inference_method.strip().lower()
    preds_map, actuals_concat, test_sector_counts, test_strength_stats, ctx_dfs_map = forecast_moe_test_series_multi(
        model=model,
        adapters=adapters,
        expert_pool=expert_pool,
        data=data,
        cl=cl,
        hl=hl,
        decomp_method=decomp_method,
        inference_methods=[inf_m],
        thresholds=thresholds,
        tsfm_name=tsfm_name,
        shared_expert=shared_expert,
    )
    return (
        preds_map[inf_m],
        actuals_concat,
        test_sector_counts,
        test_strength_stats,
        ctx_dfs_map[inf_m],
    )
