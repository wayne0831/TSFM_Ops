###########################################################################################################
# import libraries
###########################################################################################################
import ast
import gc
import os
import sys
import time
from itertools import product
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# 프로젝트 루트(TSFM_Ops) 및 src 경로를 sys.path에 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from chronos import Chronos2Pipeline
except ImportError as e:
    raise ImportError(
        "Chronos-2 실험을 실행하려면 'chronos-forecasting>=2.0.0' 패키지가 설치되어 있어야 합니다. "
        "`pip install 'chronos-forecasting>=2.0.0'`로 설치 후 다시 실행해 주세요."
    ) from e

from src.config import *
from src.model.PEFT.LoRA import (
    apply_lora_to_tsfm,
    get_trainable_params_ratio,
    remove_lora_from_tsfm,
    save_lora,
    set_lora_state,
)
from src.model.PEFT.MoE import (
    build_moe_expert_pool,
    build_test_context_windows,
    extract_all_strength_matrices,
    forecast_moe_test_series_multi,
    get_or_train_shared_expert,
    get_sector_keys,
)
from src.model.PEFT.train_forecast import (
    calculate_metrics,
    create_sliding_windows,
    forecast,
    forecast_test_series,
    save_peft_adapter,
    train_peft_adapter,
)

###########################################################################################################
# set user-defined functions & logger
###########################################################################################################


def _resolve_path(rel_path):
    """실행 위치(프로젝트 루트 또는 src 폴더)에 관계없이 경로를 안전하게 해석합니다."""
    if os.path.isabs(rel_path) or os.path.exists(rel_path):
        return rel_path
    root_candidate = os.path.normpath(os.path.join(ROOT_DIR, rel_path))
    if os.path.exists(root_candidate):
        return root_candidate
    src_candidate = os.path.normpath(os.path.join(SRC_DIR, rel_path))
    if os.path.exists(src_candidate):
        return src_candidate
    return root_candidate


class TeeLogger:
    """
    터미널(stdout/stderr) 출력과 동시에 지정된 로그(.txt) 파일들에 실시간으로 기록하는 로거 클래스입니다.
    """

    def __init__(self, *log_file_paths: str):
        self.terminal = sys.__stdout__
        self.log_files = []
        for path in log_file_paths:
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.log_files.append(open(path, mode="a", encoding="utf-8"))

    def write(self, message: str):
        if self.terminal is not None:
            self.terminal.write(message)
            self.terminal.flush()
        for f in self.log_files:
            f.write(message)
            f.flush()

    def flush(self):
        if self.terminal is not None:
            self.terminal.flush()
        for f in self.log_files:
            f.flush()

    def close(self):
        for f in self.log_files:
            try:
                f.close()
            except Exception:
                pass
        self.log_files = []


def _is_experiment_completed(
    res_csv_path: str,
    pred_npy_path: Optional[str],
    match_criteria: Dict[str, Any],
) -> bool:
    """
    이미 완료되어 성능 CSV 행과 예측 .npy 파일이 모두 저장된 실험인지 확인합니다.
    """
    if pred_npy_path and not os.path.isfile(pred_npy_path):
        return False
    if not os.path.isfile(res_csv_path):
        return False

    try:
        df = pd.read_csv(res_csv_path)
        if df.empty:
            return False
        mask = pd.Series(True, index=df.index)
        for col, val in match_criteria.items():
            if col not in df.columns:
                return False
            if isinstance(val, float):
                col_num = pd.to_numeric(df[col], errors="coerce")
                mask &= np.isclose(col_num, val, rtol=1e-5, atol=1e-8)
            else:
                mask &= df[col].astype(str).str.strip() == str(val).strip()
        return bool(mask.any())
    except Exception:
        return False


def _save_performance_row(res_csv_save_path: str, res_data: Dict[str, Any]):
    """
    성능 지표 행(res_data)을 CSV 파일에 저장합니다.
    기존 CSV 파일에 새로운 컬럼이 추가된 경우에도 컬럼 정렬이 깨지지 않도록 병합하여 저장합니다.
    """
    os.makedirs(os.path.dirname(res_csv_save_path), exist_ok=True)
    new_df = pd.DataFrame([res_data])

    if not os.path.isfile(res_csv_save_path):
        new_df.to_csv(res_csv_save_path, index=False)
        return

    try:
        existing_df = pd.read_csv(res_csv_save_path)
        if list(existing_df.columns) == list(new_df.columns):
            new_df.to_csv(res_csv_save_path, mode="a", header=False, index=False)
        else:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(res_csv_save_path, index=False)
    except Exception:
        new_df.to_csv(res_csv_save_path, mode="a", header=False, index=False)


def _format_strength_stats(stats_dict: Optional[Dict[str, float]], key: str) -> Any:
    """강도 통계량 딕셔너리에서 특정 통계량(min, q1, q2, q3, max, mean) 값을 추출합니다."""
    if stats_dict is None or key not in stats_dict:
        return np.nan
    return stats_dict[key]


def _load_chronos2_pipeline(model_ver: str, cl: int) -> Chronos2Pipeline:
    """Amazon Chronos-2 파이프라인을 로드하고 지정된 컨텍스트 길이 및 디바이스로 설정합니다."""
    pipeline = Chronos2Pipeline.from_pretrained(model_ver)
    if int(cl) > pipeline.model.chronos_config.context_length:
        pipeline.model.chronos_config.context_length = int(cl)
    pipeline.model.to(DEVICE)
    pipeline.model.eval()
    return pipeline


def run_chronos_experiment(data_name, tsfm_method, tr_data, te_data, cl, hl, patch_size):
    pred_save_dir = _resolve_path(RES_PATH['predictions'].get(tsfm_method, f'./results/predictions/{tsfm_method}/'))
    os.makedirs(pred_save_dir, exist_ok=True)
    pred_file_name = f"{DATE}_{tsfm_method}_{data_name}_cl[{cl}]_hl[{hl}]_preds.npy"
    pred_npy_save_path = os.path.join(pred_save_dir, pred_file_name)

    res_save_dir = _resolve_path(RES_PATH['performance'].get(tsfm_method, f'./results/performance/{tsfm_method}/'))
    os.makedirs(res_save_dir, exist_ok=True)
    res_file_name = f"{DATE}_{tsfm_method}_performance.csv"
    res_csv_save_path = os.path.join(res_save_dir, res_file_name)

    # 이미 저장된 실험 결과이면 pass
    if _is_experiment_completed(
        res_csv_path=res_csv_save_path,
        pred_npy_path=pred_npy_save_path,
        match_criteria={
            'data': data_name,
            'method': tsfm_method,
            'cl': cl,
            'hl': hl,
        },
    ):
        print(f"⏭️ [Skip] Chronos-2 experiment already completed for {data_name} (cl={cl}, hl={hl}). Passing...")
        return

    # 1. Chronos-2 모델 로드 (InstanceNorm 기반 정규화/역정규화 내장)
    model_ver = PARAMS[tsfm_method]['version']
    print(f"Loading Chronos-2: {model_ver}...")
    tsfm = _load_chronos2_pipeline(model_ver, cl=cl)

    # 2. Chronos-2 단독(Zero-shot) 예측 수행
    print("Predicting with Chronos-2...")
    start_inf_time = time.time()

    base_preds, base_actuals = forecast_test_series(
        model=tsfm.model,
        data=te_data,
        cl=cl,
        hl=hl,
    )

    inf_time = time.time() - start_inf_time
    print(f"# of predictions: {len(base_preds)}, # of actuals: {len(base_actuals)}")
    print(f"Predictions (first 5): {base_preds[:5]}, Actuals (first 5): {base_actuals[:5]}")
    print(f"Chronos-2 Inference Time: {inf_time:.2f}s")

    # 3. 예측 결과(.npy) 저장
    np.save(pred_npy_save_path, base_preds)
    print(f"✅ Predictions saved to: {pred_npy_save_path}")

    # 4. 성능 지표 계산 및 결과(.csv) 저장
    mae, mse, wape, smape, scaled_mae, scaled_mse = calculate_metrics(base_actuals, base_preds, tr_data=tr_data)

    res_data = {
        'data': data_name,
        'method': tsfm_method,
        'cl': cl,
        'hl': hl,
        'mae': mae,
        'mse': mse,
        'scaled_mae': scaled_mae,
        'scaled_mse': scaled_mse,
        'wape': wape,
        'smape': smape,
        'inf_time': round(inf_time, 4),
    }

    print(
        f"Performance Metrics: MAE={mae:.4f}, MSE={mse:.4f}, "
        f"Scaled_MAE={scaled_mae:.4f}, Scaled_MSE={scaled_mse:.4f}, WAPE={wape:.2f}%, sMAPE={smape:.2f}%"
    )

    _save_performance_row(res_csv_save_path, res_data)
    print(f"✅ Performance metrics saved to: {res_csv_save_path}")

    # 5. 메모리 정리
    del tsfm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_lora_experiment(
    data_name,
    tsfm_method,
    ft_method,
    tr_data,
    te_data,
    cl,
    hl,
    patch_size,
    rank,
    alpha,
    dropout,
    target_modules,
    batch_size,
    lr,
    epochs,
    stride,
):
    tgt_name = ", ".join(target_modules)
    pred_save_dir = _resolve_path(RES_PATH['predictions'][ft_method])
    os.makedirs(pred_save_dir, exist_ok=True)
    pred_file_name = (
        f"{DATE}_{tsfm_method}_{data_name}_cl[{cl}]_hl[{hl}]_{ft_method}_"
        f"r[{rank}]_a[{alpha}]_d[{dropout}]_tgt_[{tgt_name}]_lr[{lr}]_e[{epochs}]_bs[{batch_size}]_st[{stride}]_preds.npy"
    )
    pred_npy_save_path = os.path.join(pred_save_dir, pred_file_name)

    res_save_dir = _resolve_path(RES_PATH['performance'][ft_method])
    os.makedirs(res_save_dir, exist_ok=True)
    res_file_name = f"{DATE}_{tsfm_method}_{ft_method}_performance.csv"
    res_csv_save_path = os.path.join(res_save_dir, res_file_name)

    # 이미 저장된 실험 결과이면 pass
    if _is_experiment_completed(
        res_csv_path=res_csv_save_path,
        pred_npy_path=pred_npy_save_path,
        match_criteria={
            'data': data_name,
            'method': tsfm_method,
            'cl': cl,
            'hl': hl,
            'ft_method': ft_method,
            'rank': rank,
            'alpha': alpha,
            'dropout': dropout,
            'target_modules': str(target_modules),
            'lr': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'stride': stride,
        },
    ):
        print(
            f"⏭️ [Skip] Chronos-2 + LoRA experiment already completed for {data_name} "
            f"(cl={cl}, hl={hl}, tgt={target_modules}). Passing..."
        )
        return

    # 1. Chronos-2 모델 로드
    model_ver = PARAMS[tsfm_method]['version']
    print(f"Loading Chronos-2: {model_ver}...")
    tsfm = _load_chronos2_pipeline(model_ver, cl=cl)

    # 2. Chronos-2 백본에 LoRA 레이어 장착
    print("Applying LoRA to Chronos-2...")
    adapters = apply_lora_to_tsfm(
        model=tsfm.model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    tsfm.model.to(DEVICE)
    tr_params_ratio = get_trainable_params_ratio(tsfm.model)

    # 3. 학습용 슬라이딩 윈도우 데이터(X_train, Y_train) 구성
    X_train, Y_train = create_sliding_windows(tr_data, cl=cl, hl=hl, stride=stride)
    print(f"Training Windows Created (stride={stride}): X_train={X_train.shape}, Y_train={Y_train.shape}")

    # 4. LoRA 어댑터 파인튜닝
    eff_bs = min(batch_size, len(X_train))
    print(f"\n🚀 Training LoRA adapter for {epochs} steps (samples={len(X_train)}, batch_size={eff_bs}, lr={lr})...")
    train_start_time = time.time()

    train_peft_adapter(
        model=tsfm.model,
        adapters=adapters,
        X=X_train,
        Y=Y_train,
        epochs=epochs,
        bs=eff_bs,
        lr=lr,
        seed=0,
        log_every=max(1, epochs // 5),
    )

    total_train_time = time.time() - train_start_time
    avg_step_time = total_train_time / epochs if epochs > 0 else 0.0

    print("\n" + "=" * 50)
    print("✅ Training Completed!")
    print(f"Total Training Time: {total_train_time:.2f}s | Avg Time per Step: {avg_step_time:.4f}s")
    print("=" * 50)

    # 5. 학습된 LoRA 어댑터 체크포인트 저장
    chk_save_dir = _resolve_path(CHK_PATH[ft_method])
    os.makedirs(chk_save_dir, exist_ok=True)
    chk_file_name = (
        f"{DATE}_{tsfm_method}_{data_name}_cl[{cl}]_hl[{hl}]_{ft_method}_"
        f"r[{rank}]_a[{alpha}]_d[{dropout}]_tgt_[{tgt_name}]_lr[{lr}]_e[{epochs}]_bs[{batch_size}]_st[{stride}].pt"
    )
    chk_save_path = os.path.join(chk_save_dir, chk_file_name)
    save_peft_adapter(adapters, chk_save_path)
    print(f"✅ LoRA adapter checkpoint saved to: {chk_save_path}")

    # 6. Chronos-2 + LoRA 테스트셋 예측 수행
    print("Predicting with Chronos-2 + LoRA...")
    start_inf_time = time.time()

    lora_preds, lora_actuals = forecast_test_series(
        model=tsfm.model,
        data=te_data,
        cl=cl,
        hl=hl,
    )

    inf_time = time.time() - start_inf_time
    print(f"# of predictions: {len(lora_preds)}, # of actuals: {len(lora_actuals)}")
    print(f"Predictions (first 5): {lora_preds[:5]}, Actuals (first 5): {lora_actuals[:5]}")
    print(f"Chronos-2 + LoRA Inference Time: {inf_time:.2f}s")

    # 7. 예측 결과(.npy) 저장
    np.save(pred_npy_save_path, lora_preds)
    print(f"✅ Predictions saved to: {pred_npy_save_path}")

    # 8. 성능 지표 계산 및 결과(.csv) 저장
    mae, mse, wape, smape, scaled_mae, scaled_mse = calculate_metrics(lora_actuals, lora_preds, tr_data=tr_data)

    res_data = {
        'data': data_name,
        'method': tsfm_method,
        'cl': cl,
        'hl': hl,
        'ft_method': ft_method,
        'rank': rank,
        'alpha': alpha,
        'dropout': dropout,
        'target_modules': str(target_modules),
        'lr': lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'stride': stride,
        'mae': mae,
        'mse': mse,
        'scaled_mae': scaled_mae,
        'scaled_mse': scaled_mse,
        'wape': wape,
        'smape': smape,
        'tr_time': round(total_train_time, 4),
        'inf_time': round(inf_time, 4),
        'tr_params_ratio': tr_params_ratio,
    }

    print(
        f"Performance Metrics: MAE={mae:.4f}, MSE={mse:.4f}, "
        f"Scaled_MAE={scaled_mae:.4f}, Scaled_MSE={scaled_mse:.4f}, WAPE={wape:.2f}%, sMAPE={smape:.2f}%"
    )

    _save_performance_row(res_csv_save_path, res_data)
    print(f"✅ Performance metrics saved to: {res_csv_save_path}")

    # 9. 메모리 해제 및 모델 구조 원복
    print("🔄 Reverting model structure and clearing cache...")
    tsfm.model = remove_lora_from_tsfm(tsfm.model)
    del tsfm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_chronos_moe_experiment(
    data_name,
    tsfm_method,
    ft_method,
    decomp_method,
    tr_data,
    te_data,
    cl,
    hl,
    patch_size,
    rank,
    alpha,
    dropout,
    target_modules,
    batch_size,
    lr,
    epochs,
    stride,
    inference_method="single, ensemble",
    threshold_method=None,
    strength_cache: Optional[Dict[Any, Any]] = None,
):
    """
    Chronos-2 기반 TSFM-MoE 통합 3단계 파이프라인 실험 함수 (STL + STL_EMD 및 single + ensemble 통합 실행):
      - 모델 로드 및 LoRA 구조 장착: 1회
      - 학습/테스트 시퀀스 STL & STL-EMD 강도(F_T, F_S, F_I) 추출: 1회 (strength_cache 전달 시 재사용)
      - Shared Expert 학습(또는 기존 LoRA 체크포인트 로드) 및 테스트 추론: 1회 (STL / STL_EMD 양쪽 ensemble에서 공유)
      - 각 분해 방식(STL: 4섹터 / STL_EMD: 8섹터)별 Domain Expert 학습 및 One-Pass(single + ensemble) 추론 수행
    """
    moe_key = 'MoE' if 'MoE' in RES_PATH['predictions'] else ft_method
    tgt_name = ", ".join(target_modules)
    if threshold_method is None:
        threshold_method = str(PARAMS.get('MoE', {}).get('threshold_method', 'median')).strip()
    else:
        threshold_method = str(threshold_method).strip()

    if isinstance(decomp_method, str):
        decomp_methods = [x.strip() for x in decomp_method.split(',') if x.strip()]
    else:
        decomp_methods = [str(x).strip() for x in decomp_method if str(x).strip()]

    if isinstance(inference_method, str):
        inf_methods = [x.strip() for x in inference_method.split(',') if x.strip()]
    else:
        inf_methods = [str(x).strip() for x in inference_method if str(x).strip()]

    pred_save_dir = _resolve_path(RES_PATH['predictions'][moe_key])
    os.makedirs(pred_save_dir, exist_ok=True)

    res_save_dir = _resolve_path(RES_PATH['performance'][moe_key])
    os.makedirs(res_save_dir, exist_ok=True)
    res_file_name = f"{DATE}_{tsfm_method}_MoE_performance.csv"
    res_csv_save_path = os.path.join(res_save_dir, res_file_name)

    # 이미 저장된 실험 결과인지 (decomp_method, inference_method) 별로 확인하여 미완료 항목만 추출
    pending_by_decomp: Dict[str, List[str]] = {}
    for dm in decomp_methods:
        pending_infs = []
        for inf_m in inf_methods:
            pred_file_name = (
                f"{DATE}_{tsfm_method}_MoE_[{dm}]_{inf_m}_{data_name}_cl[{cl}]_hl[{hl}]_{ft_method}_"
                f"r[{rank}]_a[{alpha}]_d[{dropout}]_tgt_[{tgt_name}]_lr[{lr}]_e[{epochs}]_bs[{batch_size}]_st[{stride}]_preds.npy"
            )
            pred_npy_save_path = os.path.join(pred_save_dir, pred_file_name)
            if _is_experiment_completed(
                res_csv_path=res_csv_save_path,
                pred_npy_path=pred_npy_save_path,
                match_criteria={
                    'data': data_name,
                    'method': f"{tsfm_method}-MoE",
                    'decomp_method': dm,
                    'inference_method': inf_m,
                    'cl': cl,
                    'hl': hl,
                    'ft_method': ft_method,
                    'rank': rank,
                    'alpha': alpha,
                    'dropout': dropout,
                    'target_modules': str(target_modules),
                    'lr': lr,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'stride': stride,
                },
            ):
                print(
                    f"⏭️ [Skip] Chronos-MoE ({dm}, {inf_m}) experiment already completed for {data_name} "
                    f"(cl={cl}, hl={hl}, tgt={target_modules}). Passing..."
                )
            else:
                pending_infs.append(inf_m)
        if pending_infs:
            pending_by_decomp[dm] = pending_infs

    if not pending_by_decomp:
        return

    # 1. Chronos-2 모델 로드 (STL / STL_EMD 공용으로 1회만 로드)
    model_ver = PARAMS[tsfm_method]['version']
    print(
        f"Loading Chronos-2 for TSFM-MoE (decomps={list(pending_by_decomp.keys())}, "
        f"threshold_method={threshold_method}): {model_ver}..."
    )
    tsfm = _load_chronos2_pipeline(model_ver, cl=cl)

    # 2. Chronos-2 백본에 LoRA 구조 장착 (1회)
    print("Applying LoRA structure to Chronos-2 backbone...")
    adapters = apply_lora_to_tsfm(
        model=tsfm.model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    tsfm.model.to(DEVICE)
    tr_params_ratio = get_trainable_params_ratio(tsfm.model)

    # 3. 학습/테스트 윈도우 생성 및 STL + STL_EMD 물리적 강도(F_T, F_S, F_I) 1회 통합 추출 (캐시 활용)
    X_train, Y_train = create_sliding_windows(tr_data, cl=cl, hl=hl, stride=stride)
    test_windows = build_test_context_windows(te_data, cl=cl, hl=hl)
    ctx_arr = test_windows[0]

    train_cache_key = ("train", data_name, len(tr_data), cl, hl, stride)
    test_cache_key = ("test", data_name, len(tr_data), len(te_data), cl, hl)

    if strength_cache is not None and train_cache_key in strength_cache:
        train_strengths_all = strength_cache[train_cache_key]
        print(f"♻️ [Cache Hit] Reusing precomputed train STL & STL_EMD strengths ({len(X_train)} windows).")
    else:
        print(f"🔎 [Unified Decomposition] Computing STL & STL_EMD strengths for {len(X_train)} training sequences (once)...")
        train_strengths_all = extract_all_strength_matrices(X_train)
        if strength_cache is not None:
            strength_cache[train_cache_key] = train_strengths_all

    if strength_cache is not None and test_cache_key in strength_cache:
        test_strengths_all = strength_cache[test_cache_key]
        print(f"♻️ [Cache Hit] Reusing precomputed test STL & STL_EMD strengths ({len(ctx_arr)} windows).")
    else:
        print(f"🔎 [Unified Decomposition] Computing STL & STL_EMD strengths for {len(ctx_arr)} test sequences (once)...")
        test_strengths_all = extract_all_strength_matrices(ctx_arr)
        if strength_cache is not None:
            strength_cache[test_cache_key] = test_strengths_all

    # 4. Shared Expert 확보 및 테스트 추론 (ensemble이 하나라도 있으면 STL / STL_EMD 공용으로 1회만 수행)
    chk_save_dir = _resolve_path(CHK_PATH.get('MoE', CHK_PATH[ft_method]))
    lora_chk_dir = _resolve_path(CHK_PATH[ft_method])
    shared_chk_file_name = (
        f"{DATE}_{tsfm_method}_{data_name}_cl[{cl}]_hl[{hl}]_{ft_method}_"
        f"r[{rank}]_a[{alpha}]_d[{dropout}]_tgt_[{tgt_name}]_lr[{lr}]_e[{epochs}]_bs[{batch_size}]_st[{stride}].pt"
    )
    shared_chk_path = os.path.join(lora_chk_dir, shared_chk_file_name)

    need_shared_expert = any(
        m.strip().lower() == "ensemble"
        for inf_list in pending_by_decomp.values()
        for m in inf_list
    )
    shared_expert = None
    shared_pred_windows = None
    shared_train_time = 0.0
    shared_inf_time = 0.0

    if need_shared_expert:
        sh_tr_start = time.time()
        shared_expert = get_or_train_shared_expert(
            model=tsfm.model,
            adapters=adapters,
            X_train=X_train,
            Y_train=Y_train,
            epochs=epochs,
            bs=batch_size,
            lr=lr,
            shared_chk_path=shared_chk_path,
        )
        shared_train_time = time.time() - sh_tr_start

        if shared_expert is not None:
            sh_inf_start = time.time()
            set_lora_state(adapters, shared_expert)
            shared_pred_windows = forecast(tsfm.model, ctx_arr, hl=hl)
            shared_inf_time = time.time() - sh_inf_start
            print(f"🌐 Shared Expert Test Inference Completed (reused across STL/STL_EMD): {shared_inf_time:.2f}s")

    # 5. 각 분해 방식(STL, STL_EMD)별로 섹터 Domain Expert 학습 + One-Pass(single & ensemble) 추론 수행
    for dm, pending_inf_methods in pending_by_decomp.items():
        dm_norm = dm.strip().upper().replace('-', '_')
        sector_keys = get_sector_keys(dm)
        num_sectors = len(sector_keys)
        chk_prefix = (
            f"{DATE}_{tsfm_method}_MoE_[{dm}]_{data_name}_cl[{cl}]_hl[{hl}]_{ft_method}_"
            f"r[{rank}]_a[{alpha}]_d[{dropout}]_tgt_[{tgt_name}]_lr[{lr}]_e[{epochs}]_bs[{batch_size}]_st[{stride}]"
        )

        print(
            f"\n🚀 [Stage 1] Building Chronos-MoE Domain Expert Pool "
            f"(decomp_method={dm}, threshold_method={threshold_method}, sectors={num_sectors}, "
            f"stride={stride}, batch_size={batch_size})..."
        )
        train_start_time = time.time()

        expert_pool, train_sector_counts, thresholds, train_strength_stats, _ = build_moe_expert_pool(
            model=tsfm.model,
            adapters=adapters,
            tr_data=tr_data,
            cl=cl,
            hl=hl,
            decomp_method=dm,
            epochs=epochs,
            bs=batch_size,
            lr=lr,
            stride=stride,
            chk_dir=chk_save_dir,
            chk_prefix=chk_prefix,
            threshold_method=threshold_method,
            train_shared_expert=False,
            precomputed_windows=(X_train, Y_train),
            precomputed_train_strengths=train_strengths_all[dm_norm],
            shared_expert_override=shared_expert,
        )

        dm_has_ensemble = any(m.strip().lower() == "ensemble" for m in pending_inf_methods)
        total_train_time = (time.time() - train_start_time) + (shared_train_time if dm_has_ensemble else 0.0)
        num_trained_experts = len(expert_pool)
        print("\n" + "=" * 50)
        print(
            f"✅ [Stage 1] ({dm}) Expert Pool Construction Completed! "
            f"({num_trained_experts}/{num_sectors} domain experts, shared_expert={'ready' if shared_expert is not None else 'none'})"
        )
        print(f"Total Chronos-MoE ({dm}) Training Time: {total_train_time:.2f}s")
        print("=" * 50)

        # One-Pass 통합 추론 (사전 계산된 test_strengths 및 shared_pred_windows 재사용)
        print(
            f"\nPredicting with Chronos-MoE (One-Pass: decomp_method={dm}, "
            f"inference_methods={pending_inf_methods})..."
        )
        start_inf_time = time.time()

        preds_by_method, moe_actuals, test_sector_counts, test_strength_stats, ctx_dfs_by_method = forecast_moe_test_series_multi(
            model=tsfm.model,
            adapters=adapters,
            expert_pool=expert_pool,
            data=te_data,
            cl=cl,
            hl=hl,
            decomp_method=dm,
            inference_methods=pending_inf_methods,
            thresholds=thresholds,
            tsfm_name=tsfm_method,
            shared_expert=shared_expert,
            precomputed_test_windows=test_windows,
            precomputed_test_strengths=test_strengths_all[dm_norm],
            precomputed_shared_preds=shared_pred_windows,
        )

        inf_time = (time.time() - start_inf_time) + (shared_inf_time if dm_has_ensemble else 0.0)
        print(f"Chronos-MoE ({dm}) One-Pass Inference Time ({pending_inf_methods}): {inf_time:.2f}s")

        for inf_m in pending_inf_methods:
            inf_m_key = inf_m.strip().lower()
            moe_preds = preds_by_method[inf_m_key]
            test_context_df = ctx_dfs_by_method[inf_m_key]

            print(f"\n--- Saving Results for Chronos-MoE (decomp_method={dm}, inference_method={inf_m}) ---")
            print(f"# of predictions: {len(moe_preds)}, # of actuals: {len(moe_actuals)}")
            print(f"Predictions (first 5): {moe_preds[:5]}, Actuals (first 5): {moe_actuals[:5]}")

            # 예측 결과(.npy) 및 테스트 컨텍스트별 F_T, F_S, F_I + 호출된 expert 정보(.csv) 저장
            base_exp_tag = (
                f"{DATE}_{tsfm_method}_MoE_[{dm}]_{inf_m}_{data_name}_cl[{cl}]_hl[{hl}]_{ft_method}_"
                f"r[{rank}]_a[{alpha}]_d[{dropout}]_tgt_[{tgt_name}]_lr[{lr}]_e[{epochs}]_bs[{batch_size}]_st[{stride}]"
            )
            pred_file_name = f"{base_exp_tag}_preds.npy"
            pred_npy_save_path = os.path.join(pred_save_dir, pred_file_name)

            np.save(pred_npy_save_path, moe_preds)
            print(f"✅ MoE Predictions saved to: {pred_npy_save_path}")

            test_context_df.insert(0, 'data', data_name)
            test_context_df.insert(1, 'cl', cl)
            test_context_df.insert(2, 'hl', hl)
            test_context_df['target_modules'] = str(target_modules)

            ctx_csv_file_name = f"{base_exp_tag}_context_experts.csv"
            ctx_csv_save_path = os.path.join(pred_save_dir, ctx_csv_file_name)
            test_context_df.to_csv(ctx_csv_save_path, index=False)
            print(f"✅ MoE Test Context F_T/F_S/F_I & Called Experts saved to: {ctx_csv_save_path}")

            all_ctx_csv_path = os.path.join(res_save_dir, f"{DATE}_{tsfm_method}_MoE_test_context_experts.csv")
            all_ctx_exists = os.path.isfile(all_ctx_csv_path)
            test_context_df.to_csv(all_ctx_csv_path, mode='a', header=not all_ctx_exists, index=False)

            # 성능 지표 및 시퀀스별 F_T, F_S, F_I 통계량 결과(.csv) 저장
            mae, mse, wape, smape, scaled_mae, scaled_mse = calculate_metrics(moe_actuals, moe_preds, tr_data=tr_data)

            tr_ft = train_strength_stats.get('F_T')
            tr_fs = train_strength_stats.get('F_S')
            tr_fi = train_strength_stats.get('F_I')
            te_ft = test_strength_stats.get('F_T')
            te_fs = test_strength_stats.get('F_S')
            te_fi = test_strength_stats.get('F_I')

            res_data = {
                'data': data_name,
                'method': f"{tsfm_method}-MoE",
                'decomp_method': dm,
                'threshold_method': threshold_method,
                'inference_method': inf_m,
                'num_sectors': num_sectors,
                'active_experts': num_trained_experts,
                'cl': cl,
                'hl': hl,
                'ft_method': ft_method,
                'rank': rank,
                'alpha': alpha,
                'dropout': dropout,
                'target_modules': str(target_modules),
                'lr': lr,
                'epochs': epochs,
                'batch_size': batch_size,
                'stride': stride,
                'mae': mae,
                'mse': mse,
                'scaled_mae': scaled_mae,
                'scaled_mse': scaled_mse,
                'wape': wape,
                'smape': smape,
                'tr_time': round(total_train_time, 4),
                'inf_time': round(inf_time, 4),
                'tr_params_ratio': tr_params_ratio,
                'thresholds': str(thresholds),
                'train_sector_counts': str(train_sector_counts),
                'test_sector_counts': str(test_sector_counts),
                # Training set 시퀀스별 F_T, F_S, F_I 통계량(min, q1, q2, q3, max) 및 평균(mean)
                'train_F_T_min': _format_strength_stats(tr_ft, 'min'),
                'train_F_T_q1': _format_strength_stats(tr_ft, 'q1'),
                'train_F_T_q2': _format_strength_stats(tr_ft, 'q2'),
                'train_F_T_q3': _format_strength_stats(tr_ft, 'q3'),
                'train_F_T_max': _format_strength_stats(tr_ft, 'max'),
                'train_F_T_mean': _format_strength_stats(tr_ft, 'mean'),
                'train_F_S_min': _format_strength_stats(tr_fs, 'min'),
                'train_F_S_q1': _format_strength_stats(tr_fs, 'q1'),
                'train_F_S_q2': _format_strength_stats(tr_fs, 'q2'),
                'train_F_S_q3': _format_strength_stats(tr_fs, 'q3'),
                'train_F_S_max': _format_strength_stats(tr_fs, 'max'),
                'train_F_S_mean': _format_strength_stats(tr_fs, 'mean'),
                'train_F_I_min': _format_strength_stats(tr_fi, 'min'),
                'train_F_I_q1': _format_strength_stats(tr_fi, 'q1'),
                'train_F_I_q2': _format_strength_stats(tr_fi, 'q2'),
                'train_F_I_q3': _format_strength_stats(tr_fi, 'q3'),
                'train_F_I_max': _format_strength_stats(tr_fi, 'max'),
                'train_F_I_mean': _format_strength_stats(tr_fi, 'mean'),
                # Test set 시퀀스별 F_T, F_S, F_I 통계량(min, q1, q2, q3, max) 및 평균(mean)
                'test_F_T_min': _format_strength_stats(te_ft, 'min'),
                'test_F_T_q1': _format_strength_stats(te_ft, 'q1'),
                'test_F_T_q2': _format_strength_stats(te_ft, 'q2'),
                'test_F_T_q3': _format_strength_stats(te_ft, 'q3'),
                'test_F_T_max': _format_strength_stats(te_ft, 'max'),
                'test_F_T_mean': _format_strength_stats(te_ft, 'mean'),
                'test_F_S_min': _format_strength_stats(te_fs, 'min'),
                'test_F_S_q1': _format_strength_stats(te_fs, 'q1'),
                'test_F_S_q2': _format_strength_stats(te_fs, 'q2'),
                'test_F_S_q3': _format_strength_stats(te_fs, 'q3'),
                'test_F_S_max': _format_strength_stats(te_fs, 'max'),
                'test_F_S_mean': _format_strength_stats(te_fs, 'mean'),
                'test_F_I_min': _format_strength_stats(te_fi, 'min'),
                'test_F_I_q1': _format_strength_stats(te_fi, 'q1'),
                'test_F_I_q2': _format_strength_stats(te_fi, 'q2'),
                'test_F_I_q3': _format_strength_stats(te_fi, 'q3'),
                'test_F_I_max': _format_strength_stats(te_fi, 'max'),
                'test_F_I_mean': _format_strength_stats(te_fi, 'mean'),
            }

            print(
                f"Performance Metrics (Chronos-MoE-{dm}-{inf_m}): "
                f"MAE={mae:.4f}, MSE={mse:.4f}, Scaled_MAE={scaled_mae:.4f}, Scaled_MSE={scaled_mse:.4f}, "
                f"WAPE={wape:.2f}%, sMAPE={smape:.2f}%"
            )

            _save_performance_row(res_csv_save_path, res_data)
            print(f"✅ MoE Performance metrics saved to: {res_csv_save_path}")

    # 6. 메모리 해제 및 모델 구조 원복 (1회)
    print("🔄 Reverting model structure and clearing cache...")
    tsfm.model = remove_lora_from_tsfm(tsfm.model)
    del tsfm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    tsfm_method = TSFM_METHOD
    pipeline_cfg = PIPELINE.get(tsfm_method, {})

    # 실험 로그(.txt) 저장 설정 (results/log 폴더)
    log_base_dir = _resolve_path("./results/log/")
    os.makedirs(log_base_dir, exist_ok=True)
    main_log_path = os.path.join(log_base_dir, f"{DATE}_{tsfm_method}_experiment_log.txt")

    active_stages = [stage for stage, enabled in pipeline_cfg.items() if enabled]
    method_log_paths = []
    if 'log' in RES_PATH and isinstance(RES_PATH['log'], dict):
        for stage_key in active_stages:
            dir_key = tsfm_method if stage_key == 'Base' else stage_key
            log_dir_rel = RES_PATH['log'].get(dir_key, f"./results/log/{dir_key}/")
            m_dir = _resolve_path(log_dir_rel)
            os.makedirs(m_dir, exist_ok=True)
            method_log_paths.append(os.path.join(m_dir, f"{DATE}_{tsfm_method}_{stage_key}_log.txt"))

    tee_logger = TeeLogger(main_log_path, *method_log_paths)
    sys.stdout = tee_logger
    sys.stderr = tee_logger

    try:
        print(f"📝 Experiment Log Recording Started -> {main_log_path}")
        if torch.cuda.is_available():
            print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
            print(f"현재 디바이스: {torch.cuda.get_device_name(0)}")

        # 공통 설정 로드
        data_name = DATA

        # Chronos-2 설정 로드
        tsfm_cfg = PARAMS[tsfm_method]
        patch_size = tsfm_cfg['patch_size']
        cl = tsfm_cfg['cl']
        hl = tsfm_cfg['hl']

        # Fine-Tuning (PEFT common & LoRA) 설정 로드
        ft_method = FT_METHOD
        ft_ratio = PARAMS['FT_RATIO']
        peft_common_cfg = PARAMS['PEFT']['common']
        peft_method_cfg = PARAMS['PEFT'][ft_method]

        target_modules = tsfm_cfg.get('target_modules', peft_common_cfg['target_modules'])
        lr = peft_common_cfg['lr']
        epochs = peft_common_cfg['epochs']
        batch_size = peft_common_cfg['batch_size']
        stride = peft_common_cfg['stride']

        rank = peft_method_cfg['rank']
        alpha = peft_method_cfg['alpha']
        dropout = peft_method_cfg['dropout']

        # MoE 설정 로드
        moe_cfg = PARAMS.get('MoE', {})
        decomp_method_str = str(moe_cfg.get('decomp_method', 'STL, STL_EMD'))
        inference_method_str = str(moe_cfg.get('inference_method', 'single, ensemble'))
        threshold_method_str = str(moe_cfg.get('threshold_method', 'median')).strip()

        # 실험 조합 생성용 리스트 변환
        tsfm_method_list = [x.strip() for x in tsfm_method.split(',')]
        data_name_list = [x.strip() for x in data_name.split(',')]
        patch_size_list = [int(x.strip()) for x in patch_size.split(',')]
        cl_list = [int(x.strip()) for x in cl.split(',')]
        hl_list = [int(x.strip()) for x in hl.split(',')]

        ft_method_list = [x.strip() for x in ft_method.split(',')]
        ft_ratio_list = [float(x.strip()) for x in ft_ratio.split(',')]
        rank_list = [int(x.strip()) for x in rank.split(',')]
        alpha_list = [int(x.strip()) for x in alpha.split(',')]
        dropout_list = [float(x.strip()) for x in dropout.split(',')]
        target_modules_list = ast.literal_eval(target_modules)
        batch_size_list = [int(x.strip()) for x in str(batch_size).split(',')]
        lr_list = [float(x.strip()) for x in lr.split(',')]
        epochs_list = [int(x.strip()) for x in epochs.split(',')]
        stride_list = [int(x.strip()) for x in str(stride).split(',')]
        decomp_method_list = [x.strip() for x in decomp_method_str.split(',')]

        # 1. Chronos-2 단독 예측 실험 조합
        tsfm_comb = list(product(data_name_list, tsfm_method_list, patch_size_list, ft_ratio_list, cl_list, hl_list))
        num_tsfm_comb = len(tsfm_comb)

        # 2. Chronos-2 + LoRA 파인튜닝 및 예측 실험 조합
        lora_comb = list(
            product(
                data_name_list,
                tsfm_method_list,
                ft_method_list,
                patch_size_list,
                ft_ratio_list,
                cl_list,
                hl_list,
                rank_list,
                alpha_list,
                dropout_list,
                target_modules_list,
                batch_size_list,
                lr_list,
                epochs_list,
                stride_list,
            )
        )
        num_lora_comb = len(lora_comb)

        # 3. Chronos-MoE (STL + STL_EMD 통합 실행 조합)
        moe_comb = list(
            product(
                data_name_list,
                tsfm_method_list,
                ft_method_list,
                patch_size_list,
                ft_ratio_list,
                cl_list,
                hl_list,
                rank_list,
                alpha_list,
                dropout_list,
                target_modules_list,
                batch_size_list,
                lr_list,
                epochs_list,
                stride_list,
            )
        )
        num_moe_comb = len(moe_comb)

        # (1) Chronos-2 단독(Base) 예측 실행
        if pipeline_cfg.get('Base', False):
            for idx, (dn_item, tm_item, ps_item, fr_item, cl_item, hl_item) in enumerate(tsfm_comb, 1):
                print("\n" + "=" * 60)
                print(f"[Chronos-2 Standalone] Experiment [{idx} / {num_tsfm_comb}]")
                print(
                    f"data_name: {dn_item}, tsfm_method: {tm_item}, patch_size: {ps_item}, "
                    f"ft_ratio: {fr_item}, cl: {cl_item}, hl: {hl_item}"
                )

                df_path = _resolve_path(DATA_PATH[dn_item])
                tgt_col = DATASET[dn_item]['target_col']
                df_raw = pd.read_csv(df_path)

                target = df_raw[tgt_col].fillna(0).values.astype(np.float32)
                ft_len = int(len(target) * fr_item)

                tr_data = target[:ft_len]
                te_data = target[ft_len - cl_item :]

                run_chronos_experiment(
                    data_name=dn_item,
                    tsfm_method=tm_item,
                    tr_data=tr_data,
                    te_data=te_data,
                    cl=cl_item,
                    hl=hl_item,
                    patch_size=ps_item,
                )

        # (2) Chronos-2 + LoRA 단일 어댑터 결합 예측 실행
        if pipeline_cfg.get('LoRA', False):
            for idx, (
                dn_item,
                tm_item,
                ft_item,
                ps_item,
                fr_item,
                cl_item,
                hl_item,
                rank_item,
                alpha_item,
                dropout_item,
                target_modules_item,
                batch_size_item,
                lr_item,
                epochs_item,
                stride_item,
            ) in enumerate(lora_comb, 1):
                print("\n" + "=" * 60)
                print(f"[Chronos-2 + LoRA] Experiment [{idx} / {num_lora_comb}]")
                print(
                    f"data_name: {dn_item}, tsfm_method: {tm_item}, ft_method: {ft_item}, "
                    f"patch_size: {ps_item}, ft_ratio: {fr_item}, cl: {cl_item}, hl: {hl_item}"
                )
                print(
                    f"rank: {rank_item}, alpha: {alpha_item}, dropout: {dropout_item}, "
                    f"target_modules: {target_modules_item}, batch_size: {batch_size_item}, "
                    f"lr: {lr_item}, epochs: {epochs_item}, stride: {stride_item}"
                )

                df_path = _resolve_path(DATA_PATH[dn_item])
                tgt_col = DATASET[dn_item]['target_col']
                df_raw = pd.read_csv(df_path)

                target = df_raw[tgt_col].fillna(0).values.astype(np.float32)
                ft_len = int(len(target) * fr_item)

                tr_data = target[:ft_len]
                te_data = target[ft_len - cl_item :]

                run_lora_experiment(
                    data_name=dn_item,
                    tsfm_method=tm_item,
                    ft_method=ft_item,
                    tr_data=tr_data,
                    te_data=te_data,
                    cl=cl_item,
                    hl=hl_item,
                    patch_size=ps_item,
                    rank=rank_item,
                    alpha=alpha_item,
                    dropout=dropout_item,
                    target_modules=target_modules_item,
                    batch_size=batch_size_item,
                    lr=lr_item,
                    epochs=epochs_item,
                    stride=stride_item,
                )

        # (3) Chronos-MoE (STL: 4 Sectors & STL_EMD: 8 Sectors 통합 실행)
        if pipeline_cfg.get('MoE', False):
            moe_strength_cache: Dict[Any, Any] = {}
            for idx, (
                dn_item,
                tm_item,
                ft_item,
                ps_item,
                fr_item,
                cl_item,
                hl_item,
                rank_item,
                alpha_item,
                dropout_item,
                target_modules_item,
                batch_size_item,
                lr_item,
                epochs_item,
                stride_item,
            ) in enumerate(moe_comb, 1):
                print("\n" + "=" * 60)
                print(
                    f"[Chronos-MoE ({decomp_method_str}, threshold={threshold_method_str})] "
                    f"Experiment [{idx} / {num_moe_comb}]"
                )
                print(
                    f"data_name: {dn_item}, tsfm_method: {tm_item}, ft_method: {ft_item}, "
                    f"decomp_method: {decomp_method_str}, threshold_method: {threshold_method_str}, "
                    f"inference_method: {inference_method_str}, "
                    f"patch_size: {ps_item}, ft_ratio: {fr_item}, cl: {cl_item}, hl: {hl_item}"
                )
                print(
                    f"rank: {rank_item}, alpha: {alpha_item}, dropout: {dropout_item}, "
                    f"target_modules: {target_modules_item}, batch_size: {batch_size_item}, "
                    f"lr: {lr_item}, epochs: {epochs_item}, stride: {stride_item}"
                )

                df_path = _resolve_path(DATA_PATH[dn_item])
                tgt_col = DATASET[dn_item]['target_col']
                df_raw = pd.read_csv(df_path)

                target = df_raw[tgt_col].fillna(0).values.astype(np.float32)
                ft_len = int(len(target) * fr_item)

                tr_data = target[:ft_len]
                te_data = target[ft_len - cl_item :]

                run_chronos_moe_experiment(
                    data_name=dn_item,
                    tsfm_method=tm_item,
                    ft_method=ft_item,
                    decomp_method=decomp_method_list,
                    tr_data=tr_data,
                    te_data=te_data,
                    cl=cl_item,
                    hl=hl_item,
                    patch_size=ps_item,
                    rank=rank_item,
                    alpha=alpha_item,
                    dropout=dropout_item,
                    target_modules=target_modules_item,
                    batch_size=batch_size_item,
                    lr=lr_item,
                    epochs=epochs_item,
                    stride=stride_item,
                    inference_method=inference_method_str,
                    threshold_method=threshold_method_str,
                    strength_cache=moe_strength_cache,
                )
    finally:
        sys.stdout = tee_logger.terminal
        sys.stderr = sys.__stderr__
        tee_logger.close()
