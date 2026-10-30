###########################################################################################################
# import libraries
###########################################################################################################
import ast
import gc
import os
import sys
import time
from itertools import product
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 루트(TSFM_Ops) 및 src 경로를 sys.path에 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
except ImportError:
    from timesfm_test.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    from timesfm_test.configs import ForecastConfig

from src.config import *
from src.model.PEFT.LoRA import (
    apply_lora_to_tsfm,
    get_trainable_params_ratio,
    remove_lora_from_tsfm,
    save_lora,
)
from src.model.PEFT.train_forecast import (
    calculate_metrics,
    create_sliding_windows,
    forecast_test_series,
    save_peft_adapter,
    train_peft_adapter,
)

###########################################################################################################
# set user-defined functions
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


def run_timesfm_experiment(data_name, tsfm_method, tr_data, te_data, cl, hl, patch_size, batch_size=64):
    # 1. TimesFM 모델 로드 및 컴파일 (normalize_inputs=True)
    model_ver = PARAMS[tsfm_method]['version']
    print(f"Loading TimesFM: {model_ver}...")
    tsfm = TimesFM_2p5_200M_torch.from_pretrained(model_ver)
    tsfm_config = ForecastConfig(
        max_context=cl,
        max_horizon=hl,
        use_continuous_quantile_head=True,
        normalize_inputs=True,
    )
    tsfm.compile(tsfm_config)
    tsfm.model.to(DEVICE)

    # 2. TimesFM 단독 예측 수행 (_prefill 내부에서 패치 단위 mu, sg 기반 ReVIN 정규화/역정규화 수행)
    print("Predicting with TimesFM...")
    start_inf_time = time.time()

    base_preds, base_actuals = forecast_test_series(
        model=tsfm.model,
        data=te_data,
        cl=cl,
        hl=hl,
        batch_size=batch_size,
    )

    inf_time = time.time() - start_inf_time
    print(f"# of predictions: {len(base_preds)}, # of actuals: {len(base_actuals)}")
    print(f"Predictions (first 5): {base_preds[:5]}, Actuals (first 5): {base_actuals[:5]}")
    print(f"TimesFM Inference Time: {inf_time:.2f}s")

    # 3. 예측 결과(.npy) 저장
    pred_save_dir = _resolve_path(RES_PATH['predictions'][tsfm_method])
    os.makedirs(pred_save_dir, exist_ok=True)
    pred_file_name = f"{tsfm_method}_{data_name}_cl[{cl}]_hl[{hl}]_preds_{DATE}.npy"
    pred_npy_save_path = os.path.join(pred_save_dir, pred_file_name)

    np.save(pred_npy_save_path, base_preds)
    print(f"✅ Predictions saved to: {pred_npy_save_path}")

    # 4. 성능 지표 계산 및 결과(.csv) 저장
    mae, mse, wape, smape = calculate_metrics(base_actuals, base_preds)

    res_save_dir = _resolve_path(RES_PATH['performance'][tsfm_method])
    os.makedirs(res_save_dir, exist_ok=True)
    res_file_name = f"{tsfm_method}_performance.csv_{DATE}"
    res_csv_save_path = os.path.join(res_save_dir, res_file_name)

    res_data = {
        'data': data_name,
        'method': tsfm_method,
        'cl': cl,
        'hl': hl,
        'mae': mae,
        'mse': mse,
        'wape': wape,
        'smape': smape,
        'inf_time': round(inf_time, 4),
    }

    print(f"Performance Metrics: MAE={mae:.4f}, MSE={mse:.4f}, WAPE={wape:.2f}%, sMAPE={smape:.2f}%")

    res_df = pd.DataFrame([res_data])
    file_exists = os.path.isfile(res_csv_save_path)
    res_df.to_csv(res_csv_save_path, mode='a', header=not file_exists, index=False)
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
):
    # 1. TimesFM 모델 로드
    model_ver = PARAMS[tsfm_method]['version']
    print(f"Loading TimesFM: {model_ver}...")
    tsfm = TimesFM_2p5_200M_torch.from_pretrained(model_ver)
    tsfm_config = ForecastConfig(
        max_context=cl,
        max_horizon=hl,
        use_continuous_quantile_head=True,
        normalize_inputs=True,
    )
    tsfm.compile(tsfm_config)
    tsfm.model.to(DEVICE)

    # 2. TimesFM 백본에 LoRA 레이어 장착
    print("Applying LoRA to TimesFM...")
    adapters = apply_lora_to_tsfm(
        model=tsfm.model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    tsfm.model.to(DEVICE)
    tr_params_ratio = get_trainable_params_ratio(tsfm.model)

    # 3. 학습용 슬라이딩 윈도우 데이터(X_train, Y_train) 구성 (원본 스케일 그대로 입력 -> _prefill 내부에서 ReVIN 수행)
    X_train, Y_train = create_sliding_windows(tr_data, cl=cl, hl=hl, stride=1)
    print(f"Training Windows Created: X_train={X_train.shape}, Y_train={Y_train.shape}")

    # 4. LoRA 어댑터 파인튜닝 (train_peft_adapter: 패치 단위 Welford mu/sg 및 ReVIN 적용)
    print(f"\n🚀 Training LoRA adapter for {epochs} steps (batch_size={batch_size}, lr={lr})...")
    train_start_time = time.time()

    train_peft_adapter(
        model=tsfm.model,
        adapters=adapters,
        X=X_train,
        Y=Y_train,
        epochs=epochs,
        bs=batch_size,
        lr=lr,
        seed=0,
        log_every=max(1, epochs // 6),
    )

    total_train_time = time.time() - train_start_time
    avg_step_time = total_train_time / epochs if epochs > 0 else 0.0

    print("\n" + "=" * 50)
    print("✅ Training Completed!")
    print(f"Total Training Time: {total_train_time:.2f}s | Avg Time per Step: {avg_step_time:.4f}s")
    print("=" * 50)

    # 5. 학습된 LoRA 어댑터 체크포인트 저장
    tgt_name = ", ".join(target_modules)
    chk_save_dir = _resolve_path(CHK_PATH[ft_method])
    os.makedirs(chk_save_dir, exist_ok=True)
    chk_file_name = (
        f"{tsfm_method}_{data_name}_cl[{cl}]_hl[{hl}]_{ft_method}_"
        f"r[{rank}]_a[{alpha}]_d[{dropout}]_tgt_[{tgt_name}]_lr[{lr}]_e[{epochs}]_bs[{batch_size}]_{DATE}.pt"
    )
    chk_save_path = os.path.join(chk_save_dir, chk_file_name)
    save_peft_adapter(adapters, chk_save_path)
    print(f"✅ LoRA adapter checkpoint saved to: {chk_save_path}")

    # 6. TimesFM + LoRA 테스트셋 예측 수행 (_prefill 기반 forecast)
    print("Predicting with TimesFM + LoRA...")
    start_inf_time = time.time()

    lora_preds, lora_actuals = forecast_test_series(
        model=tsfm.model,
        data=te_data,
        cl=cl,
        hl=hl,
        batch_size=batch_size,
    )

    inf_time = time.time() - start_inf_time
    print(f"# of predictions: {len(lora_preds)}, # of actuals: {len(lora_actuals)}")
    print(f"Predictions (first 5): {lora_preds[:5]}, Actuals (first 5): {lora_actuals[:5]}")
    print(f"TimesFM + LoRA Inference Time: {inf_time:.2f}s")

    # 7. 예측 결과(.npy) 저장
    pred_save_dir = _resolve_path(RES_PATH['predictions'][ft_method])
    os.makedirs(pred_save_dir, exist_ok=True)
    pred_file_name = (
        f"{tsfm_method}_{data_name}_cl[{cl}]_hl[{hl}]_{ft_method}_"
        f"r[{rank}]_a[{alpha}]_d[{dropout}]_tgt_[{tgt_name}]_lr[{lr}]_e[{epochs}]_bs[{batch_size}]_preds_{DATE}.npy"
    )
    pred_npy_save_path = os.path.join(pred_save_dir, pred_file_name)

    np.save(pred_npy_save_path, lora_preds)
    print(f"✅ Predictions saved to: {pred_npy_save_path}")

    # 8. 성능 지표 계산 및 결과(.csv) 저장
    mae, mse, wape, smape = calculate_metrics(lora_actuals, lora_preds)

    res_save_dir = _resolve_path(RES_PATH['performance'][ft_method])
    os.makedirs(res_save_dir, exist_ok=True)
    res_file_name = f"{tsfm_method}_{ft_method}_performance_{DATE}.csv"
    res_csv_save_path = os.path.join(res_save_dir, res_file_name)

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
        'mae': mae,
        'mse': mse,
        'wape': wape,
        'smape': smape,
        'tr_time': round(total_train_time, 4),
        'inf_time': round(inf_time, 4),
        'tr_params_ratio': tr_params_ratio,
    }

    print(f"Performance Metrics: MAE={mae:.4f}, MSE={mse:.4f}, WAPE={wape:.2f}%, sMAPE={smape:.2f}%")

    res_df = pd.DataFrame([res_data])
    file_exists = os.path.isfile(res_csv_save_path)
    res_df.to_csv(res_csv_save_path, mode='a', header=not file_exists, index=False)
    print(f"✅ Performance metrics saved to: {res_csv_save_path}")

    # 9. 메모리 해제 및 모델 구조 원복
    print("🔄 Reverting model structure and clearing cache...")
    tsfm.model = remove_lora_from_tsfm(tsfm.model)
    del tsfm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
        print(f"현재 디바이스: {torch.cuda.get_device_name(0)}")

    # 공통 설정 로드
    data_name = DATA

    # TSFM 설정 로드
    tsfm_method = TSFM_METHOD
    patch_size = PARAMS[tsfm_method]['patch_size']
    cl = PARAMS[tsfm_method]['cl']
    hl = PARAMS[tsfm_method]['hl']

    # Fine-Tuning (LoRA) 설정 로드
    ft_method = FT_METHOD
    ft_ratio = PARAMS['FT_RATIO']
    rank = PARAMS[ft_method]['rank']
    alpha = PARAMS[ft_method]['alpha']
    dropout = PARAMS[ft_method]['dropout']
    target_modules = PARAMS[ft_method]['target_modules']
    batch_size = PARAMS[ft_method]['batch_size']
    lr = PARAMS[ft_method]['lr']
    epochs = PARAMS[ft_method]['epochs']

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
    batch_size_list = [int(x.strip()) for x in batch_size.split(',')]
    lr_list = [float(x.strip()) for x in lr.split(',')]
    epochs_list = [int(x.strip()) for x in epochs.split(',')]

    # 1. TimesFM 단독 예측 실험 조합
    tsfm_comb = list(product(data_name_list, tsfm_method_list, patch_size_list, ft_ratio_list, cl_list, hl_list))
    num_tsfm_comb = len(tsfm_comb)

    # 2. TimesFM + LoRA 파인튜닝 및 예측 실험 조합
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
        )
    )
    num_lora_comb = len(lora_comb)

    # TimesFM 단독 예측 실행
    if PIPELINE['TimesFM']:
        default_bs = batch_size_list[0] if batch_size_list else 64
        for idx, (dn_item, tm_item, ps_item, fr_item, cl_item, hl_item) in enumerate(tsfm_comb, 1):
            print("\n" + "=" * 60)
            print(f"[TimesFM Standalone] Experiment [{idx} / {num_tsfm_comb}]")
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

            run_timesfm_experiment(
                data_name=dn_item,
                tsfm_method=tm_item,
                tr_data=tr_data,
                te_data=te_data,
                cl=cl_item,
                hl=hl_item,
                patch_size=ps_item,
                batch_size=default_bs,
            )

    # TimesFM + LoRA 결합 예측 실행
    if PIPELINE['LoRA']:
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
        ) in enumerate(lora_comb, 1):
            print("\n" + "=" * 60)
            print(f"[TimesFM + LoRA] Experiment [{idx} / {num_lora_comb}]")
            print(
                f"data_name: {dn_item}, tsfm_method: {tm_item}, ft_method: {ft_item}, "
                f"patch_size: {ps_item}, ft_ratio: {fr_item}, cl: {cl_item}, hl: {hl_item}"
            )
            print(
                f"rank: {rank_item}, alpha: {alpha_item}, dropout: {dropout_item}, "
                f"target_modules: {target_modules_item}, batch_size: {batch_size_item}, "
                f"lr: {lr_item}, epochs: {epochs_item}"
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
            )

