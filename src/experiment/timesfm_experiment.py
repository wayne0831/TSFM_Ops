##########################################################################################################
# import libraries
###########################################################################################################

import ast
import gc
import os
import time
import math
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# [수정] PEFT 라이브러리 임포트
from peft import LoraConfig, get_peft_model
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from itertools import product

# 프로젝트 내부 모듈 (LoRA 관련 커스텀 모듈 제외)
from util.TEMP import *
from src.config import *

###########################################################################################################
# set user-defined functions
###########################################################################################################

# =========================================================================
# [신규 추가] PEFT 모델 전용 + AR(자동회귀) 루프가 포함된 훈련 함수
# =========================================================================
def train_peft_ar(peft_model, train_loader, cl, hl, patch_size, lr, epochs):
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    peft_model.train()
    history = []

    # TimesFM 2.5 고정 상수
    OUT_PATCH = 128
    POINT_FORECAST_INDEX = 5 # 10개의 Quantile 중 중앙값 인덱스
    N_Q = 10

    print(f"\n🚀 Starting PEFT AR training for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        # DataLoader에서 나오는 값의 길이에 상관없이 x와 y만 추출
        for batch_data in train_loader:
            batch_x = batch_data[0].to(DEVICE)
            batch_y = batch_data[1].to(DEVICE)
            optimizer.zero_grad()

            # ---------------------------------------------------------
            # 1. 배치 단위 로컬 정규화 (Instance Normalization)
            # ---------------------------------------------------------
            batch_mean = batch_x.mean(dim=1, keepdim=True)
            batch_std = batch_x.std(dim=1, keepdim=True) + 1e-8
            
            x_norm = (batch_x - batch_mean) / batch_std
            y_norm = (batch_y - batch_mean) / batch_std
            
            # AR 루프용 변수 초기화
            remaining_hl = hl
            current_step = 0
            loss = 0.0
            
            curr_x_norm = x_norm.clone()
            
            # ---------------------------------------------------------
            # 2. AR (자동회귀) 루프 시작
            # ---------------------------------------------------------
            while remaining_hl > 0:
                chunk_size = min(OUT_PATCH, remaining_hl)
                
                # Context 패치 변환
                num_patches = cl // int(patch_size)
                patched_x = curr_x_norm.view(batch_x.size(0), num_patches, int(patch_size))
                masks = torch.zeros_like(patched_x, dtype=torch.bool).to(DEVICE)
                
                # 모델 순방향 전파 (Forward)
                model_out = peft_model(patched_x, masks)
                
                # 출력값 언패킹
                if isinstance(model_out, tuple) and isinstance(model_out[0], tuple):
                    out_tensor = model_out[0][2]
                else:
                    out_tensor = model_out[0]
                    
                # 3차원 -> 4차원으로 차원 복구 [Batch, Num_Patches, OUT_PATCH, N_Q]
                out_tensor = out_tensor.reshape(batch_x.size(0), num_patches, OUT_PATCH, N_Q)
                
                # 현재 청크(chunk)에 해당하는 미래 예측값 추출
                chunk_preds = out_tensor[:, -1, :chunk_size, POINT_FORECAST_INDEX]
                
                # 정답(Target)과 Loss 계산
                target_chunk = y_norm[:, current_step : current_step + chunk_size]
                loss += criterion(chunk_preds, target_chunk)
                
                # 다음 스텝을 위한 업데이트
                remaining_hl -= chunk_size
                current_step += chunk_size
                
                # 아직 예측할 미래가 더 남았다면 Context 갱신
                if remaining_hl > 0:
                    # 기존 Context 뒤에 방금 예측한 값을 이어붙임 (그래프 분리(detach)로 메모리 폭발 방지)
                    combined_x = torch.cat([curr_x_norm, chunk_preds.detach()], dim=1)
                    # 항상 Context 길이(cl)만큼 최신 데이터를 잘라서 새로운 Context로 사용
                    curr_x_norm = combined_x[:, -cl:]
                    
            # 전체 청크(AR 스텝) 개수만큼 Loss 평균
            num_chunks = math.ceil(hl / OUT_PATCH)
            loss = loss / num_chunks
            
            # 역전파
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f} | Time: {time.time() - epoch_start_time:.2f}s")
    
    return peft_model, history


def run_timesfm_experiment(data_name, tsfm_method, tr_data, te_data, cl, hl, patch_size):
    eval_scaler = TimeSeriesScaler()
    _ = eval_scaler.fit_transform(tr_data)
    print(f"Global Scaling info for evaluation: Mean={eval_scaler.mean:.4f}, Std={eval_scaler.std:.4f}")

    model_ver = PARAMS[tsfm_method]['version']
    print(f"Loading TimesFM: {model_ver}...")
    tsfm = TimesFM_2p5_200M_torch.from_pretrained(model_ver)
    tsfm_config = ForecastConfig(
        max_context=cl, 
        max_horizon=hl, 
        use_continuous_quantile_head=True, 
        normalize_inputs=True
    )

    tsfm.compile(tsfm_config)
    tsfm.model.to(DEVICE)

    print("Predicting with TimesFM...")
    start_inf_time = time.time()
    
    base_preds, base_actuals = forecast(
        model_obj=tsfm, 
        data=te_data, 
        cl=cl, 
        hl=hl, 
        patch_size=patch_size
    )
    
    base_preds_scl = eval_scaler.transform(base_preds)
    base_actuals_scl = eval_scaler.transform(base_actuals)

    print(f"# of predictions: {len(base_preds)}, # of actuals: {len(base_actuals)}")
    print(f"Predictions: {base_preds[:5]}, # actuals: {base_actuals[:5]}")
    print(f'Scaled predictions: {base_preds_scl[:5]}, Scaled actuals: {base_actuals_scl[:5]}')    

    inf_time = time.time() - start_inf_time
    print(f"TimesFM Inference Time: {inf_time:.2f}s")
    
    pred_save_path = RES_PATH['predictions'][tsfm_method]
    pred_file_name = f"{tsfm_method}_{data_name}_cl[{cl}]_hl[{hl}]_preds.npy"
    pred_npy_save_path  = pred_save_path + pred_file_name
    os.makedirs(pred_save_path, exist_ok=True)
    np.save(pred_npy_save_path, base_preds)
    print(f"✅ Predictions saved to: {pred_npy_save_path}")    

    mae, mse, wape, smape = calculate_metrics(base_actuals, base_preds)
    mae_scl, mse_scl, wape_scl, smape_scl = calculate_metrics(base_actuals_scl, base_preds_scl)

    res_save_path = RES_PATH['performance'][tsfm_method]
    res_file_name = f"{tsfm_method}_performance.csv"
    res_csv_save_path = res_save_path + res_file_name
    os.makedirs(res_save_path, exist_ok=True)
    
    res_data = {
        'data': data_name, 'method': tsfm_method, 'cl': cl, 'hl': hl, 
        'mae': mae, 'mse': mse, 'wape': wape, 'smape': smape, 'inf_time':inf_time,
        'mae_scl': mae_scl, 'mse_scl': mse_scl, 'wape_scl': wape_scl, 'smape_scl': smape_scl
    }
    
    print(f"Performance Metrics: MAE={round(mae, 4)}, MSE={round(mse, 4)}, WAPE={round(wape, 2)}%, sMAPE={round(smape, 2)}%")
    print(f"Performance Metrics (Scaled): MAE={round(mae_scl, 4)}, MSE={round(mse_scl, 4)}, WAPE={round(wape_scl, 2)}%, sMAPE={round(smape_scl, 2)}%")

    res_df = pd.DataFrame([res_data])
    file_exists = os.path.isfile(res_csv_save_path)
    res_df.to_csv(res_csv_save_path, mode='a', header=not file_exists, index=False)
    print(f"✅Performance metrics saved to: {res_csv_save_path}")

    # 메모리 해제
    del tsfm
    gc.collect()
    torch.cuda.empty_cache()


def run_lora_experiment(data_name, tsfm_method, ft_method, tr_data, te_data, cl, hl, patch_size, 
                        rank, alpha, dropout, target_modules, batch_size, lr, epochs):
    
    eval_scaler = TimeSeriesScaler()
    _ = eval_scaler.fit_transform(tr_data)
    print(f"Global Scaling info for evaluation: Mean={eval_scaler.mean:.4f}, Std={eval_scaler.std:.4f}")

    model_ver = PARAMS[tsfm_method]['version']
    print(f"Loading Base TimesFM: {model_ver}...")
    tsfm = TimesFM_2p5_200M_torch.from_pretrained(model_ver)

    tsfm_config = ForecastConfig(
        max_context=cl, 
        max_horizon=hl, 
        use_continuous_quantile_head=True, 
        normalize_inputs=True 
    )

    # =========================================================================
    # [수정] PEFT 라이브러리 적용
    # =========================================================================
    print(f"Loading official PEFT LoRA Adapter...")
    base_model = tsfm.model
    lora_config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        target_modules=target_modules, 
        lora_dropout=float(dropout),
        bias="none"
        # task_type은 NLP(input_ids) 에러를 방지하기 위해 지정하지 않음
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.to(DEVICE)
    peft_model.print_trainable_parameters()
    
    # tr_params_ratio 계산 (로깅용)
    total_params = sum(p.numel() for p in peft_model.parameters())
    tr_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    tr_params_ratio = tr_params / total_params

    train_dataset = TimeSeriesDataset(tr_data, cl=cl, hl=hl, patch_size=patch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # =========================================================================
    # [수정] AR 전용 학습 함수 호출
    # =========================================================================
    train_start_time = time.time()
    
    peft_model, history = train_peft_ar(
        peft_model=peft_model,
        train_loader=train_loader, 
        cl=cl,
        hl=hl, 
        patch_size=patch_size, 
        lr=lr, 
        epochs=epochs
    )
    
    total_train_time = time.time() - train_start_time
    avg_epoch_time = total_train_time / epochs

    print("\n" + "="*50)
    print(f"✅ Training Completed!")
    print(f"Total Training Time: {total_train_time:.2f}s")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f}s")
    print("="*50)

    # 6. 추론 (tsfm 클래스 내부의 base_model이 peft_model로 래핑되어 동작함)
    tsfm.model = peft_model
    tsfm.compile(tsfm_config)
    print("Predicting with TimesFM + PEFT LoRA...")
    start_inf_time = time.time()
    
    with torch.no_grad():
        peft_model.eval()
        lora_preds, lora_actuals = forecast(
            model_obj=tsfm, 
            data=te_data, 
            cl=cl, hl=hl, 
            patch_size=patch_size
        )

    lora_preds_scl = eval_scaler.transform(lora_preds)
    lora_actuals_scl = eval_scaler.transform(lora_actuals)

    inf_time = time.time() - start_inf_time
    print(f"TimesFM + LoRA Inference Time: {inf_time:.2f}s")

    # 7. 성능 지표 계산 및 저장
    mae, mse, wape, smape = calculate_metrics(lora_actuals, lora_preds)
    mae_scl, mse_scl, wape_scl, smape_scl = calculate_metrics(lora_actuals_scl, lora_preds_scl)

    pred_save_path = RES_PATH['predictions'][ft_method]
    os.makedirs(pred_save_path, exist_ok=True)
    
    tgt_name = "_".join(target_modules)
    pred_file_name = f"{tsfm_method}_{data_name}_cl[{cl}]_hl[{hl}]_{ft_method}_r[{rank}]_a[{alpha}]_d[{dropout}]_tgt_[{tgt_name}]_lr[{lr}]_e[{epochs}]_bs[{batch_size}]_preds.npy"
    pred_npy_save_path = os.path.join(pred_save_path, pred_file_name)

    np.save(pred_npy_save_path, lora_preds)
    print(f"✅ Real-scale Predictions saved to: {pred_npy_save_path}")    

    res_save_path = RES_PATH['performance'][ft_method]
    res_file_name = f"{tsfm_method}_{ft_method}_performance.csv"
    res_csv_save_path = os.path.join(res_save_path, res_file_name)
    os.makedirs(res_save_path, exist_ok=True)
    
    res_data = {
        'data': data_name, 'method': tsfm_method, 'cl': cl, 'hl': hl, 
        'ft_method': ft_method, 'rank': rank, 'alpha': alpha, 'dropout': dropout, 
        'target_modules': str(target_modules), 'lr': lr, 'epochs': epochs, 'batch_size': batch_size, 
        'mae': mae, 'mse': mse, 'wape': wape, 'smape': smape, 
        'mae_scl': mae_scl, 'mse_scl': mse_scl, 'wape_scl': wape_scl, 'smape_scl': smape_scl,
        'tr_time': total_train_time, 'inf_time': inf_time, 'tr_params_ratio': tr_params_ratio
    }

    res_df = pd.DataFrame([res_data])
    file_exists = os.path.isfile(res_csv_save_path)
    res_df.to_csv(res_csv_save_path, mode='a', header=not file_exists, index=False)
    
    # 8. 메모리 완전 해제 (PEFT 사용 시 모델 삭제로 충분)
    print('🔄 Restoring base model and clearing memory...')
    
    # [핵심] 다음 루프를 위해 LoRA 래퍼(Wrapper)를 완전히 제거하고 순정 상태로 복구합니다.
    if hasattr(peft_model, 'unload'):
        peft_model.unload()
        
    del peft_model
    del tsfm
    
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
        print(f"현재 디바이스: {torch.cuda.get_device_name(0)}")
    
    # set common configurations
    data_name   = DATA

    # set TSFM-specific configurations
    tsfm_method = TSFM_METHOD
    patch_size  = PARAMS[tsfm_method]['patch_size']
    cl = PARAMS[TSFM_METHOD]['cl']
    hl = PARAMS[TSFM_METHOD]['hl']

    # set fine-tuning specific configurations
    ft_method   = FT_METHOD
    ft_ratio    = PARAMS['FT_RATIO']
    rank        = PARAMS[ft_method]['rank']
    alpha       = PARAMS[ft_method]['alpha']
    dropout     = PARAMS[ft_method]['dropout']
    target_modules = PARAMS[ft_method]['target_modules']
    batch_size  = PARAMS[ft_method]['batch_size']
    lr          = PARAMS[ft_method]['lr']

    # common configuration list for all experiments
    tsfm_method_list = [x.strip() for x in tsfm_method.split(',')]
    
    # TSFM configuration list
    data_name_list   = [x.strip() for x in data_name.split(',')]
    path_size_list   = [int(x.strip()) for x in patch_size.split(',')]
    cl_list = [int(x.strip()) for x in cl.split(',')]
    hl_list = [int(x.strip()) for x in hl.split(',')]

    # Fine-tuning configuration
    ft_method_list = [x.strip() for x in ft_method.split(',')]
    ft_ratio_list  = [float(x.strip()) for x in ft_ratio.split(',')]
    rank_list      = [int(x.strip()) for x in rank.split(',')]
    alpha_list     = [int(x.strip()) for x in alpha.split(',')]
    dropout_list   = [float(x.strip()) for x in dropout.split(',')]
    
    # JSON처럼 감싸진 중첩 리스트 문자열인 경우 단일 리스트로 안전하게 파싱
    parsed_tgt = ast.literal_eval(target_modules)
    target_modules_list = parsed_tgt[0] if isinstance(parsed_tgt[0], list) else parsed_tgt
    
    batch_size_list = [int(x.strip()) for x in batch_size.split(',')]
    lr_list         = [float(x.strip()) for x in lr.split(',')]    
    epochs_list     = [int(x.strip()) for x in PARAMS[ft_method]['epochs'].split(',')]

    # set TimesFM combinations for all experiments
    tsfm_comb = list(product(data_name_list, tsfm_method_list, path_size_list, ft_ratio_list, cl_list, hl_list))
    num_tsfm_comb = len(tsfm_comb)

    # set LoRA combinations for all experiments (target_modules를 단일 리스트로 넘김)
    lora_comb = list(product(data_name_list, tsfm_method_list, ft_method_list, path_size_list, ft_ratio_list, cl_list, hl_list, 
                            rank_list, alpha_list, dropout_list, [target_modules_list], batch_size_list, lr_list, epochs_list))
    num_lora_comb = len(lora_comb)

    # run TimesFM experiment
    if PIPELINE['TimesFM']:
        for idx, (dn_item, tm_item, ps_item, fr_item, cl_item, hl_item) in enumerate(tsfm_comb, 1):
            print("\n" + "="*60)        
            print(f"Experiment [{idx} / {num_tsfm_comb}]") 
            print(f'data_name: {dn_item}, tsfm_method: {tm_item}, patch_size: {ps_item}, ft_ratio: {fr_item}, cl: {cl_item}, hl: {hl_item}')
        
            df_path = DATA_PATH[dn_item]
            tgt_col = DATASET[dn_item]['target_col']
            
            if not os.path.exists(df_path):
                print(f"⚠️ {df_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
                continue
                
            df_raw  = pd.read_csv(df_path)

            target = df_raw[tgt_col].fillna(0).values.astype(np.float32)
            ft_len = int(len(target) * fr_item)

            tr_data = target[:ft_len] 
            te_data = target[ft_len - cl_item:] 

            run_timesfm_experiment(data_name=dn_item, tsfm_method=tm_item, 
                                   tr_data=tr_data, te_data=te_data, cl=cl_item, hl=hl_item, patch_size=ps_item)
        
    if PIPELINE['LoRA']:
        for idx, (dn_item, tm_item, ft_item, ps_item, fr_item, cl_item, hl_item, rank_item, alpha_item, dropout_item, target_modules_item, batch_size_item, lr_item, epochs_item) in enumerate(lora_comb, 1):
            print("\n" + "="*60)        
            print(f"Experiment [{idx} / {num_lora_comb}]") 
            print(f'data_name: {dn_item}, tsfm_method: {tm_item}, ft_method: {ft_item}, patch_size: {ps_item}, ft_ratio: {fr_item}, cl: {cl_item}, hl: {hl_item}')
            
            df_path = DATA_PATH[dn_item]
            tgt_col = DATASET[dn_item]['target_col']
            
            if not os.path.exists(df_path):
                print(f"⚠️ {df_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
                continue
                
            df_raw  = pd.read_csv(df_path)

            target = df_raw[tgt_col].fillna(0).values.astype(np.float32)
            ft_len = int(len(target) * fr_item)

            tr_data = target[:ft_len] 
            te_data = target[ft_len - cl_item:] 

            run_lora_experiment(data_name=dn_item, tsfm_method=tm_item, ft_method=ft_item, 
                                tr_data=tr_data, te_data=te_data, cl=cl_item, hl=hl_item, patch_size=ps_item, 
                                rank=rank_item, alpha=alpha_item, dropout=dropout_item, target_modules=target_modules_item, 
                                batch_size=batch_size_item, lr=lr_item, epochs=epochs_item)