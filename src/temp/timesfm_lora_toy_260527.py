import os
import sys
import ast
import pandas as pd
import numpy as np
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from peft import LoraConfig, get_peft_model
import timesfm

# ==========================================
# 1. 하이퍼파라미터 및 설정 (PARAMS)
# ==========================================
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA        = 'Etth1' #'Etth1, Etth2, Ettm1, Ettm2, Electricity, Exchange, Solar, Weather'

PARAMS = {
    'TimesFM': {
        'version': 'google/timesfm-2.5-200m-pytorch',
        'patch_size': '32',
        'cl': '96',                  # context length (여러 개 입력 가능: '96, 192')
        'hl': '96' #'96, 192, 336, 720',   # horizon length (여러 개 입력 가능: '96, 192')
    },
    'FT_RATIO': '0.7',
    'LoRA': {
        'rank': '4',
        'alpha': '8',
        'dropout': '0.1',
        'target_modules': '["qkv_proj", "attn.out", "ff0", "ff1"]', 
        'lr': '1e-3',
        'epochs': '3',
        'batch_size': '64',
    }
}

DATA_PATH = {
    'Etth1':       './data/ETTh1.csv',
    'Etth2':       './data/ETTh2.csv',
    'Ettm1':       './data/ETTm1.csv',
    'Ettm2':       './data/ETTm2.csv',
    'Electricity': './data/electricity.csv',
    'Exchange':    './data/exchange_rate.csv',
    'Solar':       './data/solar_energy_final.csv',
    'Weather':     './data/weather.csv',
}

CHK_PATH = {'LoRA': './checkpoints/LoRA/'}

DATASET = {
    'Etth1': {'target_col': 'OT'},
    'Etth2': {'target_col': 'OT'},
    'Ettm1': {'target_col': 'OT'},
    'Ettm2': {'target_col': 'OT'},
    'Electricity':  {'target_col': 'OT'},
    'Exchange':     {'target_col': 'SGD'},
    'Solar':        {'target_col': '136'},
    'Weather':      {'target_col': 'OT'}
}

PIPELINE = {'TimesFM': False, 'LoRA': True}


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    if not PIPELINE['LoRA']:
        sys.exit("LoRA pipeline is set to False. Exiting.")

    # 설정값 리스트화
    cl_list = [int(x.strip()) for x in PARAMS['TimesFM']['cl'].split(',')]
    hl_list = [int(x.strip()) for x in PARAMS['TimesFM']['hl'].split(',')]
    dataset_list = [x.strip() for x in DATA.split(',')]
    target_modules = ast.literal_eval(PARAMS['LoRA']['target_modules'])

    # ==========================================
    # 메인 실험 루프 (Dataset -> cl -> hl)
    # ==========================================
    for dataset_name in dataset_list:
        if dataset_name not in DATA_PATH:
            continue
            
        csv_path = DATA_PATH[dataset_name]
        target_col = DATASET[dataset_name]['target_col']

        if not os.path.exists(csv_path):
            print(f"⚠️ {csv_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue

        for cl, hl in product(cl_list, hl_list):
            print(f"\n{'='*70}")
            print(f"🚀 [실험 시작] Dataset: {dataset_name} | Context: {cl} | Horizon: {hl}")
            print(f"{'='*70}")

            # ==========================================
            # 2. Base TimesFM 모델 로드
            # ==========================================
            print("Loading TimesFM 2.5 Model...")
            tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                PARAMS['TimesFM']['version'],
                context_len=cl,
                horizon_len=hl,
            )
            base_model = tfm.model 

            # ==========================================
            # 3. PEFT LoRA 어댑터 설정 및 장착
            # ==========================================
            # [수정 1 반영]: task_type 속성을 삭제하여 언어모델 오작동 방지 (input_ids 에러 해결)
            lora_config = LoraConfig(
                r=int(PARAMS['LoRA']['rank']), 
                lora_alpha=int(PARAMS['LoRA']['alpha']), 
                target_modules=target_modules, 
                lora_dropout=float(PARAMS['LoRA']['dropout']),
                bias="none"
            )

            peft_model = get_peft_model(base_model, lora_config)
            peft_model.to(DEVICE)
            peft_model.print_trainable_parameters()

            # ==========================================
            # 4. CSV 데이터셋 로드 및 슬라이딩 윈도우 생성
            # ==========================================
            print(f"Loading data from {csv_path}...")
            df = pd.read_csv(csv_path)
            raw_data = df[target_col].values.astype(np.float32)

            x_data, y_data = [], []
            for i in range(len(raw_data) - cl - hl + 1):
                x_data.append(raw_data[i : i + cl])
                y_data.append(raw_data[i + cl : i + cl + hl])

            x_train = torch.tensor(np.array(x_data))
            y_train = torch.tensor(np.array(y_data))

            batch_size = int(PARAMS['LoRA']['batch_size'])
            dataset = TensorDataset(x_train, y_train)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # ==========================================
            # 5. 옵티마이저 및 설정 준비
            # ==========================================
            optimizer = optim.AdamW(peft_model.parameters(), lr=float(PARAMS['LoRA']['lr']))
            criterion = nn.MSELoss() 
            
            epochs = int(PARAMS['LoRA']['epochs'])
            patch_size = int(PARAMS['TimesFM']['patch_size'])
            POINT_FORECAST_INDEX = 5  # Quantile 배열 내에서 중간값(Point)이 위치한 인덱스

            # ==========================================
            # 6. 파인튜닝 학습 루프
            # ==========================================
            print(f"Starting fine-tuning for {epochs} epochs...")
            peft_model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)
                    optimizer.zero_grad()
                    
                    # ---------------------------------------------------------
                    # [수정 2 반영]: TimesFM 패치 변환 및 마스크 생성 (차원 에러 해결)
                    # ---------------------------------------------------------
                    # batch_x 차원: [Batch, CL] -> patched_x 차원: [Batch, Num_Patches, Patch_Size]
                    num_patches = cl // patch_size
                    patched_x = batch_x.view(batch_x.size(0), num_patches, patch_size)
                    
                    # 마스크 생성 (패딩을 하지 않으므로 전부 0=False 처리)
                    masks = torch.zeros_like(patched_x, dtype=torch.bool).to(DEVICE)
                    
                    # 모델 예측
                    model_out = peft_model(patched_x, masks) 
                    
                    # ---------------------------------------------------------
                    # [수정된 부분] 출력값 언패킹 및 차원 복구 (Reshape)
                    # ---------------------------------------------------------
                    if isinstance(model_out, tuple) and isinstance(model_out[0], tuple):
                        out_tensor = model_out[0][2] 
                    else:
                        out_tensor = model_out[0]
                    
                    # TimesFM 2.5의 고정 상수
                    OUT_PATCH = 128
                    N_Q = 10
                    
                    # 3차원 -> 4차원 복구: [Batch, Num_Patches, OUT_PATCH(128), N_Q(10)]
                    out_tensor = out_tensor.reshape(batch_x.size(0), num_patches, OUT_PATCH, N_Q)
                        
                    # ---------------------------------------------------------
                    # [수정된 부분] 출력 길이(최대 128)와 Target 길이 호환성 처리
                    # ---------------------------------------------------------
                    # TimesFM은 1회 Pass에 최대 128 시점만 예측합니다.
                    # hl이 128보다 크다면 모델 출력 크기에 맞춰 y값 길이를 임시로 자릅니다.
                    # (먼 미래를 전부 학습하려면 AR(자동회귀) 루프를 추가로 구현해야 합니다)
                    current_hl = min(hl, OUT_PATCH)
                    
                    # 마지막 패치가 뱉어낸 미래 예측값 중 Point Forecast(5번 인덱스) 추출
                    preds = out_tensor[:, -1, :current_hl, POINT_FORECAST_INDEX]
                    target_y = batch_y[:, :current_hl]
                    
                    # Loss 계산 및 역전파
                    loss = criterion(preds, target_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

            # ==========================================
            # 7. 학습된 어댑터 동적 경로 저장
            # ==========================================
            tgt_str = "_".join([t.replace("attn.", "") for t in target_modules])
            experiment_name = (
                f"{dataset_name}_TimesFM_cl[{cl}]_hl[{hl}]_"
                f"LoRA_fr[{PARAMS['FT_RATIO']}]_"
                f"r[{PARAMS['LoRA']['rank']}]_"
                f"a[{PARAMS['LoRA']['alpha']}]_"
                f"d[{PARAMS['LoRA']['dropout']}]_"
                f"tgt[{tgt_str}]_"
                f"e[{epochs}]_bs[{batch_size}]"
            )
            
            SAVE_DIR = os.path.join(CHK_PATH['LoRA'], experiment_name)
            os.makedirs(SAVE_DIR, exist_ok=True)
            
            peft_model.save_pretrained(SAVE_DIR)
            print(f"✅ LoRA Adapter saved successfully to {SAVE_DIR}!\n")