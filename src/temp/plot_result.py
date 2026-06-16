import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    print(os.getcwd())

    # 1. 데이터 및 예측 결과 로드
    DATA = 'Ettm2'  # 'Etth1', 'Etth2', 'Ettm1', 'Ettm2', 'Electricity', 'Exchange', 'Solar', 'Weather'
    CL   = 96
    HL   = 720
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
    
    DATA_PATH = {
        'Etth1':    './data/ETTh1.csv',
        'Etth2':    './data/ETTh2.csv',
        'Ettm1':    './data/ETTm1.csv',
        'Ettm2':    './data/ETTm2.csv',
        'Electricity':  './data/electricity.csv',
        'Exchange':     './data/exchange_rate.csv',
        'Solar':        './data/solar_energy_final.csv',
        'Weather':      './data/weather.csv',
    }

    df = pd.read_csv(DATA_PATH[DATA])
    target = df[DATASET[DATA]['target_col']].values  # 실제 타겟 컬럼

    # 파일명은 사용자 환경에 맞춰 확인 필요
    preds_base = np.load(f'./results/predictions/TimesFM/TimesFM_{DATA}_cl[{CL}]_hl[{HL}]_preds.npy')
    preds_lora = np.load(f'./results/predictions/LoRA/TimesFM_{DATA}_cl[{CL}]_hl[{HL}]_LoRA_r[2]_a[8]_d[0.1]_tgt_[qkv_proj, out]_lr[0.0001]_e[10]_bs[16]_preds.npy')

    # 2. 테스트 세트 구간 설정 (0.7 비율)
    # 실험 코드의 ft_len = int(len(target) * 0.7) 로직을 따름
    ft_len = int(len(target) * 0.7)

    # 예측값의 길이에 맞춰 실제값(Ground Truth) 슬라이싱
    # te_data = target[ft_len - 96:] 였고, forecast가 96(cl) 이후부터 예측하므로 
    # 실제 예측의 시작점은 target[ft_len] 임
    actuals = target[ft_len:]

    print('length')
    print(len(actuals), len(preds_base), len(preds_lora))
    print('sample values')
    print(f'Actuals: {actuals[:5]} / Preds_Base: {preds_base[:5]} / Preds_LoRA: {preds_lora[:5]}')
    # 3. 그래프 구현
    plt.figure(figsize=(15, 6))

    # 실제값 (검정 실선)
    plt.plot(actuals, label='Actual', color='black', alpha=0.7, linewidth=1.2)
    # 기본 TimesFM 예측 (파란 점선)
    plt.plot(preds_base, label='TimesFM Prediction', color='blue', linestyle='--', alpha=0.9)
    # LoRA 튜닝 후 예측 (빨간 점선)
    plt.plot(preds_lora, label='TimesFM + LoRA Prediction', color='red', linestyle=':', alpha=0.3)

    plt.title(f'TimesFM vs TimesFM + LoRA Forecasting Comparison ({DATA})')
    plt.xlabel('Time Steps (Test Set)')
    plt.ylabel('Oil Temperature (OT)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # 전체 결과 저장
    plt.savefig('forecast_comparison_full.png')

    # 가독성을 위해 앞부분 500개만 확대해서 별도 저장
    plt.xlim(0, 500)
    plt.title('Forecasting Comparison (Zoomed - First 500 steps)')
    plt.savefig('forecast_comparison_zoom.png')

    print(f"Total test steps: {len(actuals)}")