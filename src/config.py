###########################################################################################################
# import libraries
###########################################################################################################

import sys
import pandas as pd
import numpy as np
from itertools import product
import torch

###########################################################################################################
# set version configurations
###########################################################################################################

DATE        = 261103
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA        = 'Etth1, Etth2, Ettm1, Ettm2, Electricity, Exchange, Solar, Weather'
TSFM_METHOD = 'TimesFM'
FT_METHOD   = 'LoRA'
MOE_METHOD  = 'MoE'
DATA_GEN_METHOD = 'KernelSynth'

###########################################################################################################
# set hyperparmeters configurations
###########################################################################################################

# name: TimesFM_cl[96]_hl[192]_LoRA_fr[0.7],r[4]_a[16]_d[0.1]_tgt[qkv_proj_out_ff0_ff1]_e[5]_st[64]
PARAMS = {
    'KernelSynth': {
        'NUM_SAMPLES': '300',
        'LENGTH': '512, 256, 128, 64',
    },
    'STL_EMD': {
        'LENGTH': '512, 256, 128, 64',
        'F_T_thr': 0.5,
        'F_S_thr': 0.5,
        'F_I_thr': 0.5,
    },
    'TimesFM': {
        'version': 'google/timesfm-2.5-200m-pytorch',
        'patch_size': '32',          # ver. 1.0: 64 / ver. 2.5: 32 /
        'cl': '512', # '96',                  # context length
        'hl': '64, 128, 256' # '96, 192, 336, 720',   # horizon length
    },
    'FT_RATIO': '0.7',
    'PEFT':{
        'common': {
            'target_modules': '[["qkv_proj", "out"], ["ff0", "ff1"]]', #, ["qkv_proj", "out", "ff0", "ff1"]]',
            'lr': '1e-4',
            'epochs': '50',
            'stride': '32',
        },
        'LoRA': {
            'rank': '8',
            'alpha': '16',
            'dropout': '0.1',
        }
    },
    'MoE': {
        'decomp_method': 'STL, STL_EMD',        # 'STL': 4개 섹터 (F_T, F_S) / 'STL_EMD': 8개 섹터 (F_T, F_S, F_I)
        'inference_method': 'single, weighted', # 'single': 속한 섹터 전문가 단독 추론 / 'weighted': F_T, F_S(, F_I) 비례 가중 앙상블
    }
}

###########################################################################################################
# set path configurations
###########################################################################################################

# data path
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
CHK_PATH  = {
    'LoRA': f'./checkpoints/LoRA/',
    'MoE':  f'./checkpoints/MoE/',
}
RES_PATH  = {
    'plot': { # .png
        'TimesFM': f'./results/plot/TimesFM/',
        'LoRA': f'./results/plot/LoRA/',
        'MoE': f'./results/plot/MoE/',
    },
    'stl_emd': { # .csv
      'SynthData': f'./results/stl_emd/{DATE}_SynthData_Num{PARAMS[DATA_GEN_METHOD]["NUM_SAMPLES"]}_STL_EMD.csv',
      'OpenData':  f'./results/stl_emd/{DATE}_OpenData_STL_EMD.csv', 
    },
    'data_generation': { # .csv, .npy
        'KernelSynth': {
            #'METADATA': f'./results/data_generation/{DATE}_KernelSynth_Num{PARAMS["KernelSynth"]["NUM_SAMPLES"]}_Len{PARAMS["KernelSynth"]["LENGTH"]}.csv',
            #'DATA': f'./results/data_generation/{DATE}_KernelSynth_Num{PARAMS["KernelSynth"]["NUM_SAMPLES"]}_Len{PARAMS["KernelSynth"]["LENGTH"]}.npy',
            #'METADATA': f'./results/stl_emd/{DATE}_KernelSynth_Num{PARAMS["KernelSynth"]["NUM_SAMPLES"]}.csv',
            'Data': f'./results/data_generation/{DATE}_KernelSynth_Num{PARAMS["KernelSynth"]["NUM_SAMPLES"]}.npy',
        }
    },
    'predictions': { # .npy
        'TimesFM': f'./results/predictions/TimesFM/',
        'LoRA': f'./results/predictions/LoRA/',
        'MoE': f'./results/predictions/MoE/',
    },
    'performance': { # .csv
        'TimesFM': f'./results/performance/TimesFM/',
        'LoRA': f'./results/performance/LoRA/',
        'MoE': f'./results/performance/MoE/',
    },
}

###########################################################################################################
# set data configurations
###########################################################################################################

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

PIPELINE = {
    'TimesFM':  False,
    'LoRA':     False,
    'MoE':      True,
}


# if __name__ == "__main__":
#     CL = PARAMS[TSFM_METHOD]['cl']
#     HL = PARAMS[TSFM_METHOD]['hl']

#     # 1. ','를 기준으로 문자열 분리 (리스트 생성)
#     cl_list = [x.strip() for x in CL.split(',')]
#     hl_list = [x.strip() for x in HL.split(',')]

#     # 2. 하나의 for문으로 처리
#     for cl, hl in product(cl_list, hl_list):
#         print(f'cl: {cl}, hl: {hl}') # 따옴표를 넣어 공백 제거 확인

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
        print(f"현재 디바이스: {torch.cuda.get_device_name(0)}")
