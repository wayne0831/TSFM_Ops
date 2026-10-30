import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch  # noqa: E402
    from timesfm.torch import util as tfm_util  # noqa: E402
except ImportError:
    from timesfm_test.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch  # noqa: E402
    from timesfm_test.torch import util as tfm_util  # noqa: E402

from src.config import *

BATCH = int(PARAMS[FT_METHOD]['batch_size'].split(',')[0].strip()) if FT_METHOD in PARAMS and 'batch_size' in PARAMS[FT_METHOD] else 64


def _lora_peft_parameters(adapters):
    return [p for a in adapters.values() for p in (a.A, a.B)]


def save_peft_adapter(adapters, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: (a.A.detach().cpu(), a.B.detach().cpu())
                for k, a in adapters.items()}, path)


def train_peft_adapter(model, adapters, X, Y, epochs=500, bs=8, lr=1e-3, seed=0, log_every=50):
    """
    TimesFM 모델에 장착된 LoRA 어댑터(A, B 행렬)만 효율적으로 미세조정(PEFT)하는 훈련 함수입니다.
    
    매개변수:
      - model: 파인튜닝 대상이 되는 사전학습 TimesFM 백본 모델
      - adapters: 모델의 선형 레이어(qkv_proj, out 등)에 부착된 LoRA 어댑터 객체 목록
      - X: 입력 시계열 컨텍스트 데이터 (NumPy 배열, Shape: [N, cl])
      - Y: 예측 목표치(Ground Truth) 시계열 데이터 (NumPy 배열, Shape: [N, hl])
      - epochs: 전체 최적화 스텝 수 (기본값: 500)
      - bs: 미니배치 크기 (Batch Size, 기본값: 8)
      - lr: 학습률 (Learning Rate, 기본값: 1e-3)
      - seed: 재현성을 위한 난수 시드 (기본값: 0)
      - log_every: 손실(Loss)을 터미널에 출력할 주기 (기본값: 50스텝마다 출력)
    """
    # 1. PyTorch CPU/GPU 연산의 재현성을 위해 시드 고정
    torch.manual_seed(seed)
    
    # 2. 최적화기(Optimizer) 설정:
    # lora_parameters(adapters)를 통해 백본 가중치는 동결(Freeze)한 채, LoRA 어댑터의 A, B 행렬 파라미터만 Adam 옵티마이저에 전달
    opt = torch.optim.Adam(_lora_peft_parameters(adapters), lr=lr)
    
    # 3. 데이터 샘플링을 위한 NumPy 독립 난수 생성기 초기화
    rng = np.random.default_rng(seed)
    
    # 4. 모델을 학습 모드(Train Mode)로 전환 (Dropout 활성화 등)
    model.train()
    hl = Y.shape[1]
    
    # 5. 지정된 steps만큼 경사하강법 학습 루프 실행
    for epoch in range(1, epochs + 1):
        # 전체 데이터셋 X의 크기(len(X)) 내에서 미니배치 크기(bs)만큼 무작위 인덱스 복원추출
        idx = rng.integers(0, len(X), size=bs)
        
        # NumPy 배열에서 미니배치를 추출한 뒤 PyTorch Tensor로 변환하고 지정된 연산 장치(DEV: cuda/cpu)로 전송
        cx = torch.from_numpy(X[idx]).to(DEVICE)  # Shape: [bs, cl] (Context Window)
        ty = torch.from_numpy(Y[idx]).to(DEVICE)  # Shape: [bs, hl] (Prediction Horizon Target)
        
        # 6. 순전파(Forward Pass) 및 손실값(Loss) 계산:
        # - _predict_horizon(model, cx, hl): 입력 컨텍스트(cx)를 모델에 주입하여 hl 길이의 미래 예측값을 산출
        # - torch.mean((pred - ty) ** 2): 예측값과 실제 정답(ty) 간의 평균 제곱 오차(MSE Loss) 계산
        pred = _predict_horizon(model, cx, hl)
        loss = torch.mean((pred - ty) ** 2)
        
        # 7. 역전파(Backpropagation) 및 파라미터 업데이트:
        opt.zero_grad()  # 이전 스텝에서 누적된 기울기(Gradient) 초기화
        loss.backward()   # 현재 배치의 손실에 대해 LoRA 파라미터들의 기울기 계산
        opt.step()        # 계산된 기울기를 바탕으로 LoRA 어댑터 가중치(A, B) 갱신
        
        # 8. 주기적인 학습 로그 출력
        if log_every and epoch % log_every == 0:
            # flush=True로 터미널 버퍼링 없이 즉시 출력되도록 설정
            print(f"  epoch {epoch:>4} / {epochs}   loss {loss.item():.5f}", flush=True)
            
    # 9. 학습 완료 후 평가/추론 모드로 전환 (Dropout 비활성화, 결정론적 연산 유지)
    model.eval()


def _prefill(model, ctx_t):
    """
    (B, cl) 크기의 시계열 텐서를 입력받아 TimesFM 백본을 거친 뒤,
    마지막 패치 위치에서 산출되는 (B, 128, 10) 분위수(Quantile) 예측치를 반환합니다.
    (PyTorch Autograd 미분 그래프가 끊기지 않고 유지되어 LoRA 파인튜닝 시 역전파 가능)

    동작 요약:
      1. 시계열을 패치 단위(model.p = 32)로 분할
      2. Welford 알고리즘 기반 Running Statistics(평균 mu, 표준편차 sg)를 패치 축을 따라 순차 계산
      3. 누적 정규화(ReVIN) 적용 후 백본 모델 연산 수행
      4. 모델 출력을 다시 원래 통계량(cmu, csg)으로 역정규화(De-normalization)
      5. 최종 예측 구간(Horizon) 및 10개 분위수 차원 슬라이싱
    """
    # 1. 배치 크기(Batch Size) 추출 (예: B)
    bsz, seq_len = ctx_t.shape
    p_size = model.p

    # 컨텍스트 길이가 패치 크기(model.p)의 배수가 아닌 경우 좌측 패딩(Left Padding) 및 마스크 처리
    rem = seq_len % p_size
    if rem != 0:
        pad_len = p_size - rem
        pad_tensor = torch.zeros((bsz, pad_len), dtype=ctx_t.dtype, device=ctx_t.device)
        ctx_padded = torch.cat([pad_tensor, ctx_t], dim=1)
        patched = ctx_padded.reshape(bsz, -1, p_size)
        masks = torch.zeros_like(patched, dtype=torch.bool)
        masks[:, 0, :pad_len] = True
    else:
        # 2. 1D 컨텍스트 시계열을 모델 패치 길이(model.p, 보통 32) 단위로 패치화(Patching)
        # Shape: (B, 512) -> (B, num_patches, model.p)
        # 512 길이 기준 model.p가 32이면 (B, 16, 32)가 됨
        patched = ctx_t.reshape(bsz, -1, p_size)

        # 3. 마스크(Masks) 생성:
        # 패딩이나 결측치가 없으므로 전부 유효 관측치(False)로 채운 bool 텐서 생성
        # Shape: (B, num_patches, model.p)
        masks = torch.zeros_like(patched, dtype=torch.bool)

    # 4. 인과적(Causal) 온라인 누적 정규화(Running Stats)를 위한 통계량 버퍼 초기화
    # - n: 누적된 유효 데이터 수 (Shape: [B])
    # - mu: 누적 평균 (Shape: [B])
    # - sg: 누적 표준편차 (Shape: [B])
    n = torch.zeros(bsz, device=ctx_t.device)
    mu = torch.zeros(bsz, device=ctx_t.device)
    sg = torch.zeros(bsz, device=ctx_t.device)

    # 각 패치 시점별로 누적된 평균과 표준편차를 저장할 리스트
    mus, sgs = [], []

    # 5. 시간 축(패치 인덱스)을 따라 순차적으로 누적 통계량(Welford 방식) 업데이트
    # 미래 정보가 과거 패치 통계량에 섞이지 않도록 인과적(Causal) 방식으로 계산함
    for i in range(patched.shape[1]):
        # i번째 패치 patched[:, i] (Shape: [B, 32])를 반영하여 n, mu, sg를 갱신
        (n, mu, sg), _ = tfm_util.update_running_stats(
            n, mu, sg, patched[:, i], masks[:, i]
        )
        mus.append(mu)  # 현재 패치 시점까지의 누적 평균 저장
        sgs.append(sg)  # 현재 패치 시점까지의 누적 표준편차 저장

    # 리스트에 모인 통계량을 패치 차원(dim 1)으로 결합
    # cmu, csg Shape: (B, num_patches, 1) 또는 (B, num_patches)
    cmu, csg = torch.stack(mus, 1), torch.stack(sgs, 1)

    # 6. ReVIN 정규화 (Forward ReVIN):
    # 각 패치를 해당 패치 시점의 누적 통계량(cmu, csg)으로 표준화 (y_norm = (y - cmu) / csg)
    # Shape: (B, num_patches, model.p) -> (B, 16, 32)
    normed = tfm_util.revin(patched, cmu, csg, reverse=False)

    # 7. TimesFM 모델 순전파 (Forward Pass):
    # Transformer 백본 및 출력 헤드를 거쳐 은닉 상태와 예측 텐서(out_ts)를 산출
    # out_ts Shape: (B, num_patches, model.o * model.q) 형태의 정규화된 출력
    # (여기서 model.o: 예측 horizon 출력 길이=128, model.q: 분위수 개수=10)
    (_, _, out_ts, _), _ = model(normed, masks, None)

    # 8. ReVIN 역정규화 (Reverse ReVIN):
    # 모델의 정규화된 출력을 다시 원래 데이터의 스케일로 복원 (y_restored = y_norm * csg + cmu)
    # Shape 유지: (B, num_patches, ...)
    restored = tfm_util.revin(out_ts, cmu, csg, reverse=True)

    # 9. 최종 예측 차원 재구성(Reshape) 및 슬라이싱:
    # - restored.reshape(bsz, -1, model.o, model.q):
    #   텐서를 (배치 크기, 패치 수, 예측 스텝 수, j) = (B, 16, 128, 10) 형태로 변경
    # -[:, 0]:
    # $t=1 \sim 32$ (첫 32개 데이터만 봄) / $t=32$ 시점에서 바라본 **미래 $t=33 \sim 160$**의 예측값
    # -[:, 01:
    # $t=1 \sim 64$ (첫 64개 데이터만 봄) / $t=64$ 시점에서 바라본 미래 $t=65$부터 192스텝의 예측값
    # - [:, -1]:
    #   인과적 디코더(Causal Decoder) 구조상, '마지막 패치(Context의 끝)' 시점에서
    #   미래를 향해 쏜 최종 128스텝 예측 헤드 값만 슬라이싱하여 추출
    # 최종 반환 Shape: (B, 128, 10)
    return restored.reshape(bsz, -1, model.o, model.q)[:, -1]


def _predict_horizon(model, ctx_t, hl=None):
    """
    _prefill을 활용하여 원하는 예측 길이(hl)만큼 포인트 예측치(Shape: [B, hl])를 반환합니다.
    - hl <= model.o (128)인 경우: 1회 _prefill 수행 후 [:, :hl] 슬라이싱
    - hl > model.o (128)인 경우 (예: 192, 336, 720): model.o 단위로 자기회귀(Autoregressive) 연장 후 [:, :hl] 슬라이싱
    """
    if hl is None:
        hl = model.o

    if hl <= model.o:
        return _prefill(model, ctx_t)[..., 0][:, :hl]

    preds = []
    cur_ctx = ctx_t
    steps = math.ceil(hl / model.o)
    for _ in range(steps):
        step_pred = _prefill(model, cur_ctx)[..., 0]  # Shape: (B, model.o)
        preds.append(step_pred)
        cur_ctx = torch.cat([cur_ctx, step_pred], dim=1)

    return torch.cat(preds, dim=1)[:, :hl]


def forecast(model, ctx, hl=None, batch_size=BATCH):
    """
    (N, cl) 크기의 시계열 컨텍스트 NumPy 배열을 입력받아,
    메모리 절약을 위해 batch_size 단위로 순차 추론한 후 (N, hl) 크기의 최종 포인트 예측 NumPy 배열로 반환합니다.
    (평가 및 테스트 전용 함수로, 역전파 그래프를 생성하지 않아 VRAM을 절약합니다.)

    매개변수:
      - model: 사전학습 백본 또는 LoRA가 결합된 TimesFM 모델 (tsfm.model)
      - ctx: 입력 시계열 컨텍스트 데이터 (NumPy ndarray, Shape: [N, cl])
      - hl: 예측할 미래 스텝 수 (기본값: None이면 model.o=128 사용)
      - batch_size: 추론 시 미니배치 크기 (기본값: BATCH)

    반환값:
      - (N, hl) 형태의 미래 예측 NumPy ndarray
    """
    # 1. 배치별 추론 결과(NumPy 배열)를 차례대로 담을 빈 리스트 초기화
    outs = []

    # 2. PyTorch의 자동 미분(Autograd) 엔진 비활성화:
    # 추론 시 불필요한 계산 그래프(Computation Graph) 생성을 막아 GPU VRAM을 대폭 절약하고 연산 속도 향상
    model.eval()
    with torch.no_grad():
        # 3. 전체 데이터셋 N개를 batch_size 크기 단위로 슬라이싱하여 순회
        # range(start, stop, step) -> 인덱스 0부터 ctx의 끝까지 batch_size 간격으로 건너뜀
        for i in range(0, len(ctx), batch_size):
            # 4. 현재 슬라이스의 NumPy 배열을 PyTorch Tensor로 변환 후 연산 장치(DEV: cuda 또는 cpu)로 전송
            # Shape: [현재 배치 크기, cl] (마지막 배치는 batch_size보다 작을 수 있음)
            c = torch.from_numpy(ctx[i : i + batch_size]).to(DEVICE)

            # 5. 핵심 순전파 및 차원 슬라이싱 / 타입 캐스팅 파이프라인:
            # - _predict_horizon(model, c, hl): ReVIN 및 트랜스포머 순전파를 거쳐 [현재 배치 크기, hl] 포인트 예측 텐서 반환
            # - .float(): 출력 정밀도를 float32로 명시적 변환
            # - .cpu(): GPU 메모리에 있는 텐서를 CPU 메모리로 복사
            # - .numpy(): PyTorch 텐서를 NumPy ndarray로 변환하여 outs 리스트에 추가
            outs.append(_predict_horizon(model, c, hl=hl).float().cpu().numpy())

    # 6. outs 리스트에 쌓인 미니배치 결과 배열들을 첫 번째 축(배치 축, axis=0)을 따라 수직 결합
    # 예: [batch_size, hl] 배열들이 합쳐져 최종 [N, hl] 크기의 단일 NumPy ndarray 완성
    return np.concatenate(outs, axis=0)


def create_sliding_windows(data, cl, hl, stride=1):
    """
    1D 시계열 배열(data)로부터 (X, Y) 슬라이딩 윈도우 배열을 생성합니다.
      - X Shape: (N, cl)
      - Y Shape: (N, hl)
    """
    cl, hl = int(cl), int(hl)
    n_samples = (len(data) - cl - hl) // stride + 1
    if n_samples <= 0:
        raise ValueError(f"데이터 길이({len(data)})가 cl({cl}) + hl({hl})보다 짧아 윈도우를 생성할 수 없습니다.")

    X = np.stack([data[i : i + cl] for i in range(0, len(data) - cl - hl + 1, stride)]).astype(np.float32)
    Y = np.stack([data[i + cl : i + cl + hl] for i in range(0, len(data) - cl - hl + 1, stride)]).astype(np.float32)
    return X, Y


def forecast_test_series(model, data, cl, hl, batch_size=BATCH):
    """
    1D 테스트 시계열 데이터(data, 앞부분 cl 길이의 컨텍스트 포함)에 대해
    hl 간격으로 컨텍스트 윈도우를 구성하고, _prefill 기반 forecast 함수를 통해
    전체 테스트 구간의 예측치(predictions)와 실제값(actuals)을 반환합니다.
    """
    cl, hl = int(cl), int(hl)
    ctx_list = []
    rem_lens = []
    actuals = []

    i = cl
    while i < len(data):
        rem_len = min(hl, len(data) - i)
        ctx_list.append(data[i - cl : i])
        actuals.append(data[i : i + rem_len])
        rem_lens.append(rem_len)
        i += hl

    ctx_arr = np.stack(ctx_list).astype(np.float32)  # Shape: (N_windows, cl)
    pred_windows = forecast(model, ctx_arr, hl=hl, batch_size=batch_size)  # Shape: (N_windows, hl)

    predictions = [pred_windows[idx, :rem_len] for idx, rem_len in enumerate(rem_lens)]
    return np.concatenate(predictions, axis=0), np.concatenate(actuals, axis=0)


def calculate_metrics(actual, pred):
    mae = np.mean(np.abs(actual - pred))
    mse = np.mean((actual - pred) ** 2)
    wape = (np.sum(np.abs(actual - pred)) / (np.sum(np.abs(actual)) + 1e-8)) * 100
    smape = np.mean(np.abs(actual - pred) / ((np.abs(actual) + np.abs(pred)) / 2 + 1e-8)) * 100
    return round(float(mae), 4), round(float(mse), 4), round(float(wape), 2), round(float(smape), 2)
