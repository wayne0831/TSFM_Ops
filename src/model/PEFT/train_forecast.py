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
    try:
        from timesfm_test.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch  # noqa: E402
        from timesfm_test.torch import util as tfm_util  # noqa: E402
    except ImportError:
        TimesFM_2p5_200M_torch = None
        tfm_util = None

from src.config import *

_peft_common = PARAMS.get('PEFT', {}).get('common', {})
BATCH = int(str(_peft_common.get('batch_size', '64')).split(',')[0].strip())
STRIDE = int(str(_peft_common.get('stride', '32')).split(',')[0].strip())


def _detect_tsfm_backend(model) -> str:
    """
    입력된 모델 객체로부터 TSFM 백엔드 종류('timesfm', 'chronos2', 'moirai2')를 판별합니다.
    """
    if hasattr(model, "stacked_xf"):
        return "timesfm"

    core_chronos = model.model if hasattr(model, "model") and hasattr(model.model, "encoder") else model
    if hasattr(core_chronos, "chronos_config") and hasattr(core_chronos, "encoder") and hasattr(core_chronos.encoder, "block"):
        return "chronos2"

    core_moirai = model.module if hasattr(model, "module") and hasattr(model.module, "encoder") else model
    if hasattr(core_moirai, "encoder") and hasattr(core_moirai.encoder, "layers"):
        return "moirai2"

    raise TypeError(f"지원하지 않는 TSFM 백본 모델입니다: {type(model)}")


def _set_model_mode(model, training: bool):
    """래퍼(Chronos2Pipeline, Moirai2Forecast 등) 또는 순수 nn.Module의 train/eval 모드를 안전하게 전환합니다."""
    target = model.model if (not isinstance(model, nn.Module) and hasattr(model, "model")) else model
    if training:
        target.train()
    else:
        target.eval()


def _lora_peft_parameters(adapters):
    return [p for a in adapters.values() for p in (a.A, a.B)]


def save_peft_adapter(adapters, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: (a.A.detach().cpu(), a.B.detach().cpu())
                for k, a in adapters.items()}, path)


def train_peft_adapter(model, adapters, X, Y, epochs=100, bs=64, lr=1e-4, mixup_alpha=0.4, seed=0, log_every=50):
    """
    TSFM(TimesFM 2.5 / Chronos 2 / Moirai 2) 백본에 장착된 LoRA 어댑터(A, B 행렬)만
    효율적으로 미세조정(PEFT)하는 통합 훈련 함수입니다.

    매개변수:
      - model: 파인튜닝 대상이 되는 사전학습 TSFM 백본 모델 (TimesFM, Chronos2Model, Moirai2Forecast 등)
      - adapters: 모델의 선형 레이어에 부착된 LoRA 어댑터 객체 목록
      - X: stride 간격으로 생성된 입력 시계열 컨텍스트 데이터 (NumPy 배열, Shape: [N, cl])
      - Y: 예측 목표치(Ground Truth) 시계열 데이터 (NumPy 배열, Shape: [N, hl])
      - epochs: 전체 최적화 스텝 수 (기본값: 100)
      - bs: 매 스텝마다 무작위로 샘플링할 미니배치 크기 (Batch Size, 기본값: 64)
      - lr: 학습률 (Learning Rate, 기본값: 1e-4)
      - mixup_alpha: Dynamic MixUp의 Beta(alpha, alpha) 분포 파라미터 (기본값: 0.4)
      - seed: 재현성을 위한 난수 시드 (기본값: 0)
      - log_every: 손실(Loss)을 터미널에 출력할 주기 (기본값: 50스텝마다 출력)
    """
    backend = _detect_tsfm_backend(model)

    # 1. PyTorch 및 NumPy 연산의 재현성을 위해 시드 고정
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # 2. 최적화기(Optimizer) 설정: 백본 가중치는 동결한 채 LoRA 어댑터의 A, B 행렬만 최적화
    lora_params = _lora_peft_parameters(adapters)
    opt = torch.optim.Adam(lora_params, lr=lr)

    # 3. 모델을 학습 모드(Train Mode)로 전환
    _set_model_mode(model, training=True)
    hl = Y.shape[1]
    eff_bs = min(int(bs), len(X))

    # 4. 지정된 epochs만큼 미니배치 샘플링 + Dynamic MixUp 기반 경사하강법 학습 루프 실행
    for epoch in range(1, epochs + 1):
        idx = torch.randint(0, len(X), (eff_bs,)).numpy()
        cx = torch.from_numpy(X[idx]).to(DEVICE)  # Shape: [eff_bs, cl]
        ty = torch.from_numpy(Y[idx]).to(DEVICE)  # Shape: [eff_bs, hl]

        # 데이터 스케일 왜곡 및 Gradient 폭발 방지를 위해 각 컨텍스트 시퀀스의 평균(mu), 표준편차(sigma)로 정규화
        cx_mean = cx.mean(dim=1, keepdim=True)
        cx_std = torch.clamp(cx.std(dim=1, keepdim=True), min=1e-5)
        cx_norm = (cx - cx_mean) / cx_std
        ty_norm = (ty - cx_mean) / cx_std

        # Dynamic MixUp 데이터 증강 적용 (배치 샘플이 2개 이상일 때 항상 적용)
        if eff_bs > 1:
            lam = float(rng.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(eff_bs, device=DEVICE)
            cx_in = lam * cx_norm + (1.0 - lam) * cx_norm[perm]
            ty_target = lam * ty_norm + (1.0 - lam) * ty_norm[perm]
        else:
            cx_in = cx_norm
            ty_target = ty_norm

        # 백엔드별 미분 가능한 순전파 및 손실(Loss) 계산
        if backend == "timesfm":
            pred = _predict_horizon(model, cx_in, hl)
            loss = torch.mean((pred - ty_target) ** 2)
        elif backend == "chronos2":
            loss = _compute_chronos2_loss(model, cx_in, ty_target)
        elif backend == "moirai2":
            loss = _compute_moirai2_loss(model, cx_in, ty_target)
        else:
            raise ValueError(f"알 수 없는 TSFM 백엔드입니다: {backend}")

        # 역전파(Backpropagation), 그래디언트 클리핑 및 파라미터 업데이트
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
        opt.step()

        if log_every and epoch % log_every == 0:
            print(f"  epoch {epoch:>4} / {epochs}   loss {loss.item():.5f}", flush=True)

    # 학습 완료 후 평가/추론 모드로 전환
    _set_model_mode(model, training=False)


# =====================================================================
# 1. TimesFM 2.5 전용 순전파 및 추론 헬퍼
# =====================================================================
def _prefill(model, ctx_t):
    """
    (B, cl) 크기의 시계열 텐서를 입력받아 TimesFM 백본을 거친 뒤,
    마지막 패치 위치에서 산출되는 (B, 128, 10) 분위수(Quantile) 예측치를 반환합니다.
    """
    if tfm_util is None:
        raise ImportError("TimesFM 실행을 위해 'timesfm' 패키지가 필요합니다.")

    bsz, seq_len = ctx_t.shape
    p_size = model.p

    rem = seq_len % p_size
    if rem != 0:
        pad_len = p_size - rem
        pad_tensor = torch.zeros((bsz, pad_len), dtype=ctx_t.dtype, device=ctx_t.device)
        ctx_padded = torch.cat([pad_tensor, ctx_t], dim=1)
        patched = ctx_padded.reshape(bsz, -1, p_size)
        masks = torch.zeros_like(patched, dtype=torch.bool)
        masks[:, 0, :pad_len] = True
    else:
        patched = ctx_t.reshape(bsz, -1, p_size)
        masks = torch.zeros_like(patched, dtype=torch.bool)

    n = torch.zeros(bsz, device=ctx_t.device)
    mu = torch.zeros(bsz, device=ctx_t.device)
    sg = torch.zeros(bsz, device=ctx_t.device)

    mus, sgs = [], []
    for i in range(patched.shape[1]):
        (n, mu, sg), _ = tfm_util.update_running_stats(
            n, mu, sg, patched[:, i], masks[:, i]
        )
        mus.append(mu)
        sgs.append(sg)

    cmu, csg = torch.stack(mus, 1), torch.stack(sgs, 1)
    normed = tfm_util.revin(patched, cmu, csg, reverse=False)
    (_, _, out_ts, _), _ = model(normed, masks, None)
    restored = tfm_util.revin(out_ts, cmu, csg, reverse=True)

    return restored.reshape(bsz, -1, model.o, model.q)[:, -1]


def _predict_horizon(model, ctx_t, hl=None):
    """
    _prefill을 활용하여 원하는 예측 길이(hl)만큼 포인트 예측치(Shape: [B, hl])를 반환합니다.
    """
    if hl is None:
        hl = model.o

    if hl <= model.o:
        return _prefill(model, ctx_t)[..., 0][:, :hl]

    preds = []
    cur_ctx = ctx_t
    steps = math.ceil(hl / model.o)
    for _ in range(steps):
        step_pred = _prefill(model, cur_ctx)[..., 0]
        preds.append(step_pred)
        cur_ctx = torch.cat([cur_ctx, step_pred], dim=1)

    return torch.cat(preds, dim=1)[:, :hl]


# =====================================================================
# 2. Amazon Chronos 2 전용 순전파(Loss) 및 추론 헬퍼
# =====================================================================
def _get_chronos2_core(model):
    return model.model if hasattr(model, "model") and hasattr(model.model, "chronos_config") else model


def _get_chronos2_median_idx(core_model) -> int:
    q_list = list(core_model.chronos_config.quantiles)
    if 0.5 in q_list:
        return q_list.index(0.5)
    return int(np.argmin(np.abs(np.asarray(q_list, dtype=np.float64) - 0.5)))


def _compute_chronos2_loss(model, cx_in: torch.Tensor, ty_target: torch.Tensor) -> torch.Tensor:
    """
    Chronos2Model의 내장 InstanceNorm(arcsinh 포함) 및 Multi-Quantile Pinball Loss를 사용하여
    수치적으로 안정적이고 미분 그래프가 온전히 유지되는 손실값을 산출합니다.
    """
    core_model = _get_chronos2_core(model)
    bsz, hl = ty_target.shape
    out_patch = core_model.chronos_config.output_patch_size
    max_patches = core_model.chronos_config.max_output_patches
    max_step_hl = max_patches * out_patch
    group_ids = torch.arange(bsz, dtype=torch.long, device=cx_in.device)

    if hl <= max_step_hl:
        num_patches = math.ceil(hl / out_patch)
        out = core_model(
            context=cx_in,
            group_ids=group_ids,
            num_output_patches=num_patches,
            future_target=ty_target,
        )
        return out.loss

    # 초장기 예측(hl > max_output_patches * output_patch_size) 시 청크 단위 누적 손실 계산
    q50_idx = _get_chronos2_median_idx(core_model)
    cur_ctx = cx_in
    losses = []
    offset = 0
    while offset < hl:
        step_hl = min(max_step_hl, hl - offset)
        num_patches = math.ceil(step_hl / out_patch)
        step_target = ty_target[:, offset : offset + step_hl]
        out = core_model(
            context=cur_ctx,
            group_ids=group_ids,
            num_output_patches=num_patches,
            future_target=step_target,
        )
        losses.append(out.loss)
        step_pred = out.quantile_preds[:, q50_idx, :step_hl]
        cur_ctx = torch.cat([cur_ctx, step_pred], dim=1)[..., -core_model.chronos_config.context_length :]
        offset += step_hl

    return torch.stack(losses).mean()


def _predict_horizon_chronos2(core_model, ctx_t: torch.Tensor, hl: int) -> torch.Tensor:
    bsz = ctx_t.shape[0]
    out_patch = core_model.chronos_config.output_patch_size
    max_patches = core_model.chronos_config.max_output_patches
    max_step_hl = max_patches * out_patch
    q50_idx = _get_chronos2_median_idx(core_model)
    group_ids = torch.arange(bsz, dtype=torch.long, device=ctx_t.device)

    if hl <= max_step_hl:
        num_patches = math.ceil(hl / out_patch)
        out = core_model(
            context=ctx_t,
            group_ids=group_ids,
            num_output_patches=num_patches,
        )
        return out.quantile_preds[:, q50_idx, :hl]

    preds = []
    cur_ctx = ctx_t
    rem = hl
    while rem > 0:
        step_hl = min(max_step_hl, rem)
        num_patches = math.ceil(step_hl / out_patch)
        out = core_model(
            context=cur_ctx,
            group_ids=group_ids,
            num_output_patches=num_patches,
        )
        step_pred = out.quantile_preds[:, q50_idx, :]
        preds.append(step_pred)
        cur_ctx = torch.cat([cur_ctx, step_pred], dim=1)[..., -core_model.chronos_config.context_length :]
        rem -= step_pred.shape[-1]

    return torch.cat(preds, dim=1)[:, :hl]


def _forecast_chronos2(model, ctx: np.ndarray, hl: int, inf_batch_size: int = 64) -> np.ndarray:
    core_model = _get_chronos2_core(model)
    core_model.eval()
    dev = next(core_model.parameters()).device
    out_chunks = []
    with torch.no_grad():
        for start in range(0, len(ctx), inf_batch_size):
            c_batch = torch.from_numpy(ctx[start : start + inf_batch_size]).to(device=dev, dtype=torch.float32)
            pred_batch = _predict_horizon_chronos2(core_model, c_batch, hl=int(hl)).float().cpu().numpy()
            out_chunks.append(pred_batch)
    return np.concatenate(out_chunks, axis=0)


# =====================================================================
# 3. Salesforce Moirai 2 전용 순전파(Loss) 및 추론 헬퍼
# =====================================================================
def _ensure_moirai2_forecast_wrapper(model, cl: int, hl: int):
    """
    Moirai2Forecast 래퍼가 전달되면 그대로 활용하고, 순수 Moirai2Module이 전달되면
    Moirai2Forecast 래퍼로 감싸서 반환합니다.
    """
    if hasattr(model, "_convert") and hasattr(model, "module"):
        return model

    from uni2ts.model.moirai2 import Moirai2Forecast

    dev = next(model.parameters()).device
    wrapper = Moirai2Forecast(
        module=model,
        prediction_length=int(hl),
        context_length=int(cl),
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    wrapper.to(dev)
    return wrapper


def _compute_moirai2_loss(model, cx_in: torch.Tensor, ty_target: torch.Tensor) -> torch.Tensor:
    """
    Moirai-2의 Causal Multi-Token Quantile 예측 구조(training_mode=True)를 그대로 활용하여,
    인과적 마스크(미래 누수 차단) 및 컨텍스트 전용 스케일러(PackedStdScaler) 하에서
    전체 미래 패치 및 다중 토큰 헤드(num_predict_token), 전체 분위수(quantile_levels)에 대한
    미분 가능한 Pinball Quantile Loss를 1회 순전파로 계산합니다.
    """
    from einops import rearrange

    bsz, cl = cx_in.shape
    hl = ty_target.shape[1]
    forecast_model = _ensure_moirai2_forecast_wrapper(model, cl=cl, hl=hl)
    core_module = forecast_model.module
    patch_size = core_module.patch_size
    num_pred_tok = core_module.num_predict_token
    num_quantiles = core_module.num_quantiles

    past_target = cx_in.unsqueeze(-1)
    past_observed = torch.ones_like(past_target, dtype=torch.bool)
    past_is_pad = torch.zeros((bsz, cl), dtype=torch.bool, device=cx_in.device)

    future_target = ty_target.unsqueeze(-1)
    future_observed = torch.ones_like(future_target, dtype=torch.bool)
    future_is_pad = torch.zeros((bsz, hl), dtype=torch.long, device=cx_in.device)

    with forecast_model.hparams_context(prediction_length=hl, context_length=cl):
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = forecast_model._convert(
            patch_size,
            past_target,
            past_observed,
            past_is_pad,
            future_target=future_target,
            future_observed_target=future_observed,
            future_is_pad=future_is_pad,
        )
        n_ctx = forecast_model.context_token_length(patch_size)
        n_fut = forecast_model.prediction_token_length(patch_size)

    preds, scaled_target = core_module(
        target,
        observed_mask,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
        training_mode=True,
    )

    preds = rearrange(
        preds,
        "b s (k q p) -> b s k q p",
        k=num_pred_tok,
        q=num_quantiles,
        p=patch_size,
    )
    q_levels = torch.tensor(
        core_module.quantile_levels,
        device=preds.device,
        dtype=preds.dtype,
    ).view(1, 1, num_quantiles, 1)

    step_losses = []
    max_k = min(num_pred_tok, n_fut)
    for k in range(max_k):
        t_start = n_ctx - 1
        t_end = n_ctx + n_fut - 1 - k
        pred_k = preds[:, t_start:t_end, k, :, :]                       # (B, n_fut - k, Q, P)
        tgt_k = scaled_target[:, n_ctx + k : n_ctx + n_fut, None, :]    # (B, n_fut - k, 1, P)
        mask_k = observed_mask[:, n_ctx + k : n_ctx + n_fut, None, :].to(preds.dtype)

        err = tgt_k - pred_k
        pinball = 2.0 * torch.maximum(q_levels * err, (q_levels - 1.0) * err) * mask_k
        denom = torch.clamp(mask_k.sum(), min=1.0)
        # 분위수 차원 합산 후 유효 타임스텝 평균
        step_losses.append(pinball.sum(dim=2).sum() / denom)

    return torch.stack(step_losses).mean()


def _forecast_moirai2(model, ctx: np.ndarray, hl: int, inf_batch_size: int = 32) -> np.ndarray:
    cl = ctx.shape[1]
    hl = int(hl)
    forecast_model = _ensure_moirai2_forecast_wrapper(model, cl=cl, hl=hl)
    forecast_model.eval()
    forecast_model.module.eval()

    dev = next(forecast_model.module.parameters()).device
    forecast_model.to(dev)

    q_list = list(forecast_model.module.quantile_levels)
    q50_idx = q_list.index(0.5) if 0.5 in q_list else int(np.argmin(np.abs(np.asarray(q_list) - 0.5)))

    out_chunks = []
    with torch.no_grad(), forecast_model.hparams_context(prediction_length=hl, context_length=cl):
        for start in range(0, len(ctx), inf_batch_size):
            c_np = ctx[start : start + inf_batch_size]
            bsz = len(c_np)
            past_target = torch.from_numpy(c_np).to(device=dev, dtype=torch.float32).unsqueeze(-1)
            past_observed = ~torch.isnan(past_target)
            past_is_pad = torch.zeros((bsz, cl), dtype=torch.bool, device=dev)

            q_preds = forecast_model(
                past_target=past_target,
                past_observed_target=past_observed,
                past_is_pad=past_is_pad,
            )  # Shape: (bsz, num_quantiles, hl)
            out_chunks.append(q_preds[:, q50_idx, :hl].float().cpu().numpy())

    return np.concatenate(out_chunks, axis=0)


# =====================================================================
# 4. 공통 예측(Forecast), 윈도우 생성 및 성능 지표 계산 함수
# =====================================================================
def forecast(model, ctx, hl=None):
    """
    (N, cl) 크기의 테스트 시계열 컨텍스트 전체(ctx)를 입력받아,
    (N, hl) 크기의 최종 포인트 예측 NumPy 배열을 반환합니다.
    TimesFM 2.5 / Chronos 2 / Moirai 2 백본을 모두 지원합니다.
    """
    backend = _detect_tsfm_backend(model)
    if backend == "timesfm":
        model.eval()
        with torch.no_grad():
            c = torch.from_numpy(ctx).to(DEVICE)
            out = _predict_horizon(model, c, hl=hl).float().cpu().numpy()
        return out
    elif backend == "chronos2":
        if hl is None:
            core = _get_chronos2_core(model)
            hl = core.chronos_config.max_output_patches * core.chronos_config.output_patch_size
        return _forecast_chronos2(model, ctx, hl=int(hl))
    elif backend == "moirai2":
        if hl is None:
            hl = int(getattr(getattr(model, "hparams", None), "prediction_length", 96))
        return _forecast_moirai2(model, ctx, hl=int(hl))
    else:
        raise ValueError(f"알 수 없는 TSFM 백엔드입니다: {backend}")


def create_sliding_windows(data, cl, hl, stride=None):
    """
    1D 시계열 배열(data)로부터 (X, Y) 슬라이딩 윈도우 배열을 생성합니다.
      - stride가 None이면 config.py의 PARAMS['PEFT']['common']['stride'] 값을 사용합니다.
      - X Shape: (N, cl)
      - Y Shape: (N, hl)
    """
    cl, hl = int(cl), int(hl)
    if stride is None:
        stride = int(str(PARAMS['PEFT']['common']['stride']).split(',')[0].strip())
    else:
        stride = max(1, int(stride))

    n_samples = (len(data) - cl - hl) // stride + 1
    if n_samples <= 0:
        raise ValueError(f"데이터 길이({len(data)})가 cl({cl}) + hl({hl})보다 짧아 윈도우를 생성할 수 없습니다.")

    X = np.stack([data[i : i + cl] for i in range(0, len(data) - cl - hl + 1, stride)]).astype(np.float32)
    Y = np.stack([data[i + cl : i + cl + hl] for i in range(0, len(data) - cl - hl + 1, stride)]).astype(np.float32)
    return X, Y


def forecast_test_series(model, data, cl, hl):
    """
    1D 테스트 시계열 데이터(data, 앞부분 cl 길이의 컨텍스트 포함)에 대해
    hl 간격으로 컨텍스트 윈도우를 구성하고, 통합 forecast 함수를 통해
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
    pred_windows = forecast(model, ctx_arr, hl=hl)   # Shape: (N_windows, hl)

    predictions = [pred_windows[idx, :rem_len] for idx, rem_len in enumerate(rem_lens)]
    return np.concatenate(predictions, axis=0), np.concatenate(actuals, axis=0)


def calculate_metrics(actual, pred, tr_data=None):
    """
    원래 스케일(Raw Scale) 지표(MAE, MSE, WAPE, sMAPE)와 함께
    iTransformer 등 표준 LTSF 벤치마크 방식(StandardScaler: 학습셋 평균/표준편차 기반 Z-score 정규화)의
    Scaled MAE, Scaled MSE를 계산하여 반환합니다.
    """
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    mae = np.mean(np.abs(actual - pred))
    mse = np.mean((actual - pred) ** 2)
    wape = (np.sum(np.abs(actual - pred)) / (np.sum(np.abs(actual)) + 1e-8)) * 100
    smape = np.mean(np.abs(actual - pred) / ((np.abs(actual) + np.abs(pred)) / 2 + 1e-8)) * 100

    # StandardScaler (Z-score 정규화) 기반 Scaled MAE / Scaled MSE 계산
    ref_data = np.asarray(tr_data, dtype=np.float64) if tr_data is not None else actual
    ref_mean = float(np.mean(ref_data))
    ref_std = float(np.std(ref_data))
    if ref_std < 1e-8:
        ref_std = 1.0

    actual_scaled = (actual - ref_mean) / ref_std
    pred_scaled = (pred - ref_mean) / ref_std
    scaled_mae = np.mean(np.abs(actual_scaled - pred_scaled))
    scaled_mse = np.mean((actual_scaled - pred_scaled) ** 2)

    return (
        round(float(mae), 4),
        round(float(mse), 4),
        round(float(wape), 2),
        round(float(smape), 2),
        round(float(scaled_mae), 4),
        round(float(scaled_mse), 4),
    )
