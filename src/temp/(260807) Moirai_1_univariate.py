import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

torch.set_printoptions(linewidth=200)

# ==========================================
# 1. Toy 데이터 생성 (Target 1개)
# ==========================================
CTX = 50  # Context length
PDT = 10  # Prediction length

dates = pd.date_range(start="2026-01-01", periods=CTX + PDT, freq="h")
t = np.arange(CTX + PDT)

# dates = pd.date_range(start="2026-01-01", periods=CTX, freq="h")
# t = np.arange(CTX)

# Target: 타겟 시계열 데이터 생성 (예: 주기적 + 잡음)
target_val = np.array(list(range(101, 161)))

# print(f"=" * 50)
# print(f"Raw context: {target_val[:CTX]} \n")
# print(f"Raw horizon: {target_val[CTX:]}")

df = pd.DataFrame({
    "timestamp": dates,
    "target": target_val,
}).set_index("timestamp")

#plt.plot(df.index, df["target"], label="Target", color="blue")
#plt.show()

# GluonTS 인스턴스로 변환
ds = PandasDataset(df, target="target")

_, test_template = split(ds, offset=-PDT)
test_data = test_template.generate_instances(prediction_length=PDT, windows=1)

# ==========================================
# 2. Moirai-1.0 모델 설정 및 추론
# ==========================================
MODEL_ID = "Salesforce/moirai-1.0-R-base"
module = MoiraiModule.from_pretrained(MODEL_ID)

# 추론용 Wrapper 객체 생성
model = MoiraiForecast(
    module=module,         # 사전학습된 Moirai 모듈
    prediction_length=PDT, # 예측 길이
    context_length=CTX,    # 입력 길이 
    patch_size=8,          # Patch size (8, 16, 32, 64, 128, "auto" 중 선택 가능)
                           #  "auto" 선택 시, 입력 길이와 예측 길이에 따라 최적의 Patch size 자동 결정
    num_samples=100,       # 확률적 예측을 위한 샘플 수 
    target_dim=1,                   # Target 변수 수
    feat_dynamic_real_dim=0,        # 과거-미래 모든 시점의 값을 알고 있는 Covariates 수
    past_feat_dynamic_real_dim=0,   # 과거 시점의 값만 알고 있는 Covariates 수
)

# 예측 파이프라인 구동
predictor = model.create_predictor(batch_size=1)
forecasts = list(predictor.predict(test_data.input))
forecast = forecasts[0]

median_pred = np.median(forecast.samples, axis=0)
p10_pred = np.percentile(forecast.samples, 10, axis=0)
p90_pred = np.percentile(forecast.samples, 90, axis=0)

# print(f"=" * 50)
# print(f"p10 Prediction: {p10_pred} \n")
# print(f"p50 (Median) Prediction: {median_pred} \n")
# print(f"p90 Prediction: {p90_pred} \n")


# # ==========================================
# # 결과 시각화 (Target 예측 단독 그래프)
# # ==========================================
# fig, ax1 = plt.subplots(figsize=(12, 6))

# past_dates = dates[:CTX]
# future_dates = dates[CTX:]

# # ---- Target 예측 결과 ----
# ax1.plot(past_dates, df["target"].values[:CTX], label="Context (Past Target)", color="black")
# ax1.plot(future_dates, df["target"].values[CTX:], label="Ground Truth (Future)", color="green", linestyle="--")

# # 확률 예측 샘플에서 중앙값 및 신뢰구간 계산
# median_pred = np.median(forecast.samples, axis=0)
# p10_pred = np.percentile(forecast.samples, 10, axis=0)
# p90_pred = np.percentile(forecast.samples, 90, axis=0)

# ax1.plot(future_dates, median_pred, label="Moirai Median (p50)", color="blue")
# ax1.fill_between(future_dates, p10_pred, p90_pred, color="blue", alpha=0.2, label="10%-90% Confidence Interval")
# ax1.set_title("Target Forecasting with Moirai-1.0", fontsize=12, fontweight="bold")
# ax1.set_xlabel("Timestamp")
# ax1.set_ylabel("Target Value")
# ax1.legend(loc="upper left")
# ax1.grid(True, linestyle=":", alpha=0.6)

# plt.tight_layout()
# plt.show()