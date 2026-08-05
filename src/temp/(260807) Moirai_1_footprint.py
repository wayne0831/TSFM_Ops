import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# ==========================================
# 1. Toy 데이터 생성 (Target 1개 + Covariate 2개)
# ==========================================
CTX = 96  # Context length
PDT = 24  # Prediction length

dates = pd.date_range(start="2026-01-01", periods=CTX + PDT, freq="h")
t = np.arange(CTX + PDT)

# Target: 타겟 시계열 데이터 생성 (예: 주기적 + 잡음)
target_val = 0.05 * t + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.5, size=CTX + PDT)

# Covariates: 과거와 미래 예측 구간의 보조 변수 생성
covariate_1 = np.cos(2 * np.pi * t / 12)  # 주기적 특성을 가진 보조 변수
covariate_2 = t * 0.01                    # 추세적 특성을 가진 보조 변수

df = pd.DataFrame({
    "timestamp": dates,
    "target": target_val,
    "covariate_1": covariate_1,
    "covariate_2": covariate_2,
    #"item_id": "covariate_series_01"
}).set_index("timestamp")

# GluonTS 인스턴스로 변환
ds = PandasDataset(
    df, 
    target="target", 
    feat_dynamic_real=["covariate_1", "covariate_2"],
)

_, test_template = split(ds, offset=-PDT)
test_data = test_template.generate_instances(prediction_length=PDT, windows=1)

# # ==========================================
# # Dataframe (df) 변수별 시계열 Plotting
# # ==========================================
# fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# # 1. Target (예측 대상 시계열)
# axes[0].plot(df.index, df["target"], color="black", linewidth=1.5, label="Target (target_val)")
# axes[0].set_title("1. Target Variable (Prediction Target)", fontsize=11, fontweight="bold")
# axes[0].set_ylabel("Value")
# axes[0].grid(True, linestyle=":", alpha=0.6)
# axes[0].legend(loc="upper left")

# # 2. Covariate 1 (보조 변수 1: Periodic)
# axes[1].plot(df.index, df["covariate_1"], color="purple", linewidth=1.5, label="Covariate 1 (Periodic Cosine)")
# axes[1].set_title("2. Dynamic Covariate 1 (Known Feature 1)", fontsize=11, fontweight="bold")
# axes[1].set_ylabel("Value")
# axes[1].grid(True, linestyle=":", alpha=0.6)
# axes[1].legend(loc="upper left")

# # 3. Covariate 2 (보조 변수 2: Trend)
# axes[2].plot(df.index, df["covariate_2"], color="orange", linewidth=1.5, label="Covariate 2 (Linear Trend)")
# axes[2].set_title("3. Dynamic Covariate 2 (Known Feature 2)", fontsize=11, fontweight="bold")
# axes[2].set_xlabel("Timestamp")
# axes[2].set_ylabel("Value")
# axes[2].grid(True, linestyle=":", alpha=0.6)
# axes[2].legend(loc="upper left")

# # Context(96)와 Horizon(24) 분할 경계선 추가
# boundary_date = df.index[CTX]
# for ax in axes:
#     ax.axvline(x=boundary_date, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Context / Horizon Boundary")

# plt.tight_layout()
# plt.show()

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
    feat_dynamic_real_dim=2,        # 과거-미래 모든 시점의 값을 알고 있는 Covariates 수
    past_feat_dynamic_real_dim=0,   # 과거 시점의 값만 알고 있는 Covariates 수
)

# 예측 파이프라인 구동
predictor = model.create_predictor(batch_size=1)
forecasts = list(predictor.predict(test_data.input))
forecast = forecasts[0]

median_pred = np.median(forecast.samples, axis=0)
p10_pred = np.percentile(forecast.samples, 10, axis=0)
p90_pred = np.percentile(forecast.samples, 90, axis=0)

# ==========================================
# 3. 텐서 규격 확인 및 결과 시각화
# ==========================================
print("\n" + "="*60)
print(f"📊 Moirai Forecast Samples Shape: {forecast.samples.shape}")
print("   규격 의미: [Num_Samples(100), Prediction_Length(24)]")
print("   (Target이 1개이므로 다변량 때와 달리 가운데 차원이 사라짐)")
print("="*60 + "\n")

# 시각화 (Target 예측 결과와 입력된 Covariate를 위아래로 나누어 표시)
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

past_dates = dates[:CTX]
future_dates = dates[CTX:]

# ---- [Subplot 1] Target 예측 결과 ----
ax1 = axes[0]
ax1.plot(past_dates, df["target"].values[:CTX], label="Context (Past Target)", color="black")
ax1.plot(future_dates, df["target"].values[CTX:], label="Ground Truth (Future)", color="green", linestyle="--")

# 확률 예측 샘플에서 중앙값 및 신뢰구간 계산
median_pred = np.median(forecast.samples, axis=0)
p10_pred = np.percentile(forecast.samples, 10, axis=0)
p90_pred = np.percentile(forecast.samples, 90, axis=0)

ax1.plot(future_dates, median_pred, label="Moirai Median (p50)", color="blue")
ax1.fill_between(future_dates, p10_pred, p90_pred, color="blue", alpha=0.2, label="10%-90% Confidence Interval")
ax1.set_title("Target Forecasting with Covariates", fontsize=12, fontweight="bold")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle=":", alpha=0.6)

# ---- [Subplot 2] 입력된 Covariates (보조 변수) 패턴 ----
ax2 = axes[1]
ax2.plot(dates, df["covariate_1"].values, label="Covariate 1 (Periodic)", color="purple")
ax2.plot(dates, df["covariate_2"].values, label="Covariate 2 (Trend)", color="orange")
ax2.axvline(x=dates[CTX], color='red', linestyle='-', alpha=0.5, label='Prediction Start')
ax2.set_title("Covariates Provided to Model", fontsize=10)
ax2.set_xlabel("Timestamp")
ax2.legend(loc="upper left")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()