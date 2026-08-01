import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# ==========================================
# 1. Toy Dataset 생성 (120개 타임스텝)
# ==========================================
dates = pd.date_range(start="2026-01-01", periods=120, freq="h")
t = np.arange(120)

# 추세 + 주기성 + 노이즈 기반 합성 시계열 생성
target_values = 0.05 * t + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.5, size=120)

df = pd.DataFrame({
    "timestamp": dates,
    "target": target_values,
    "item_id": "toy_series_01"
}).set_index("timestamp")


# ==========================================
# 2. GluonTS Dataset 변환 및 분할
# ==========================================
CTX = 96  # Context Window (입력 데이터 길이)
PDT = 24  # Prediction Horizon (예측 미래 길이)

ds = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")
train, test_template = split(ds, offset=-PDT)

test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=1,
)


# ==========================================
# 3. Moirai-1.0-R-base 모델 로드 및 Predictor 생성
# ==========================================
MODEL_ID = "Salesforce/moirai-1.0-R-base"

module = MoiraiModule.from_pretrained(MODEL_ID)
model = MoiraiForecast(
    module=module,
    prediction_length=PDT,
    context_length=CTX,
    patch_size="auto",
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
)

predictor = model.create_predictor(batch_size=16)


# ==========================================
# 4. Zero-Shot 추론 수행
# ==========================================
forecasts = list(predictor.predict(test_data.input))
forecast = forecasts[0]


# ==========================================
# 5. 예측 결과 시각화
# ==========================================
plt.figure(figsize=(12, 5))

past_dates = df.index[:CTX]
future_dates = df.index[CTX:]

# 1. 과거 Context 데이터
plt.plot(past_dates, df["target"].iloc[:CTX], label="Context (Past Data)", color="black", linewidth=1.5)

# 2. 실제 미래 데이터 (Ground Truth)
plt.plot(future_dates, df["target"].iloc[CTX:], label="Ground Truth", color="blue", linestyle="--", linewidth=1.5)

# 3. Moirai 예측 (중앙값 p50 및 80% 예측 구간 p10~p90)
p10 = forecast.quantile(0.1)
p50 = forecast.quantile(0.5)
p90 = forecast.quantile(0.9)

plt.plot(future_dates, p50, label="Moirai Median Forecast (p50)", color="crimson", linewidth=2)
plt.fill_between(future_dates, p10, p90, color="crimson", alpha=0.25, label="80% Prediction Interval (p10~p90)")

plt.title(f"Moirai-1.0-R-base Zero-Shot Forecast (CTX={CTX}, PDT={PDT})", fontsize=12, fontweight="bold")
plt.xlabel("Timestamp", fontsize=10)
plt.ylabel("Target Value", fontsize=10)
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()