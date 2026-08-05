import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# uni2ts 내부 실제 클래스 임포트 (Hook 필터링용)
from uni2ts.module.ts_embed import MultiInSizeLinear
from uni2ts.module.transformer import TransformerEncoder
try:
    from uni2ts.model.moirai.head import MoiraiOutputHead
except ImportError:
    MoiraiOutputHead = None # 버전에 따라 Head 위치가 다를 수 있음

# ==========================================
# 1. Toy 데이터 생성 (Step 1. Data Prep)
# ==========================================
CTX, PDT = 96, 24
dates = pd.date_range(start="2026-01-01", periods=CTX + PDT, freq="h") 
t = np.arange(CTX + PDT)
target_values = 0.05 * t + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.5, size=CTX + PDT)

df = pd.DataFrame({"timestamp": dates, "target": target_values, "item_id": "series_01"}).set_index("timestamp")
ds = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")
_, test_template = split(ds, offset=-PDT)
test_data = test_template.generate_instances(prediction_length=PDT, windows=1)

# ==========================================
# 2. Moirai-1.0 모델 호출
# ==========================================
MODEL_ID = "Salesforce/moirai-1.0-R-base"
module = MoiraiModule.from_pretrained(MODEL_ID)
model = MoiraiForecast(
    module=module,
    prediction_length=PDT, context_length=CTX, patch_size="auto",
    num_samples=100, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
)

# ==========================================
# 3. Moirai 5단계 파이프라인 Hook (Print) 설정
# ==========================================
print("\n" + "="*80)
print("🐾 [Moirai-1.0 Inference Pipeline Print Trace]")
print("="*80)

def footprint_hook(step_name):
    def hook(sub_module, input, output):
        print(f"\n✅ {step_name}")
        print(f"   ├─ Module Class : {sub_module.__class__.__name__}")
        
        # 1. Input Tensor 확인
        if isinstance(input, tuple) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            print(f"   ├─ Input Shape  : {list(input[0].shape)} (ex: [Batch, Variate, Tokens, Dim])")
        
        # 2. Output Tensor / Object 확인
        if isinstance(output, torch.Tensor):
            print(f"   └─ Output Shape : {list(output.shape)}")
        elif hasattr(output, 'sample'): # Distribution 객체 (Step 4)
            print(f"   └─ Output Distr : {type(output).__name__} (Mixture Distribution Parameters)")
    return hook

hooks = []
# 전체 모듈 순회하며 정확한 클래스에 Hook 걸기
for name, sub_mod in module.named_modules():
    
    # [Step 2] Patch Embedding (MultiInSizeLinear 추적)
    if isinstance(sub_mod, MultiInSizeLinear):
        hooks.append(sub_mod.register_forward_hook(footprint_hook("[Step 2] Patch Tokenization & Embedding")))
        
    # [Step 3] Transformer Encoder Backbone
    elif isinstance(sub_mod, TransformerEncoder):
        hooks.append(sub_mod.register_forward_hook(footprint_hook("[Step 3] Masked Transformer Encoder Backbone")))
        
    # [Step 4] Distribution Head (MoiraiOutputHead 또는 모듈 이름에 head가 포함된 최하위 모듈)
    elif (MoiraiOutputHead and isinstance(sub_mod, MoiraiOutputHead)) or \
         ("head" in name and len(list(sub_mod.children())) == 0 and "param_proj" not in name):
        hooks.append(sub_mod.register_forward_hook(footprint_hook("[Step 4] Distribution Output Head")))

# ==========================================
# 4. Predictor 실행 (Inference 트리거)
# ==========================================
predictor = model.create_predictor(batch_size=1)

print("\n🚀 Starting .predict() ...")
forecasts = list(predictor.predict(test_data.input))
forecast = forecasts[0]

# [Step 5] Sampling 결과 Print
print(f"\n✅ [Step 5] Monte Carlo Sampling & Output")
print(f"   ├─ Process      : Sampled {model.num_samples} times from Mixture Distribution")
print(f"   └─ Final Shape  : {list(forecast.samples.shape)} [Num_Samples, Prediction_Horizon]")

# Hook 해제
for h in hooks: h.remove()
print("\n" + "="*80)