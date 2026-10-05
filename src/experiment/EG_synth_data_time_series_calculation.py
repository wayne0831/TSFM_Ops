## Kernelsynth 기반 시계열 샘플링 및 STL-EMD 3차원 분해 지표 산출 

import os
from time import time
from src.config import *
from src.util.data_analysis import *
from src.util.data_generation import *

start_time = time.time()
np.random.seed(42)

num_samples_per_case = int(PARAMS[DATA_GEN_METHOD]["NUM_SAMPLES"])
raw_lengths = str(PARAMS[DATA_GEN_METHOD]["LENGTH"])
lengths = [int(l.strip()) for l in raw_lengths.split(",") if l.strip()]

stl_emd_path = RES_PATH['stl_emd']['SynthData']
data_path = RES_PATH['data_generation'][DATA_GEN_METHOD]['Data']

print(f'STL_EMD 결과 경로: {stl_emd_path}')
print(f'가상 데이터 저장 경로: {data_path}')

stl_emd_dir = os.path.dirname(stl_emd_path)
data_dir = os.path.dirname(data_path)
data_name = os.path.splitext(os.path.basename(data_path))[0]

os.makedirs(stl_emd_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

cases = get_all_kernel_cases()

print(f"🚀 [KernelSynth] 4대 커널 기반 전수 케이스({len(cases)}개) 데이터 생성 시작")
print(f"   - 타겟 길이 목록: {lengths} (총 {len(lengths)}개 길이)")
print(f"   - 정규화 방식: Standard Normalization (Z-score Normalization)")
print(f"   - 케이스당 샘플 수: {num_samples_per_case}개 (총 {len(cases) * num_samples_per_case * len(lengths)}개 샘플)")

all_records = []

for length in lengths:
    bank = build_kernel_bank(length=length)
    available_periods = [k.p for k in bank["PER"]]
    print(f"\n▶ Current Sequence Length: {length} (허용 유효 주기: {available_periods})")
    
    ts_list_for_len = []
    
    for case in cases:
        c_id = case["Case_ID"]
        k_keys = case["Keys"]
        ops = case["Ops"]
        sec, pat = assign_sector_label(kernels_used=k_keys)
        
        for s_idx in range(num_samples_per_case):
            # 1. GP 시계열 샘플링
            ts, expr = sample_from_case(case, bank, length=length)
            
            # 2. ★ Standard Normalization (Z-score 표준화) 적용
            ts_scaled = (ts - np.mean(ts)) / (np.std(ts) + 1e-9)
            ts_list_for_len.append(ts_scaled)
            
            # 3. STL-EMD 3차원 분해 지표 산출
            ft_stl, fs_stl, fr_stl, ft_stl_emd, fs_stl_emd, fi_stl_emd = calculate_time_series_strength(ts_scaled)
            
            # 4. 통합 레코드 적재
            record = {
                "case_id": c_id,
                "true_sector": sec,
                "pattern": pat,
                "length": length,
                "num_kernels": case["Num_Kernels"],
                "kernels_used": str(k_keys),
                "operations": str(ops) if ops else "['None']",
                "kernel_expression": expr,
                "f_t_stl": ft_stl,
                "f_s_stl": fs_stl,
                "f_r_stl": fr_stl,
                "f_t_stl_emd": ft_stl_emd,
                "f_s_stl_emd": fs_stl_emd,
                "f_i_stl_emd": fi_stl_emd
            }
            all_records.append(record)
    
    # [길이별 NPY 저장] (Num_samples, length) float32 행렬
    ts_array = np.vstack(ts_list_for_len)
    cur_npy_path = os.path.join(data_dir, f"{data_name}_len{length}.npy")
    np.save(cur_npy_path, ts_array)
    print(f"   💾 [NPY 저장 완료] {cur_npy_path} (Shape: {ts_array.shape})")

# [통합 CSV 단일 저장] 모든 LENGTH의 메타데이터를 하나의 CSV로 저장
df_all = pd.DataFrame(all_records)
df_all.to_csv(stl_emd_path, index=False)

elapsed = (time.time() - start_time) / 60
print(f"\n" + "=" * 70)
print(f"✅ 전체 데이터 생성 및 저장 완료")
print(f"📄 [STL-EMD 결과 데이터] {stl_emd_path} (총 {len(df_all)}행)")
for l_val, count in df_all['Length'].value_counts().sort_index().items():
    print(f"   - Length {l_val}: {count}개 샘플")
print(f"📊 [섹터별 전체 데이터 분포]\n{dict(df_all['True_Sector'].value_counts())}")
print(f"⏱️ 총 소요 시간: {elapsed:.2f}분")
print(f"=" * 70)