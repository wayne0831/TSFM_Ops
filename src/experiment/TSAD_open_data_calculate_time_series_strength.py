## Open data 활용 STL-EMD 3차원 분해 지표 산출 


# config.py의 DATA 변수 활용해 데이터셋 load

# config.py의 STL_EMD 변수의 LENGTH  활용해 시계열 강도 계산할 시퀀스 길이 설정
# 시퀀스 간에는 LENGTH/2 만큼 겹치도록 설정

# 시퀀스별로 min-max normalization 수행 후 calculate_time_series_strength 함수에 입력
# data_analysis.py의 calculate_time_series_strength 함수 사용해 시퀀스별로 F_T_STL, F_S_STL, F_R_STL, F_T_STL_EMD, F_S_STL_EMD, F_I_STL_EMD 계산

# 결과 dataframe 생성: config.py의 RES_PATH['stl_emd']['RESULT'] 경로에 저장
# dataframe columns: DATA, LENGTH, STEP_SIZE, F_T_STL, F_S_STL, F_R_STL, F_T_STL_EMD, F_S_STL_EMD, F_I_STL_EMD

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from src.config import *
from src.util.data_analysis import calculate_time_series_strength

# 1. 설정값 파싱
data_list = [d.strip() for d in DATA.split(",") if d.strip()]
lengths = [int(l.strip()) for l in PARAMS["STL_EMD"]["LENGTH"].split(",") if l.strip()]

records = []

start_time = time.time()

# 2. 데이터셋별 순회
for data_name in data_list:
    print(f"\n🔍 Processing Dataset: {data_name}")
    if data_name not in DATA_PATH:
        print(
            f"[Warning] DATA_PATH에 '{data_name}' 경로가 존재하지 않아 건너뜁니다."
        )
        continue

    file_path = DATA_PATH[data_name]
    target_col = DATASET[data_name]["target_col"]

    if not os.path.exists(file_path):
        print(
            f"[Warning] 파일이 존재하지 않습니다: {file_path}. 건너뜁니다."
        )
        continue

    print(f"\nProcessing Dataset: {data_name} (Column: {target_col})")
    df_raw = pd.read_csv(file_path)

    if target_col not in df_raw.columns:
        print(
            f"[Error] '{data_name}'에 '{target_col}' 컬럼이 없습니다. 건너뜁니다."
        )
        continue

    series_vals = df_raw[target_col].to_numpy(dtype=np.float64)
    total_len = len(series_vals)

    # 3. 시퀀스 길이별 순회
    for length in lengths:
        step_size = length // 2  # LENGTH / 2 만큼 겹치도록 이동 간격 설정

        if total_len < length:
            print(
                f"  - Length {length}: 데이터 길이({total_len})가 설정 길이보다 짧아 제외합니다."
            )
            continue

        # 슬라이딩 윈도우 인덱스 목록
        start_indices = list(range(0, total_len - length + 1, step_size))
        print(
            f"  - Length: {length} | Step Size: {step_size} | Total Windows: {len(start_indices)}"
        )

        # 4. 각 윈도우별 정규화 및 강도 지표 산출
        for start_idx in tqdm(
            start_indices, desc=f"  [{data_name} | L={length}]", leave=False
        ):
            window = series_vals[start_idx : start_idx + length]

            # standard scaling 적용
            norm_window = (window - np.mean(window)) / (np.std(window) + 1e-9)

            print(
                f"    - Window [{start_idx}:{start_idx + length}] | "
                f"Mean: {np.mean(norm_window):.4f}, Std: {np.std(norm_window):.4f}"
            )

            # STL-EMD 지표 산출
            (
                f_t_stl,
                f_s_stl,
                f_r_stl,
                f_t_stl_emd,
                f_s_stl_emd,
                f_i_stl_emd,
            ) = calculate_time_series_strength(norm_window)

            records.append({
                "Data": data_name,
                "Length": length,
                'Start_Idx': start_idx,
                'End_Idx': start_idx + length,
                "Step_Size": step_size,
                "F_T_STL": f_t_stl,
                "F_S_STL": f_s_stl,
                "F_R_STL": f_r_stl,
                "F_T_STL_EMD": f_t_stl_emd,
                "F_S_STL_EMD": f_s_stl_emd,
                "F_I_STL_EMD": f_i_stl_emd,
            })

end_time = time.time()
elapsed_time = end_time - start_time

# 5. 결과 DataFrame 생성 및 CSV 저장
df_result = pd.DataFrame(records)

save_path = RES_PATH["stl_emd"]["OpenData"]
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df_result.to_csv(save_path, index=False)

print(f"\n분석 완료! 결과가 성공적으로 저장되었습니다 -> {save_path}")
print(f"총 추출된 시퀀스 레코드 수: {len(df_result):,}건")
print(f"총 소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")