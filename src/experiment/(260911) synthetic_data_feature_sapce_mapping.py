import os
import glob
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import *

#print(os.getcwd())

NUM_SAMPLES = PARAMS['KernelSynth']['NUM_SAMPLES']
target_lengths = [64, 128, 256, 512]

# 1. 괄호 우선순위가 반영된 조합 라벨 생성 함수
def make_combo_label(row):
    try:
        k_list = ast.literal_eval(row['Kernels_Used']) if isinstance(row['Kernels_Used'], str) else row['Kernels_Used']
        op_list = ast.literal_eval(row['Operations']) if isinstance(row['Operations'], str) else row['Operations']
    except Exception:
        return f"{row['Kernels_Used']} | {row['Operations']}"
    
    if not op_list or op_list == ['None'] or len(op_list) == 0:
        return k_list[0]
    
    if len(k_list) == 2:
        return f"{k_list[0]} {op_list[0]} {k_list[1]}"
    
    expr = f"({k_list[0]} {op_list[0]} {k_list[1]})"
    for i in range(2, len(k_list)):
        op = op_list[i - 1]
        if i == len(k_list) - 1:
            expr = f"{expr} {op} {k_list[i]}"
        else:
            expr = f"({expr} {op} {k_list[i]})"
    return expr

metric_palette = {
    'Trend ($F_T$)': '#2b5c8f', 
    'Seasonal ($F_S$)': '#2a9d8f', 
    'Intervention ($F_I$)': '#e76f51'
}
metric_labels = {
    'F_T_STL': 'Trend ($F_T$)', 
    'F_S_STL': 'Seasonal ($F_S$)', 
    'F_I_STL_EMD': 'Intervention ($F_I$)'
}

# 2. 캔버스 생성
fig, axes = plt.subplots(2, 2, figsize=(28, 19))
axes_flat = axes.flatten()

base_dir = r'./results/data_generation'

for idx, length in enumerate(target_lengths):
    ax = axes_flat[idx]
    
    exact_path = os.path.join(base_dir, f'260908_KernelSynth_Num{NUM_SAMPLES}_len{length}.csv')
    found_file = None
    if os.path.exists(exact_path):
        found_file = exact_path
    else:
        candidates = glob.glob(os.path.join(base_dir, f'*KernelSynth*Num{NUM_SAMPLES}*len{length}*.csv'))
        if candidates:
            found_file = candidates[0]
            
    if not found_file or not os.path.exists(found_file):
        ax.set_title(f"Sequence Length = {length} (File Not Found)", fontsize=13, fontweight='bold', color='darkred', pad=12)
        ax.text(0.5, 0.5, f"Missing File for L = {length}:\n{os.path.basename(exact_path)}", 
                ha='center', va='center', fontsize=12, fontweight='bold', color='#c0392b')
        ax.set_xticks([])
        ax.set_yticks([])
        continue
        
    df = pd.read_csv(found_file)
    df['Case_ID'] = df['Case_ID'].astype(int)
    df = df[~df['Kernels_Used'].astype(str).str.contains('RQ', na=False)].copy()
    df['Combo_Label'] = df.apply(make_combo_label, axis=1)
    
    ordered_cases = (
        df.sort_values(by='Case_ID')
        [['Case_ID', 'Combo_Label', 'Num_Kernels']]
        .drop_duplicates(subset=['Case_ID'])
        .reset_index(drop=True)
    )
    case_order = ordered_cases['Combo_Label'].tolist()
    
    df_melted = pd.melt(
        df, 
        id_vars=['Combo_Label'], 
        value_vars=['F_T_STL', 'F_S_STL', 'F_I_STL_EMD'], 
        var_name='Metric', 
        value_name='Strength'
    )
    df_melted['Metric'] = df_melted['Metric'].map(metric_labels)
    
    # 박스플롯 렌더링
    sns.boxplot(
        data=df_melted, 
        x='Combo_Label', 
        y='Strength', 
        hue='Metric',
        order=case_order,
        palette=metric_palette,
        width=0.75,
        fliersize=2,
        linewidth=0.7,
        ax=ax
    )
    
    # N=1, 2, 3 구간 경계 및 하단 좌표 계산
    boundaries = []
    stage_spans = []
    for n, group in ordered_cases.groupby('Num_Kernels', sort=False):
        start_idx = group.index[0]
        end_idx = group.index[-1]
        mid_idx = (start_idx + end_idx) / 2.0
        stage_spans.append((start_idx, end_idx, mid_idx, f"# of Kernels = {n}"))
        if start_idx > 0:
            boundaries.append(start_idx - 0.5)
            
    for b in boundaries:
        ax.axvline(b, color='darkred', linestyle='--', linewidth=1.2, alpha=0.5)
        
    # X축 하단 계층형 N 라벨링
    y_line = -0.25
    y_text = -0.29
    for start, end, mid, label in stage_spans:
        ax.plot([start - 0.35, end + 0.35], [y_line, y_line],
                transform=ax.get_xaxis_transform(),
                color='#660000', linewidth=1.3, clip_on=False)
        ax.text(mid, y_text, label,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=10.5, fontweight='bold',
                color='#660000', clip_on=False)
        
    ax.set_title(f"Sequence Length = {length}", 
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Strength (0.0 ~ 1.0)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Kernel & Operation Combination", fontsize=11, fontweight='bold', labelpad=38)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticklabels(case_order, rotation=45, ha='right', fontsize=8.5, fontweight='semibold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    
    # [수정] 모든 서브플롯(L=64, 128, 256, 512)에 범례 개별 적용
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels,
              title="Decomposition Metrics", title_fontsize=10.5, fontsize=9.5, 
              loc='upper right', framealpha=0.9)

plt.suptitle("Comparative Distribution of F_T, F_S, and F_I across Sequence Lengths", 
             fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(hspace=0.58, wspace=0.15, top=0.94, bottom=0.16, left=0.05, right=0.98)

plt.savefig("Comparative_2x2_Length_Distributions.png", dpi=300, bbox_inches='tight')
plt.show()