import numpy as np
import plotly.graph_objects as go
import pandas as pd
from src.config import *

file_path = r'C:\Users\AICT\Desktop\PythonProject\TSFM_Ops\results\data_generation\260908_KernelSynth_Num300_len64.csv'

print(f"Reading experimental results from: {file_path}")

df_exp_results = pd.read_csv(file_path)

# 1. 임곗값(Threshold) 상수 정의
F_T_thr = PARAMS['TS_STRENGTH']['F_T_thr']
F_S_thr = PARAMS['TS_STRENGTH']['F_S_thr']
F_I_thr = PARAMS['TS_STRENGTH']['F_I_thr']

# 축의 전체 렌더링 범위 정의
axis_min, axis_max = -0.05, 1.05

# 2. 평면(Plane) 격자 데이터 생성을 위한 좌표 배열 준비
grid_resolution = 10
u = np.linspace(axis_min, axis_max, grid_resolution)
v = np.linspace(axis_min, axis_max, grid_resolution)
U, V = np.meshgrid(u, v)

# 3. 고유 시각 세팅 테이블 선언 (Solid 마커 규격)
pattern_configs = {
    'Composite':   {'color': '#d63031', 'symbol': 'circle', 'label': 'S1: Composite'},
    'Seasonal':    {'color': '#0984e3', 'symbol': 'circle', 'label': 'S2: Seasonal'},
    'Stationary':  {'color': '#009432', 'symbol': 'circle', 'label': 'S3: Stationary'},
    'Trending':    {'color': '#8e44ad', 'symbol': 'circle', 'label': 'S4: Trending'},
}

# 범례 고정 정렬 순서
legend_order = [
    'Composite', 'Seasonal', 'Stationary', 'Trending'
]

# 4. Plotly 3D Figure 객체 초기화
fig = go.Figure() 

# 5. 데이터 트레이스(Trace) 주입 (모든 패턴을 기본 True로 설정)
for pattern_label in legend_order:
    
    if pattern_label not in df_exp_results['Pattern'].unique():
        continue
        
    group = df_exp_results[df_exp_results['Pattern'] == pattern_label]
    config = pattern_configs[pattern_label]

    # 
    fig.add_trace(go.Scatter3d(
        x=group['F_T_STL'],
        y=group['F_S_STL'],
        z=group['F_I_STL_EMD'], # F_R_STL, F_I_STL_EMD 등 다양한 z축 피처를 선택 가능
        mode='markers',
        name=config['label'],
        # [핵심] visible=True를 명시하여 모든 Regime이 처음 실행 시 즉각 보이도록 설정합니다.
        # 이 상태에서 범례를 클릭하면 Plotly 엔진이 알아서 'invisible'로 토글합니다.
        visible=True, 
        marker=dict(
            size=6,
            color=config['color'],
            symbol=config['symbol'],
            opacity=0.90,
            # 흰색 테두리 중첩으로 하얗게 타는 현상을 방지하기 위해 테두리를 제거하거나 최소화
            line=dict(width=0, color='rgba(0,0,0,0)') 
        ),
        text=group['Pattern'],
        hovertemplate=(
            "<b>Pattern: %{text}</b><br>" +
            "F_T_STL: %{x:.3f}<br>" +
            "F_S_STL: %{y:.3f}<br>" +
            "F_I_STL_EMD: %{z:.3f}<extra></extra>"
        )
    ))

# 6. 임곗값 기준 3D 평면(Surface Planes) 추가
# (1) F_T Threshold Plane (X = 0.8)
fig.add_trace(go.Surface(
    x=np.full_like(U, F_T_thr), y=U, z=V,
    colorscale=[[0, 'rgba(128,128,128,0.10)'], [1, 'rgba(128,128,128,0.10)']],
    showscale=False,
    name=f'F_T Gate Plane ({F_T_thr})',
    showlegend=True
))

# (2) F_S Threshold Plane (Y = 0.64)
fig.add_trace(go.Surface(
    x=U, y=np.full_like(V, F_S_thr), z=V,
    colorscale=[[0, 'rgba(128,128,128,0.10)'], [1, 'rgba(128,128,128,0.10)']],
    showscale=False,
    name=f'F_S Gate Plane ({F_S_thr})',
    showlegend=True
))

# (3) F_I Threshold Plane (Z = 0.45)
fig.add_trace(go.Surface(
    x=U, y=V, z=np.full_like(U, F_I_thr),
    colorscale=[[0, 'rgba(128,128,128,0.20)'], [1, 'rgba(128,128,128,0.20)']],
    showscale=False,
    name=f'F_I Gate Plane ({F_I_thr})',
    showlegend=True
))

# 7. 레이아웃 뷰포트 정돈 및 테마 구성
fig.update_layout(
    #title='<b>Interactive 3D Feature Space: Calibrated 3D Threshold Planes</b>',
    #title_x=0.5,
    #margin=dict(l=0, r=0, b=0, t=60),
    template='plotly_white',
    scene=dict(
        xaxis=dict(title='Trend Strength (F_T_STL)', range=[axis_min, axis_max], gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(title='Seasonal Strength (F_S_STL)', range=[axis_min, axis_max], gridcolor='rgba(0,0,0,0.1)'),
        zaxis=dict(title='Intervention Strength (F_I_STL_EMD)', range=[axis_min, axis_max], gridcolor='rgba(0,0,0,0.1)'),
        camera=dict(
            eye=dict(x=1.6, y=1.6, z=1.3)
        )
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5,
        font=dict(size=10.5),
        bordercolor="gray",
        borderwidth=1
    )
)

# 8. 기본 웹 브라우저 창으로 화면 강제 출력
fig.show(renderer="browser")

######################################################################################

import pandas as pd
import numpy as np

# 1. 임곗값(Threshold) 상수 정의
F_T_thr = 0.80 # 0.64
F_S_thr = 0.64 # 0.64
F_I_thr = 0.45 # 0.45

# 2. Regime 한글 명칭 정의 (출력용)
sector_names = {
    'S1': 'Composite (복합 패턴 영역)',
    'S2': 'Seasonal (계절성 패턴 영역)',
    'S3': 'Stationary (정상성 노이즈 영역)',
    'S4': 'Trending (추세성 패턴 영역)'
}

# 변수 초기화
total_success = 0

print("=" * 60)
print("  [3D Feature Space Regime Mapping Verification Result]  ")
print("=" * 60)

# 3. 각 True Regime 그룹별 매핑 루프 돌기
for r_label in ['S1', 'S2', 'S3', 'S4']:
    # True_Sector 기준으로 서브 데이터프레임 분리
    sub_df = df_exp_results[df_exp_results['True_Sector'] == r_label]
    
    if len(sub_df) == 0:
        continue

    # [핵심] 사용자가 지정한 3차원 공간 게이트 분할 조건문 구현
    # 데이터셋의 실제 Intervention 강도 컬럼명은 'F_I_STL_EMD'입니다.
    if r_label == 'S1':
        success_condition = (
            (sub_df['F_T_STL'] >= F_T_thr) & 
            (sub_df['F_S_STL'] >= F_S_thr) & 
            (sub_df['F_I_STL_EMD'] >= F_I_thr)
        )
    elif r_label == 'S2':
        success_condition = (
            (sub_df['F_T_STL'] < F_T_thr) & 
            (sub_df['F_S_STL'] >= F_S_thr) #& 
            #(sub_df['F_I_STL_EMD'] < F_I_thr)
        )
    elif r_label == 'S3':
        success_condition = (
            (sub_df['F_T_STL'] < F_T_thr) & 
            (sub_df['F_S_STL'] < F_S_thr) #& 
            #(sub_df['F_I_STL_EMD'] < F_I_thr)
        )
    elif r_label == 'S4':
        # # S4의 이중 영역 조건 결합: 수평 분할 평면(F_I) 하단 영역 + 상단 영역의 S4 지분
        cond_1 = (sub_df['F_T_STL'] >= F_T_thr) & (sub_df['F_S_STL'] < F_S_thr)
        cond_2 = (sub_df['F_T_STL'] >= F_T_thr) & (sub_df['F_S_STL'] >= F_S_thr) & (sub_df['F_I_STL_EMD'] < F_I_thr)
        success_condition = cond_1 | cond_2
        # success_condition = (
        #     (sub_df['F_T_STL'] >= F_T_thr) & 
        #     (sub_df['F_S_STL'] < F_S_thr) & 
        #     (sub_df['F_I_STL_EMD'] < F_I_thr)
        # )

    # 성공 데이터 개수 및 정확도 산출
    success_count = len(sub_df[success_condition])
    total_success += success_count
    accuracy = (success_count / len(sub_df)) * 100
    
    # 기초 메타데이터 추출
    unique_patterns = list(sub_df['Pattern'].unique())
    
    # 결과 출력
    print(f"▶ {r_label}: {sector_names[r_label]}")
    print(f"  - 포함된 패턴 종류: {unique_patterns} (총 {len(unique_patterns)}개)")
    print(f"  - 평균 특징 강도  : Mean F_T = {sub_df['F_T_STL'].mean():.3f} | Mean F_S = {sub_df['F_S_STL'].mean():.3f} | Mean F_I = {sub_df['F_I_STL_EMD'].mean():.3f}")

    # 주기 매칭 검증 (FFT 주기 추정 성과는 주기 성분이 지支配적인 S1, S2만 출력)
    if r_label in ['S1', 'S2'] and 'True_P' in sub_df.columns and 'Est_P' in sub_df.columns:
        same_p_count = len(sub_df[sub_df['True_P'] == sub_df['Est_P']])
        p_match_rate = (same_p_count / len(sub_df)) * 100
        print(f"  - FFT 주기 추정 정확도: {p_match_rate:.1f}%")
        
    print(f"  - 사분면 매핑 성공률 : [ {success_count} / {len(sub_df)} ] ({accuracy:.1f}%)")
    print("-" * 60)

# 4. 전체 데이터 기준 통합 정확도 출력
print(f"■ 전체 통합 오라클 라우팅 정확도: {(total_success / len(df_exp_results))*100:.2f}%")
print("=" * 60)