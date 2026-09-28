import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.config import *


def plot_3d_partitioned_cube(target_dataset: str = "Etth1"):
    # -------------------------------------------------------------------------
    # 2. 결과 데이터 및 Threshold 설정값 로드
    # -------------------------------------------------------------------------
    result_csv_path = RES_PATH["stl_emd"]["RESULT"]
    if not os.path.exists(result_csv_path):
        raise FileNotFoundError(
            f"결과 CSV 파일을 찾을 수 없습니다: {result_csv_path}\n"
            "calculate_time_series_strength.py를 먼저 실행해 주세요."
        )

    df_all = pd.read_csv(result_csv_path)

    # 타겟 데이터셋 필터링
    df = df_all[df_all["DATA"] == target_dataset].copy()
    if df.empty:
        available_datasets = df_all["DATA"].unique().tolist()
        raise ValueError(
            f"'{target_dataset}' 데이터가 CSV에 없습니다. "
            f"가능한 데이터셋 목록: {available_datasets}"
        )

    # config.py 임곗값 및 LENGTH 파싱
    f_t_thr = float(PARAMS["STL_EMD"]["F_T_thr"])
    f_s_thr = float(PARAMS["STL_EMD"]["F_S_thr"])
    f_i_thr = float(PARAMS["STL_EMD"]["F_I_thr"])

    length_order = [
        int(l.strip())
        for l in PARAMS["STL_EMD"]["LENGTH"].split(",")
        if l.strip()
    ]

    # -------------------------------------------------------------------------
    # 3. 분할 평면(Surface Plane) 격자 데이터 생성[cite: 4]
    # -------------------------------------------------------------------------
    axis_min, axis_max = -0.05, 1.05
    grid_resolution = 10
    u = np.linspace(axis_min, axis_max, grid_resolution)
    v = np.linspace(axis_min, axis_max, grid_resolution)
    U, V = np.meshgrid(u, v)

    fig = go.Figure()

    # -------------------------------------------------------------------------
    # 4. 데이터 포인트 트레이스 추가 (LENGTH별 범례 토글 지원)[cite: 4]
    # -------------------------------------------------------------------------
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color_map = {
        l: palette[i % len(palette)] for i, l in enumerate(length_order)
    }

    for l in length_order:
        sub_df = df[df["LENGTH"] == l]
        if sub_df.empty:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=sub_df["F_T_STL"],
                y=sub_df["F_S_STL"],
                z=sub_df["F_I_STL_EMD"],
                mode="markers",
                name=f"Length {l} (N={len(sub_df)})",
                marker=dict(
                    size=5,
                    color=color_map[l],
                    opacity=0.85,
                    line=dict(width=0.3, color="rgba(255,255,255,0.8)"),
                ),
                hovertemplate=(
                    f"<b>{target_dataset}</b> (L={l})<br>"
                    + "F_T_STL: %{x:.3f}<br>"
                    + "F_S_STL: %{y:.3f}<br>"
                    + "F_I_STL_EMD: %{z:.3f}<extra></extra>"
                ),
            )
        )

    # -------------------------------------------------------------------------
    # 5. 3차원 공간 분할 평면(Surface Planes) 주입
    # -------------------------------------------------------------------------
    # (1) F_T Trend 분할 수직 평면 (X = F_T_thr)
    fig.add_trace(
        go.Surface(
            x=np.full_like(U, f_t_thr),
            y=U,
            z=V,
            colorscale=[
                [0, "rgba(214, 48, 49, 0.15)"],
                [1, "rgba(214, 48, 49, 0.15)"],
            ],
            showscale=False,
            name=f"F_T Plane ({f_t_thr})",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # (2) F_S Seasonal 분할 수직 평면 (Y = F_S_thr)
    fig.add_trace(
        go.Surface(
            x=U,
            y=np.full_like(V, f_s_thr),
            z=V,
            colorscale=[
                [0, "rgba(9, 132, 227, 0.15)"],
                [1, "rgba(9, 132, 227, 0.15)"],
            ],
            showscale=False,
            name=f"F_S Plane ({f_s_thr})",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # (3) F_I Intervention 분할 수평 평면 (Z = F_I_thr)
    fig.add_trace(
        go.Surface(
            x=U,
            y=V,
            z=np.full_like(U, f_i_thr),
            colorscale=[
                [0, "rgba(254, 202, 87, 0.25)"],
                [1, "rgba(254, 202, 87, 0.25)"],
            ],
            showscale=False,
            name=f"F_I Plane ({f_i_thr})",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # -------------------------------------------------------------------------
    # 6. 레이아웃 및 3차원 큐브 뷰포트 구성
    # -------------------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=f"<b>3D Partitioned Feature Space Cube: {target_dataset}</b>",
            x=0.5,
            font=dict(size=16),
        ),
        template="plotly_white",
        #width=1000,
        autosize=True,  # 브라우저 창 크기에 맞춰 자동 확장
        height=850,
        scene=dict(
            xaxis=dict(
                title="<b>Trend Strength (F_T)</b>",
                range=[axis_min, axis_max],
                gridcolor="rgba(0,0,0,0.1)",
            ),
            yaxis=dict(
                title="<b>Seasonal Strength (F_S)</b>",
                range=[axis_min, axis_max],
                gridcolor="rgba(0,0,0,0.1)",
            ),
            zaxis=dict(
                title="<b>Intervention Strength (F_I)</b>",
                range=[axis_min, axis_max],
                gridcolor="rgba(0,0,0,0.1)",
            ),
            camera=dict(
                eye=dict(x=1.65, y=1.65, z=1.3)  # 기본 대각 조망 뷰[cite: 2, 4]
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bordercolor="gray",
            borderwidth=1,
        ),
    )

    # HTML 파일 저장 및 웹 브라우저 렌더링[cite: 2, 4]
    save_dir = Path("./results/plot/stl_emd")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_html = save_dir / f"{DATE}_STL_EMD_3D_Partitioned_Cube_{target_dataset}.html"
    fig.write_html(str(save_html))
    print(
        f"[{target_dataset}] 3D 인터랙티브 큐브 저장 완료 -> {save_html}"
    )

    #fig.show(renderer="browser")


if __name__ == "__main__":
    # 원하는 데이터셋명을 지정하여 실행
    # (예: 'Etth1', 'Etth2', 'Ettm1', 'Ettm2', 'Electricity', 'Exchange', 'Solar', 'Weather')

    for dataset in ["Etth1", "Etth2", "Ettm1", "Ettm2", "Electricity", "Exchange", "Solar", "Weather"]:
        TARGET_DATASET = dataset
        plot_3d_partitioned_cube(target_dataset=TARGET_DATASET)