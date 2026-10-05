import os   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import warnings

from src.config import *
from src.util.data_analysis import *

warnings.filterwarnings('ignore')

# =============================================================================
# 2. 합성 데이터 시계열 강도 및 2D vs 3D 분리능 평가 함수
# =============================================================================

def evaluate_synt_data_ts_strength(
    df: pd.DataFrame,
    case_id: str = "all",
    length: int = 512,
    ts_thr_method: str = "otsu",
):
  """합성 데이터의 소문자 지표 컬럼을 기반으로 2D(STL) 대비 3D(STL-EMD) 분리능을 평가 및 시각화합니다.

  Args:
      df (pd.DataFrame): 시계열 강도 데이터프레임 (소문자 컬럼 기준)
      case_id (str): 필터링할 Case ID ('all' 또는 특정 번호)
      length (int): 시계열 윈도우 길이 (예: 512)
      ts_thr_method (str): 임곗값 도출 알고리즘 ('otsu', 'gmm', 'kde', 'kneedle')
  """
  # 1. 조건 필터링
  filtered_df = df.copy()

  if case_id != "all":
    if "case_id" in filtered_df.columns:
      filtered_df = filtered_df[filtered_df["case_id"] == case_id]

  if "length" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["length"] == length]

  if filtered_df.empty:
    print(
        f"[Error] 조건에 일치하는 데이터가 없습니다. (case_id={case_id},"
        f" length={length})"
    )
    return None

  # 2. 소문자 전용 지표 컬럼 정의
  col_t_stl = "f_t_stl"
  col_s_stl = "f_s_stl"
  col_t_emd = "f_t_stl_emd"
  col_s_emd = "f_s_stl_emd"
  col_i_emd = "f_i_stl_emd"

  # 필수 컬럼 존재 여부 검증
  required_cols = [col_t_stl, col_s_stl, col_t_emd, col_s_emd, col_i_emd]
  missing_cols = [c for c in required_cols if c not in filtered_df.columns]
  if missing_cols:
    raise KeyError(
        f"데이터프레임에 다음 필수 소문자 컬럼이 누락되었습니다: {missing_cols}"
    )

  # 3. 임곗값 알고리즘 선택 및 자동 산출
  thr_algorithms = {
      "otsu": get_otsu_threshold,
      "gmm": get_gmm_threshold,
      "kde": get_kde_valley_threshold,
      "kneedle": get_kneedle_threshold,
  }

  method_key = ts_thr_method.lower()
  if method_key not in thr_algorithms:
    raise ValueError(
        f"지원하지 않는 알고리즘입니다: {ts_thr_method}. ('otsu', 'gmm', 'kde',"
        " 'kneedle' 중 선택)"
    )

  selected_thr_func = thr_algorithms[method_key]

  # (A) 2D STL 지표 임곗값 도출
  thr_t_stl = selected_thr_func(filtered_df[col_t_stl].dropna().values)
  thr_s_stl = selected_thr_func(filtered_df[col_s_stl].dropna().values)

  # (B) 3D STL-EMD 지표 임곗값 도출 (F_I 포함 완전 자동 산출)
  thr_t_emd = selected_thr_func(filtered_df[col_t_emd].dropna().values)
  thr_s_emd = selected_thr_func(filtered_df[col_s_emd].dropna().values)
  thr_i_emd = selected_thr_func(filtered_df[col_i_emd].dropna().values)

  print(f"\n=======================================================")
  print(f" [알고리즘: {ts_thr_method.upper()}] 도출된 시계열 강도 임곗값")
  print(f"  [2D STL 지표]")
  print(f"   • f_t_stl threshold     : {thr_t_stl:.3f}")
  print(f"   • f_s_stl threshold     : {thr_s_stl:.3f}")
  print(f"  [3D STL-EMD 지표]")
  print(f"   • f_t_stl_emd threshold : {thr_t_emd:.3f}")
  print(f"   • f_s_stl_emd threshold : {thr_s_emd:.3f}")
  print(f"   • f_i_stl_emd threshold : {thr_i_emd:.3f} (Auto-derived Plane)")
  print(f"=======================================================\n")

  # 4. 1x2 Subplots 시각화
  fig = plt.figure(figsize=(18, 8))
  plt.rcParams["font.family"] = "sans-serif"

  # 레짐별 색상 및 마커 팔레트
  color_palette = {
      "R1": "#d63031",  # 복합형 (Composite)
      "R2": "#0984e3",  # 계절성 (Seasonal)
      "R3": "#009432",  # 정상성 (Stationary)
      "R4": "#8e44ad",  # 단조 추세 (Trending)
      "R4_ambig": "#f39c12",  # 2D 평면 모호 구간
  }
  marker_palette = {
      "R1": "o",
      "R2": "s",
      "R3": "^",
      "R4": "D",
      "R4_ambig": "X",
  }

  regime_col = "regime" if "regime" in filtered_df.columns else None
  unique_regimes = filtered_df[regime_col].unique() if regime_col else ["All"]

  # -------------------------------------------------------------------------
  # (A) 좌측 Subplot: 2D 고전 STL 지표 공간 (f_t_stl vs f_s_stl)
  # -------------------------------------------------------------------------
  ax1 = fig.add_subplot(1, 2, 1)

  if regime_col:
    for reg in unique_regimes:
      group = filtered_df[filtered_df[regime_col] == reg]
      key = "R4_ambig" if "ambig" in str(reg).lower() else str(reg)[:2]
      c = color_palette.get(key, "#7f8c8d")
      m = marker_palette.get(key, "o")
      ax1.scatter(
          group[col_t_stl],
          group[col_s_stl],
          c=c,
          marker=m,
          label=f"{reg}",
          alpha=0.75,
          edgecolors="k",
          s=60,
      )
  else:
    ax1.scatter(
        filtered_df[col_t_stl],
        filtered_df[col_s_stl],
        c="#2c3e50",
        alpha=0.7,
        edgecolors="k",
        s=50,
    )

  # 2D STL 임곗값 기준선
  ax1.axvline(
      thr_t_stl,
      color="#e74c3c",
      linestyle="--",
      linewidth=2.0,
      label=f"$F_{{T,STL}}$ Thr ({thr_t_stl:.2f})",
  )
  ax1.axhline(
      thr_s_stl,
      color="#2980b9",
      linestyle="--",
      linewidth=2.0,
      label=f"$F_{{S,STL}}$ Thr ({thr_s_stl:.2f})",
  )

  # 우상단 중첩(Aliasing Zone) 모호 영역 음영 처리
  rect = plt.Rectangle(
      (thr_t_stl, thr_s_stl),
      1.05 - thr_t_stl,
      1.05 - thr_s_stl,
      fill=True,
      color="gray",
      alpha=0.15,
      zorder=0,
  )
  ax1.add_patch(rect)
  ax1.text(
      thr_t_stl + 0.02,
      0.98,
      "Aliasing Zone\n(R1 & High-R4 Overlap)",
      fontsize=10.5,
      fontweight="bold",
      color="#2c3e50",
      va="top",
  )

  ax1.set_xlim(-0.05, 1.05)
  ax1.set_ylim(-0.05, 1.05)
  ax1.set_xlabel("Trend Strength ($F_{T,STL}$)", fontsize=12, fontweight="bold")
  ax1.set_ylabel(
      "Seasonal Strength ($F_{S,STL}$)", fontsize=12, fontweight="bold"
  )
  ax1.set_title(
      f"(A) 2D STL Space ({ts_thr_method.upper()} Boundary)\nAmbiguity in High"
      " $F_T$ & High $F_S$",
      fontsize=13,
      fontweight="bold",
  )
  ax1.grid(True, linestyle=":", alpha=0.6)
  ax1.legend(loc="lower left", framealpha=0.9)

  # -------------------------------------------------------------------------
  # (B) 우측 Subplot: 3D STL-EMD 지표 공간 (f_t_stl_emd vs f_s_stl_emd vs f_i_stl_emd)
  # -------------------------------------------------------------------------
  ax2 = fig.add_subplot(1, 2, 2, projection="3d")

  if regime_col:
    for reg in unique_regimes:
      group = filtered_df[filtered_df[regime_col] == reg]
      key = "R4_ambig" if "ambig" in str(reg).lower() else str(reg)[:2]
      c = color_palette.get(key, "#7f8c8d")
      m = marker_palette.get(key, "o")
      ax2.scatter(
          group[col_t_emd],
          group[col_s_emd],
          group[col_i_emd],
          c=c,
          marker=m,
          label=f"{reg}",
          alpha=0.85,
          edgecolors="white",
          s=55,
      )
  else:
    ax2.scatter(
        filtered_df[col_t_emd],
        filtered_df[col_s_emd],
        filtered_df[col_i_emd],
        c="#2c3e50",
        alpha=0.8,
        edgecolors="white",
        s=50,
    )

  # 동적 임곗값 thr_i_emd 기반 Z축 분할 평면 렌더링
  xx, yy = np.meshgrid(
      np.linspace(-0.05, 1.05, 10), np.linspace(-0.05, 1.05, 10)
  )
  zz = np.full_like(xx, thr_i_emd)
  ax2.plot_surface(xx, yy, zz, alpha=0.25, color="#f1c40f")
  ax2.text2D(
      0.05,
      0.88,
      f"$F_{{I,EMD}}$ Gate Plane = {thr_i_emd:.3f}",
      transform=ax2.transAxes,
      fontsize=11.5,
      fontweight="bold",
      color="#b7950b",
      bbox=dict(
          boxstyle="round,pad=0.3", fc="#fef9e7", ec="#f1c40f", lw=1.2
      ),
  )

  ax2.set_xlim(-0.05, 1.05)
  ax2.set_ylim(-0.05, 1.05)
  ax2.set_zlim(-0.05, 1.05)
  ax2.set_xlabel("$F_{T,EMD}$ (Trend)", fontsize=11, fontweight="bold")
  ax2.set_ylabel("$F_{S,EMD}$ (Seasonal)", fontsize=11, fontweight="bold")
  ax2.set_zlabel("$F_{I,EMD}$ (Intervention)", fontsize=11, fontweight="bold")
  ax2.set_title(
      f"(B) 3D STL-EMD Space ({ts_thr_method.upper()} Boundary)\nLinear"
      " Separation via $F_I$ Gate Plane",
      fontsize=13,
      fontweight="bold",
  )
  ax2.view_init(elev=20, azim=-45)

  plt.tight_layout()
  plt.show()

  return {
      "2D_STL": {"thr_t": thr_t_stl, "thr_s": thr_s_stl},
      "3D_STL_EMD": {
          "thr_t": thr_t_emd,
          "thr_s": thr_s_emd,
          "thr_i": thr_i_emd,
      },
  }


if __name__ == "__main__":
  file_path = "./results/stl_emd/261001_SynthData_Num300_STL_EMD.csv"

  if os.path.exists(file_path):
    df_synth = pd.read_csv(file_path)

    evaluate_synt_data_ts_strength(df=df_synth, case_id=21, length=512, ts_thr_method="kneedle")
  else:
    print(f"[Warning] 파일을 찾을 수 없습니다: {file_path}")