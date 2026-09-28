import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.mixture import GaussianMixture
from src.config import *

# =============================================================================
# 2. 4대 임곗값 알고리즘 구현
# =============================================================================


def get_otsu_threshold(data: np.ndarray, num_bins: int = 101) -> float:
    """1) Otsu: 집단 간 분산(Between-Class Variance)을 최대화하는 경계점 탐색"""
    hist, bin_edges = np.histogram(data, bins=num_bins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = data.size

    current_max = 0.0
    threshold = bin_centers[0]
    w_bg = 0.0
    sum_bg = 0.0
    sum_total = np.dot(hist, bin_centers)

    for i in range(num_bins):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += hist[i] * bin_centers[i]
        m_bg = sum_bg / w_bg
        m_fg = (sum_total - sum_bg) / w_fg
        var_between = w_bg * w_fg * ((m_bg - m_fg) ** 2)

        if var_between > current_max:
            current_max = var_between
            threshold = bin_centers[i]

    return float(threshold)


def get_gmm_threshold(data: np.ndarray) -> float:
    """2) 2-컴포넌트 GMM: 두 가우시안의 사후확률이 같아지는 베이지안 결정 경계"""
    data = data[~np.isnan(data)]
    if len(data) < 10 or np.ptp(data) < 0.05:
        return float(np.mean(data))

    X = data.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    # 평균 기준 정렬 (idx[0]: Low, idx[1]: High)
    idx = np.argsort(means)
    m1, m2 = float(means[idx[0]]), float(means[idx[1]])
    s1 = float(np.sqrt(variances[idx[0]]))
    s2 = float(np.sqrt(variances[idx[1]]))
    w1, w2 = float(weights[idx[0]]), float(weights[idx[1]])

    # 0 나눗셈 방지
    s1 = max(s1, 1e-6)
    s2 = max(s2, 1e-6)

    # 두 가우시안 확률밀도 교차 2차방정식: A*x^2 + B*x + C = 0
    A = 1.0 / (2.0 * s1**2) - 1.0 / (2.0 * s2**2)
    B = m2 / (s2**2) - m1 / (s1**2)
    C = (
        m1**2 / (2.0 * s1**2)
        - m2**2 / (2.0 * s2**2)
        - np.log((w1 * s2) / (w2 * s1 + 1e-9) + 1e-9)
    )

    # 1차 방정식으로 퇴화하는 경우
    if abs(A) < 1e-6:
        if abs(B) > 1e-6:
            return float(np.clip(-C / B, 0.0, 1.0))
        return float((m1 + m2) / 2.0)

    # 2차 방정식 판별식
    det = B**2 - 4.0 * A * C
    if det >= 0:
        roots = [
            (-B + np.sqrt(det)) / (2.0 * A),
            (-B - np.sqrt(det)) / (2.0 * A),
        ]
        # 두 평균 m1과 m2 사이에 존재하는 실근 선택
        valid = [r for r in roots if m1 <= r <= m2]
        if valid:
            return float(valid[0])

        # 구간 내 실근이 없는 경우 m1, m2와 가장 가까운 실근 클리핑
        closest_root = min(roots, key=lambda r: min(abs(r - m1), abs(r - m2)))
        return float(np.clip(closest_root, 0.0, 1.0))

    # 판별식이 음수일 경우 표준편차 가중 중간값 반환
    return float(np.average([m1, m2], weights=[s2, s1]))


def get_kde_valley_threshold(data: np.ndarray) -> float:
    """3) KDE Valley: 확률밀도함수의 두 봉우리 사이 국소 극소점(골짜기) 탐색"""
    kde = gaussian_kde(data, bw_method="silverman")
    x_grid = np.linspace(0.0, 1.0, 101) # 100개의 균등한 점 생성
    density = kde(x_grid)

    peaks, _ = find_peaks(density, distance=30)

    # 봉우리가 2개 이상일 때: 가장 높은 두 봉우리 사이의 최솟값
    if len(peaks) >= 2:
        top2 = sorted(peaks, key=lambda idx: density[idx], reverse=True)[:2]
        p_left, p_right = sorted(top2)
        valley_idx = p_left + np.argmin(density[p_left:p_right])
        return float(x_grid[valley_idx])

    # 단봉(Unimodal)인 경우: 밀도 함수의 2차 도함수가 0이 되는 변곡점 탐색
    d2 = np.gradient(np.gradient(density))
    inflection_pts = np.where(np.diff(np.sign(d2)))[0]
    if len(inflection_pts) > 0:
        return float(x_grid[inflection_pts[len(inflection_pts) // 2]])

    return float(np.median(data))


def get_kneedle_threshold(data: np.ndarray) -> float:
    """4) Kneedle (Elbow): 정렬 누적 곡선에서 대각 직선과의 수직 거리가 최대가 되는 무릎점"""
    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    x = np.linspace(0.0, 1.0, n)
    y = (sorted_vals - sorted_vals[0]) / (np.ptp(sorted_vals) + 1e-9)

    # (0,0)과 (1,1)을 잇는 대각 직선 대비 수직 편차 곡선
    # 위로 볼록하면 y - x, 아래로 볼록하면 x - y
    diff = np.abs(y - x)
    knee_idx = np.argmax(diff)
    return float(sorted_vals[knee_idx])


# =============================================================================
# 3. 종합 시각화 함수
# =============================================================================


def plot_all_threshold_methods(target_dataset: str = "Etth1", length: int = 512):
    result_csv = RES_PATH["stl_emd"]["RESULT"]
    df = pd.read_csv(result_csv)

    if target_dataset != "ALL":
        df = df[df["DATA"] == target_dataset].copy()
        df = df[df["LENGTH"] == length].copy()
        if df.empty:
            raise ValueError(f"데이터셋 '{target_dataset}'을 찾을 수 없습니다.")

    # 타겟 3개 지표
    metrics_info = [
        (
            "F_T_STL",
            "Trend Strength ($F_T$)",
            float(PARAMS["STL_EMD"]["F_T_thr"]),
        ),
        (
            "F_S_STL",
            "Seasonal Strength ($F_S$)",
            float(PARAMS["STL_EMD"]["F_S_thr"]),
        ),
        (
            "F_I_STL_EMD",
            "Intervention Strength ($F_I$)",
            float(PARAMS["STL_EMD"]["F_I_thr"]),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    sns.set_theme(style="whitegrid")

    summary_records = []

    for idx, (col_name, display_title, default_thr) in enumerate(metrics_info):
        ax = axes[idx]
        vals = df[col_name].dropna().to_numpy()

        # 4대 알고리즘별 임곗값 산출
        t_otsu = get_otsu_threshold(vals)
        t_gmm = get_gmm_threshold(vals)
        t_kde = get_kde_valley_threshold(vals)
        t_knee = get_kneedle_threshold(vals)

        summary_records.append({
            "Metric": col_name,
            "Config_Default": default_thr,
            "Otsu": round(t_otsu, 3),
            "GMM": round(t_gmm, 3),
            "KDE_Valley": round(t_kde, 3),
            "Kneedle": round(t_knee, 3),
        })

        # 히스토그램 및 확률밀도(KDE) 곡선
        sns.histplot(
            vals,
            bins=40,
            kde=True,
            stat="density",
            color="#95a5a6",
            alpha=0.35,
            edgecolor="white",
            ax=ax,
        )

        # 4대 기법 임곗값 수직선 표시
        ax.axvline(
            t_otsu,
            color="#e74c3c",
            linestyle="--",
            linewidth=2.0,
            label=f"Otsu ({t_otsu:.2f})",
        )
        ax.axvline(
            t_gmm,
            color="#2ecc71",
            linestyle="-.",
            linewidth=2.0,
            label=f"GMM ({t_gmm:.2f})",
        )
        ax.axvline(
            t_kde,
            color="#3498db",
            linestyle=":",
            linewidth=2.2,
            label=f"KDE Valley ({t_kde:.2f})",
        )
        ax.axvline(
            t_knee,
            color="#9b59b6",
            linestyle=(0, (3, 5)),
            linewidth=2.0,
            label=f"Kneedle ({t_knee:.2f})",
        )

        # config.py 기본 설정값 (참조선)
        ax.axvline(
            default_thr,
            color="#0026ff", # blue로 수정해줘
            linestyle="-", # 실선으로 수정해줘
            linewidth=1.5,
            alpha=0.8,
            label=f"Config Ref ({default_thr})",
        )

        ax.set_title(f"{display_title}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Strength Value (0.0 ~ 1.0)", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

    plt.suptitle(
        f"Comparative Thresholding Analysis across 4 Methods [{target_dataset}]",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    # 결과 이미지 저장
    save_dir = Path("./results/plot/stl_emd")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = (
        save_dir / f"threshold_comparison_4methods_{target_dataset}_LEN{length}.png"
    )
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    print(f"\n[성공] 플롯이 저장되었습니다 -> {save_file}")

    # 콘솔 수치 비교 출력
    print(f"\n=== [{target_dataset}] 4대 기법별 임곗값 수치 비교표 ===")
    df_summary = pd.DataFrame(summary_records)
    print(df_summary.to_string(index=False))

    plt.show()


if __name__ == "__main__":
    # 특정 데이터셋('Etth1', 'Electricity' 등) 또는 전체 통합('ALL')을 지정해 실행
    plot_all_threshold_methods(target_dataset="Electricity", length=64)