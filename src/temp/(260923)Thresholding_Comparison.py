########################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import beta, gaussian_kde, norm
import seaborn as sns
from sklearn.mixture import GaussianMixture

# -----------------------------------------------------------------------------
# 1. 4대 임곗값 알고리즘 정의
# -----------------------------------------------------------------------------

def get_otsu_threshold(data: np.ndarray, num_bins: int = 120) -> float:
    """Otsu: 클래스 간 분산(Between-Class Variance)을 최대화하는 지점"""
    data = data[~np.isnan(data)]
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
    """2-컴포넌트 GMM: 두 가우시안의 사후확률이 같아지는 베이지안 결정 경계"""
    data = data[~np.isnan(data)]
    if len(data) < 10 or np.ptp(data) < 0.03:
        return float(np.mean(data))

    X = data.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    idx = np.argsort(means)
    m1, m2 = float(means[idx[0]]), float(means[idx[1]])
    s1, s2 = (
        max(float(np.sqrt(variances[idx[0]])), 1e-6),
        max(float(np.sqrt(variances[idx[1]])), 1e-6),
    )
    w1, w2 = float(weights[idx[0]]), float(weights[idx[1]])

    A = 1.0 / (2.0 * s1**2) - 1.0 / (2.0 * s2**2)
    B = m2 / (s2**2) - m1 / (s1**2)
    C = (
        m1**2 / (2.0 * s1**2)
        - m2**2 / (2.0 * s2**2)
        - np.log((w1 * s2) / (w2 * s1 + 1e-9) + 1e-9)
    )

    if abs(A) < 1e-6:
        if abs(B) > 1e-6:
            return float(np.clip(-C / B, 0.0, 1.0))
        return float((m1 + m2) / 2.0)

    det = B**2 - 4.0 * A * C
    if det >= 0:
        roots = [
            (-B + np.sqrt(det)) / (2.0 * A),
            (-B - np.sqrt(det)) / (2.0 * A),
        ]
        valid = [r for r in roots if m1 <= r <= m2]
        if valid:
            return float(valid[0])
        closest_root = min(roots, key=lambda r: min(abs(r - m1), abs(r - m2)))
        return float(np.clip(closest_root, 0.0, 1.0))

    return float(np.average([m1, m2], weights=[s2, s1]))


def get_kde_threshold(data: np.ndarray) -> float:
    """KDE: 이봉형은 골짜기(Valley), 단봉형은 우측 변곡점(Inflection) 탐색"""
    data = data[~np.isnan(data)]
    kde = gaussian_kde(data, bw_method="silverman")
    x_grid = np.linspace(0.0, 1.0, 500)
    density = kde(x_grid)

    peaks, _ = find_peaks(density, distance=30)

    # 1) 이봉형(Bimodal): 두 피크 사이 골짜기 바닥
    if len(peaks) >= 2:
        top2 = sorted(peaks, key=lambda i: density[i], reverse=True)[:2]
        p1, p2 = sorted(top2)
        valley_idx = p1 + np.argmin(density[p1:p2])
        return float(x_grid[valley_idx])

    # 2) 단봉형(Unimodal): 2차 도함수 부호가 바뀌는 변곡점 탐색
    peak_idx = np.argmax(density)
    d2 = np.gradient(np.gradient(density))
    inflections = np.where(np.diff(np.sign(d2)))[0]

    # 피크 기준 우측 꼬리로 진입하는 첫 번째 변곡점 선택
    candidates = [i for i in inflections if i > peak_idx]
    if candidates:
        return float(x_grid[candidates[0]])

    return float(np.median(data))


def get_kneedle_threshold(data: np.ndarray) -> float:
    """Kneedle: 정렬 누적 곡선에서 대각 기준선 대비 최대 편차점(Elbow)"""
    data = data[~np.isnan(data)]
    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    t_seq = np.linspace(0.0, 1.0, n)
    denom = np.ptp(sorted_vals)
    if denom == 0:
        return float(sorted_vals[0])

    y_norm = (sorted_vals - sorted_vals[0]) / (denom + 1e-9)
    diff = np.abs(t_seq - y_norm)
    knee_idx = np.argmax(diff)
    return float(sorted_vals[knee_idx])


# -----------------------------------------------------------------------------
# 2. 4대 Case 데이터셋 생성 및 2x2 서브플롯 렌더링
# -----------------------------------------------------------------------------
def plot_4cases_comparison():
    np.random.seed(42)
    n_samples = 1500

    # Case 1: 이봉형 데이터 (두 개의 정규분포 혼합)
    case1_data = np.clip(
        np.concatenate([
            np.random.normal(0.28, 0.08, n_samples // 2),
            np.random.normal(0.74, 0.08, n_samples // 2),
        ]),
        0.0,
        1.0,
    )

    # Case 2: 단봉형 + 중앙 위치 데이터 (대칭 종형 분포, Beta(5, 5))
    case2_data = np.clip(beta.rvs(a=5.0, b=5.0, size=n_samples), 0.0, 1.0)

    # Case 3: 단봉형 + 어느정도 한쪽에 치우친 데이터 (중간 왜도, Beta(2, 5))
    case3_data = np.clip(beta.rvs(a=2.0, b=5.0, size=n_samples), 0.0, 1.0)

    # Case 4: 단봉형 + 상당히 한쪽에 치우친 데이터 (극단적 왜도/스파이크, Beta(0.6, 7))
    case4_data = np.clip(beta.rvs(a=0.6, b=7.0, size=n_samples), 0.0, 1.0)

    cases_info = [
        (case1_data, "Case 1: Bimodal Distribution"),
        (
            case2_data,
            "Case 2: Unimodal Centered Distribution",
        ),
        (
            case3_data,
            "Case 3: Moderately Skewed Distribution",
        ),
        (
            case4_data,
            "Case 4: Severely Skewed Distribution",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    axes = axes.flatten()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    records = []

    for idx, (data, title) in enumerate(cases_info):
        ax = axes[idx]

        # 4대 알고리즘 임곗값 계산
        t_otsu = get_otsu_threshold(data)
        t_gmm = get_gmm_threshold(data)
        t_kde = get_kde_threshold(data)
        t_knee = get_kneedle_threshold(data)

        records.append({
            "Distribution Case": title.split(" (")[0],
            "Otsu": round(t_otsu, 3),
            "GMM": round(t_gmm, 3),
            "KDE": round(t_kde, 3),
            "Kneedle": round(t_knee, 3),
        })

        # 히스토그램 및 밀도 곡선 플롯
        sns.histplot(
            data,
            bins=40,
            stat="density",
            color="#bdc3c7",
            alpha=0.5,
            edgecolor="white",
            ax=ax,
        )
        x_eval = np.linspace(0, 1, 400)
        kde_fit = gaussian_kde(data, bw_method="silverman")
        ax.plot(
            x_eval,
            kde_fit(x_eval),
            color="#2c3e50",
            linewidth=2.0,
            label="Density Curve",
        )

        # 4개 임곗값 수직선 표시
        ax.axvline(
            t_otsu,
            color="#e74c3c",
            linestyle="--",
            linewidth=2.2,
            label=f"Otsu: {t_otsu:.2f}",
        )
        ax.axvline(
            t_gmm,
            color="#2ecc71",
            linestyle="-.",
            linewidth=2.2,
            label=f"GMM: {t_gmm:.2f}",
        )
        ax.axvline(
            t_kde,
            color="#3498db",
            linestyle=":",
            linewidth=2.5,
            label=f"KDE: {t_kde:.2f}",
        )
        ax.axvline(
            t_knee,
            color="#9b59b6",
            linestyle="-",
            linewidth=2.0,
            label=f"Kneedle: {t_knee:.2f}",
        )

        ax.set_title(f"({chr(65+idx)}) {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel(
            "Strength Metric Value ($x$)", fontsize=10, fontweight="bold"
        )
        ax.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
        ax.set_xlim(-0.03, 1.03)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    plt.suptitle(
        "Thresholding Behaviors across 4 Representative Data Distributions",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    print("\n=== 4대 케이스별 임곗값 도출 수치 요약표 ===")
    df_summary = pd.DataFrame(records)
    print(df_summary.to_string(index=False))

    plt.show()


if __name__ == "__main__":
    plot_4cases_comparison()
    
####################################################################################

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy.stats import beta

# # -----------------------------------------------------------------------------
# # 1. 재현 가능한 4대 Case 데이터셋 생성 (이전 실험과 완전 동일)
# # -----------------------------------------------------------------------------
# np.random.seed(42)
# n_samples = 1500

# # Case 1: 이봉형 데이터
# case1_data = np.clip(
#     np.concatenate([
#         np.random.normal(0.28, 0.08, n_samples // 2),
#         np.random.normal(0.74, 0.08, n_samples // 2),
#     ]),
#     0.0,
#     1.0,
# )

# # Case 2: 단봉형 + 중앙 대칭 데이터 (Beta(5, 5))
# case2_data = np.clip(beta.rvs(a=5.0, b=5.0, size=n_samples), 0.0, 1.0)

# # Case 3: 단봉형 + 중간 치우침 데이터 (Beta(2, 5))
# case3_data = np.clip(beta.rvs(a=2.0, b=5.0, size=n_samples), 0.0, 1.0)

# # Case 4: 단봉형 + 극단적 치우침 데이터 (Beta(0.6, 7))
# case4_data = np.clip(beta.rvs(a=0.6, b=7.0, size=n_samples), 0.0, 1.0)

# cases_info = [
#     (case1_data, "Case 1: Bimodal Distribution", 0.65),
#     (case2_data, "Case 2: Unimodal Centered Distribution", 0.30),
#     (case3_data, "Case 3: Moderately Skewed Distribution", 0.47),
#     (case4_data, "Case 4: Severely Skewed Distribution", 0.18),
# ]

# # -----------------------------------------------------------------------------
# # 2. Kneedle 누적 곡선 및 편차 탐색 2x2 서브플롯 렌더링
# # -----------------------------------------------------------------------------
# fig, axes = plt.subplots(2, 2, figsize=(18, 12))
# axes = axes.flatten()
# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["axes.unicode_minus"] = False

# for idx, (data, title, expected_thr) in enumerate(cases_info):
#     ax = axes[idx]

#     # Kneedle 좌표축 연산
#     sorted_vals = np.sort(data)
#     n = len(sorted_vals)
#     t_seq = np.linspace(0.0, 1.0, n)  # X축: 정규화 샘플 순위 (0.0 ~ 1.0)
#     denom = np.ptp(sorted_vals)
#     y_norm = (
#         (sorted_vals - sorted_vals[0]) / (denom + 1e-9)
#         if denom > 0
#         else np.zeros_like(sorted_vals)
#     )  # Y축: 정규화 지표 수치

#     # 대각 기준선(y = t)과의 수직 편차 곡선
#     diff_curve = np.abs(t_seq - y_norm)
#     knee_idx = np.argmax(diff_curve)

#     t_knee = t_seq[knee_idx]  # 최대 편차점의 X축 (누적 순위)
#     y_knee = y_norm[knee_idx]  # 최대 편차점의 Y축 (정규화 수치)
#     thr_val = sorted_vals[knee_idx]  # 실제 임곗값 (원본 지표)
#     max_d = diff_curve[knee_idx]  # 최대 편차 거리

#     # --- [좌측 축: 누적 곡선 및 대각선] ---
#     line_cum = ax.plot(
#         t_seq,
#         y_norm,
#         color="#8e44ad",
#         linewidth=2.5,
#         label="Sorted Cumulative Metric $y(t)$",
#     )
#     line_diag = ax.plot(
#         t_seq,
#         t_seq,
#         color="#7f8c8d",
#         linestyle="--",
#         linewidth=1.8,
#         label="Reference Diagonal ($y = t$)",
#     )

#     # 최대 수직 편차 선(Max Deviation) 표시
#     ax.vlines(
#         t_knee,
#         min(y_knee, t_knee),
#         max(y_knee, t_knee),
#         color="#e74c3c",
#         linewidth=2.5,
#         linestyle="-",
#         label=f"Max Deviation $D_{{\max}} = {max_d:.3f}$",
#     )
#     ax.plot(
#         t_knee,
#         y_knee,
#         marker="o",
#         markersize=9,
#         color="#8e44ad",
#         label=f"Knee Point ($T^* = {thr_val:.2f}$)",
#     )

#     # --- [우측 축 (Twinx): 편차 곡선 D(t)] ---
#     ax_twin = ax.twinx()
#     line_diff = ax_twin.plot(
#         t_seq,
#         diff_curve,
#         color="#e67e22",
#         linestyle="-.",
#         linewidth=2.0,
#         label="Deviation Curve $D(t) = |y(t) - t|$",
#     )
#     ax_twin.plot(
#         t_knee, max_d, marker="^", markersize=9, color="#d35400", zorder=5
#     )
#     ax_twin.set_ylabel(
#         "Deviation $|y(t) - t|$", fontsize=11, fontweight="bold", color="#d35400"
#     )
#     ax_twin.tick_params(axis="y", labelcolor="#d35400")
#     ax_twin.set_ylim(-0.02, max(0.65, np.max(diff_curve) * 1.25))
#     ax_twin.grid(False)

#     # 주석 박스: Knee 포인트와 임곗값 설명
#     text_x = t_knee - 0.35 if t_knee > 0.5 else t_knee + 0.08
#     text_y = y_knee + 0.15 if y_knee < 0.6 else y_knee - 0.25
#     ax.annotate(
#         f"Detected Knee Point:\n"
#         f"• Sample Rank ($t^*$): {t_knee*100:.1f}%\n"
#         f"• Metric Threshold ($T^*$): {thr_val:.2f}\n"
#         f"• Max Deviation ($D_{{\max}}$): {max_d:.3f}",
#         xy=(t_knee, (y_knee + t_knee) / 2.0),
#         xytext=(text_x, text_y),
#         arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.8),
#         fontsize=10,
#         fontweight="bold",
#         bbox=dict(boxstyle="round,pad=0.4", fc="#fdfefe", ec="#8e44ad", lw=1.5),
#     )

#     # 서브플롯 서식 설정
#     ax.set_title(
#         f"({chr(65+idx)}) {title} → Threshold $T^* = {thr_val:.2f}$",
#         fontsize=13,
#         fontweight="bold",
#         pad=10,
#     )
#     ax.set_xlabel(
#         "Normalized Sample Rank ($t \in [0, 1]$)",
#         fontsize=11,
#         fontweight="bold",
#     )
#     ax.set_ylabel(
#         "Normalized Metric Value ($y \in [0, 1]$)",
#         fontsize=11,
#         fontweight="bold",
#         color="#8e44ad",
#     )
#     ax.tick_params(axis="y", labelcolor="#8e44ad")
#     ax.set_xlim(-0.02, 1.02)
#     ax.set_ylim(-0.02, 1.02)
#     ax.grid(True, linestyle=":", alpha=0.5)

#     # 범례 통합 표시 (좌측 축 + 우측 축)
#     lines_left, labels_left = ax.get_legend_handles_labels()
#     lines_right, labels_right = ax_twin.get_legend_handles_labels()
#     ax.legend(
#         lines_left + lines_right,
#         labels_left + labels_right,
#         loc="upper left",
#         fontsize=9,
#         framealpha=0.9,
#     )

# plt.suptitle(
#     "Geometric Mechanics of Kneedle Algorithm across 4 Representative Cases\n"
#     "[Cumulative Sorted Curve vs. Diagonal Deviation]",
#     fontsize=16,
#     fontweight="bold",
#     y=0.99,
# )
# plt.tight_layout()
# plt.show()



####################################################################################


# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.signal import find_peaks
# from scipy.stats import gaussian_kde, norm

# # -----------------------------------------------------------------------------
# # 1. 캔버스 및 학술 장표 스타일 설정
# # -----------------------------------------------------------------------------
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["axes.unicode_minus"] = False

# np.random.seed(42)

# # =============================================================================
# # (A) Otsu 알고리즘: 클래스 간 분산(Between-Class Variance) 최대화
# # =============================================================================
# ax_otsu = axes[0, 0]
# # 이봉형 샘플 데이터 생성
# data_otsu = np.concatenate([
#     np.random.normal(0.32, 0.08, 600),
#     np.random.normal(0.76, 0.09, 600),
# ])
# data_otsu = np.clip(data_otsu, 0.0, 1.0)

# # 히스토그램 및 클래스 간 분산 곡선 산출
# bins = np.linspace(0, 1, 101)
# hist, bin_edges = np.histogram(data_otsu, bins=bins)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
# total = len(data_otsu)

# sigma_b = []
# for t in bin_centers:
#     w0 = np.sum(data_otsu <= t) / total
#     w1 = 1.0 - w0
#     if w0 == 0 or w1 == 0:
#         sigma_b.append(0.0)
#         continue
#     m0 = np.mean(data_otsu[data_otsu <= t])
#     m1 = np.mean(data_otsu[data_otsu > t])
#     sigma_b.append(w0 * w1 * ((m0 - m1) ** 2))

# sigma_b = np.array(sigma_b)
# t_otsu = bin_centers[np.argmax(sigma_b)]

# # 히스토그램 플롯
# ax_otsu.hist(
#     data_otsu,
#     bins=35,
#     density=True,
#     color="#bdc3c7",
#     alpha=0.6,
#     edgecolor="white",
#     label="Strength Distribution",
# )

# # 클래스 간 분산곡선 (Twin Axis)
# ax_otsu_twin = ax_otsu.twinx()
# ax_otsu_twin.plot(
#     bin_centers,
#     sigma_b,
#     color="#c0392b",
#     linewidth=2.5,
#     label="Between-Class Variance $\sigma_B^2(T)$",
# )
# ax_otsu_twin.set_ylabel(
#     "$\sigma_B^2(T)$", fontsize=11, fontweight="bold", color="#c0392b"
# )
# ax_otsu_twin.tick_params(axis="y", labelcolor="#c0392b")
# ax_otsu_twin.grid(False)

# # 최적 임곗값 표시
# ax_otsu.axvline(
#     t_otsu,
#     color="#d63031",
#     linestyle="--",
#     linewidth=2.2,
#     label=f"Otsu $T^*$ ({t_otsu:.2f})",
# )
# ax_otsu_twin.plot(
#     t_otsu,
#     np.max(sigma_b),
#     marker="o",
#     markersize=8,
#     color="#d63031",
#     label="Global Maxima",
# )

# # 영역 음영 표시
# ax_otsu.axvspan(
#     0,
#     t_otsu,
#     color="#3498db",
#     alpha=0.08,
#     label="Class $C_0$ ($\mu_0, \sigma_0^2$)",
# )
# ax_otsu.axvspan(
#     t_otsu,
#     1,
#     color="#e67e22",
#     alpha=0.08,
#     label="Class $C_1$ ($\mu_1, \sigma_1^2$)",
# )

# ax_otsu.set_title(
#     "(A) Otsu's Thresholding: Maximizing Between-Class Variance",
#     fontsize=12,
#     fontweight="bold",
#     pad=10,
# )
# ax_otsu.set_xlabel("Strength Metric Value", fontsize=10, fontweight="bold")
# ax_otsu.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
# ax_otsu.set_xlim(-0.02, 1.02)
# ax_otsu.legend(loc="upper left", fontsize=8.5)

# # =============================================================================
# # (B) 2-컴포넌트 GMM: 베이지안 사후확률 교차 결정 경계
# # =============================================================================
# ax_gmm = axes[0, 1]
# x = np.linspace(0, 1, 500)

# w1, m1, s1 = 0.45, 0.30, 0.08
# w2, m2, s2 = 0.55, 0.75, 0.11

# pdf1 = w1 * norm.pdf(x, m1, s1)
# pdf2 = w2 * norm.pdf(x, m2, s2)
# mixture_pdf = pdf1 + pdf2

# # 두 가우시안 교차점 (x*) 수치해 탐색
# idx_search = (x >= m1) & (x <= m2)
# diff = np.abs(pdf1[idx_search] - pdf2[idx_search])
# t_gmm = x[idx_search][np.argmin(diff)]

# ax_gmm.plot(
#     x,
#     pdf1,
#     color="#2980b9",
#     linestyle="--",
#     linewidth=2,
#     label="Component 1: $w_1 \mathcal{N}(\mu_1, \sigma_1^2)$",
# )
# ax_gmm.plot(
#     x,
#     pdf2,
#     color="#e67e22",
#     linestyle="--",
#     linewidth=2,
#     label="Component 2: $w_2 \mathcal{N}(\mu_2, \sigma_2^2)$",
# )
# ax_gmm.plot(
#     x,
#     mixture_pdf,
#     color="#2c3e50",
#     linewidth=2.5,
#     label="Mixture Density $p(x)$",
# )

# ax_gmm.axvline(
#     t_gmm,
#     color="#27ae60",
#     linestyle="-.",
#     linewidth=2.2,
#     label=f"Bayesian Boundary $x^*$ ({t_gmm:.2f})",
# )
# ax_gmm.plot(
#     t_gmm,
#     pdf1[idx_search][np.argmin(diff)],
#     marker="s",
#     markersize=8,
#     color="#27ae60",
# )

# # 사후확률 영역 주석
# ax_gmm.annotate(
#     "Decision Boundary:\n$P(C_1|x^*) = P(C_2|x^*) = 0.5$",
#     xy=(t_gmm, pdf1[idx_search][np.argmin(diff)]),
#     xytext=(t_gmm - 0.28, 2.2),
#     arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5),
#     fontsize=9.5,
#     fontweight="bold",
#     bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#27ae60"),
# )

# ax_gmm.fill_between(x[x <= t_gmm], 0, pdf1[x <= t_gmm], color="#2980b9", alpha=0.1)
# ax_gmm.fill_between(x[x > t_gmm], 0, pdf2[x > t_gmm], color="#e67e22", alpha=0.1)

# ax_gmm.set_title(
#     "(B) 2-Component GMM: Bayesian Decision Boundary Intersection",
#     fontsize=12,
#     fontweight="bold",
#     pad=10,
# )
# ax_gmm.set_xlabel("Strength Metric Value", fontsize=10, fontweight="bold")
# ax_gmm.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
# ax_gmm.set_xlim(-0.02, 1.02)
# ax_gmm.legend(loc="upper right", fontsize=8.5)

# # =============================================================================
# # (C) KDE Valley: 확률밀도함수의 국소 극소점(골짜기) 탐색
# # =============================================================================
# ax_kde = axes[1, 0]
# kde = gaussian_kde(data_otsu, bw_method=0.25)
# kde_y = kde(x)

# peaks, _ = find_peaks(kde_y, distance=50)
# p1, p2 = sorted(peaks[:2])
# valley_idx = p1 + np.argmin(kde_y[p1:p2])
# t_kde = x[valley_idx]

# ax_kde.hist(
#     data_otsu,
#     bins=35,
#     density=True,
#     color="#bdc3c7",
#     alpha=0.4,
#     edgecolor="white",
# )
# ax_kde.plot(
#     x,
#     kde_y,
#     color="#2980b9",
#     linewidth=2.5,
#     label="Estimated Density $\hat{f}(x)$",
# )

# # Peaks 및 Valley 마킹
# ax_kde.plot(
#     x[p1],
#     kde_y[p1],
#     marker="^",
#     markersize=9,
#     color="#8e44ad",
#     label=f"Mode 1 ({x[p1]:.2f})",
# )
# ax_kde.plot(
#     x[p2],
#     kde_y[p2],
#     marker="^",
#     markersize=9,
#     color="#8e44ad",
#     label=f"Mode 2 ({x[p2]:.2f})",
# )
# ax_kde.axvline(
#     t_kde,
#     color="#3498db",
#     linestyle=":",
#     linewidth=2.5,
#     label=f"KDE Valley $T^*$ ({t_kde:.2f})",
# )
# ax_kde.plot(t_kde, kde_y[valley_idx], marker="v", markersize=9, color="#d63031")

# ax_kde.annotate(
#     "Density Valley Point:\n$f'(x) = 0, \ f''(x) > 0$\n(Physical Data Gap)",
#     xy=(t_kde, kde_y[valley_idx]),
#     xytext=(t_kde - 0.25, 1.4),
#     arrowprops=dict(arrowstyle="->", color="#d63031", lw=1.5),
#     fontsize=9.5,
#     fontweight="bold",
#     bbox=dict(boxstyle="round,pad=0.3", fc="#fdedec", ec="#d63031"),
# )

# ax_kde.set_title(
#     "(C) KDE Valley Search: Local Minimum in Bimodal Density",
#     fontsize=12,
#     fontweight="bold",
#     pad=10,
# )
# ax_kde.set_xlabel("Strength Metric Value", fontsize=10, fontweight="bold")
# ax_kde.set_ylabel("Density", fontsize=10, fontweight="bold")
# ax_kde.set_xlim(-0.02, 1.02)
# ax_kde.legend(loc="upper right", fontsize=8.5)

# # =============================================================================
# # (D) Kneedle (Elbow / Knee): 대각 기준선 대비 최대 수직 편차점
# # =============================================================================
# ax_knee = axes[1, 1]

# # 단봉형/오름차순 누적 곡선 모사 (Beta/Exponential 누적 형태)
# t_seq = np.linspace(0, 1, 300)
# # 아래로 볼록한 누적 곡선 생성
# curve_y = t_seq**2.8
# diag_line = t_seq

# # 대각선과의 수직 거리 차이 D(t)
# diff_curve = np.abs(diag_line - curve_y)
# knee_idx = np.argmax(diff_curve)
# t_knee = t_seq[knee_idx]
# y_knee = curve_y[knee_idx]

# ax_knee.plot(
#     t_seq,
#     curve_y,
#     color="#8e44ad",
#     linewidth=2.5,
#     label="Sorted Cumulative Metric $y(t)$",
# )
# ax_knee.plot(
#     t_seq,
#     diag_line,
#     color="#7f8c8d",
#     linestyle="--",
#     linewidth=1.8,
#     label="Reference Diagonal ($y = t$)",
# )

# # 최대 편차 수직 화살표
# ax_knee.vlines(
#     t_knee,
#     y_knee,
#     t_knee,
#     color="#e74c3c",
#     linewidth=2.2,
#     linestyle="-",
#     label=f"Max Deviation $D_{{\max}}$",
# )
# ax_knee.plot(
#     t_knee,
#     y_knee,
#     marker="o",
#     markersize=8,
#     color="#8e44ad",
#     label=f"Knee Point $T^*$ ({t_knee:.2f})",
# )

# ax_knee.annotate(
#     f"Knee / Elbow Point:\n$\max_t |y(t) - t|$\n(Maximum Curvature)",
#     xy=(t_knee, y_knee),
#     xytext=(t_knee + 0.08, y_knee + 0.32),
#     arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
#     fontsize=9.5,
#     fontweight="bold",
#     bbox=dict(boxstyle="round,pad=0.3", fc="#f5eef8", ec="#8e44ad"),
# )

# ax_knee.set_title(
#     "(D) Kneedle Algorithm: Maximum Distance to Diagonal Line",
#     fontsize=12,
#     fontweight="bold",
#     pad=10,
# )
# ax_knee.set_xlabel(
#     "Normalized Sample Index ($t$)", fontsize=10, fontweight="bold"
# )
# ax_knee.set_ylabel(
#     "Normalized Metric Value ($y$)", fontsize=10, fontweight="bold"
# )
# ax_knee.set_xlim(-0.02, 1.02)
# ax_knee.set_ylim(-0.02, 1.02)
# ax_knee.legend(loc="upper left", fontsize=8.5)

# # -----------------------------------------------------------------------------
# # 2. 레이아웃 정돈 및 렌더링
# # -----------------------------------------------------------------------------
# plt.suptitle(
#     "Mathematical & Geometric Mechanics of 4 Thresholding Methods",
#     fontsize=15,
#     fontweight="bold",
#     y=0.99,
# )
# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.signal import find_peaks
# from scipy.stats import beta, gaussian_kde, norm
# from sklearn.mixture import GaussianMixture

# # -----------------------------------------------------------------------------
# # 1. 단봉형 우측 꼬리 왜도(Right-Skewed Unimodal) 데이터 생성
# # -----------------------------------------------------------------------------
# np.random.seed(42)
# # Beta(2, 6): 0.15~0.2 부근에 단일 피크가 있고 우측으로 긴 꼬리가 늘어진 단봉형 분포
# data_skewed = np.random.beta(a=2.0, b=6.0, size=1500)
# data_skewed = np.clip(data_skewed, 0.0, 1.0)

# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["axes.unicode_minus"] = False

# # =============================================================================
# # (A) Otsu 알고리즘: 단봉형 데이터에서의 집단 간 분산 최대화
# # =============================================================================
# ax_otsu = axes[0, 0]
# bins = np.linspace(0, 1, 101)
# hist, bin_edges = np.histogram(data_skewed, bins=bins)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
# total = len(data_skewed)

# sigma_b = []
# for t in bin_centers:
#     w0 = np.sum(data_skewed <= t) / total
#     w1 = 1.0 - w0
#     if w0 == 0 or w1 == 0:
#         sigma_b.append(0.0)
#         continue
#     m0 = np.mean(data_skewed[data_skewed <= t])
#     m1 = np.mean(data_skewed[data_skewed > t])
#     sigma_b.append(w0 * w1 * ((m0 - m1) ** 2))

# sigma_b = np.array(sigma_b)
# t_otsu = bin_centers[np.argmax(sigma_b)]

# ax_otsu.hist(
#     data_skewed,
#     bins=35,
#     density=True,
#     color="#bdc3c7",
#     alpha=0.6,
#     edgecolor="white",
#     label="Skewed Distribution",
# )

# ax_otsu_twin = ax_otsu.twinx()
# ax_otsu_twin.plot(
#     bin_centers,
#     sigma_b,
#     color="#c0392b",
#     linewidth=2.5,
#     label="Between-Class Variance $\sigma_B^2(T)$",
# )
# ax_otsu_twin.set_ylabel(
#     "$\sigma_B^2(T)$", fontsize=11, fontweight="bold", color="#c0392b"
# )
# ax_otsu_twin.tick_params(axis="y", labelcolor="#c0392b")
# ax_otsu_twin.grid(False)

# ax_otsu.axvline(
#     t_otsu,
#     color="#d63031",
#     linestyle="--",
#     linewidth=2.2,
#     label=f"Otsu $T^*$ ({t_otsu:.2f})",
# )
# ax_otsu_twin.plot(
#     t_otsu,
#     np.max(sigma_b),
#     marker="o",
#     markersize=8,
#     color="#d63031",
#     label="Max Variance",
# )

# ax_otsu.axvspan(
#     0,
#     t_otsu,
#     color="#3498db",
#     alpha=0.08,
#     label="Head Group $C_0$ ($\mu_0, \sigma_0^2$)",
# )
# ax_otsu.axvspan(
#     t_otsu,
#     1,
#     color="#e67e22",
#     alpha=0.08,
#     label="Tail Group $C_1$ ($\mu_1, \sigma_1^2$)",
# )

# ax_otsu.annotate(
#     f"Variance Maximum:\nSplits into Mass ({np.mean(data_skewed<=t_otsu)*100:.0f}%)\nand Tail ({np.mean(data_skewed>t_otsu)*100:.0f}%)",
#     xy=(t_otsu, 1.8),
#     xytext=(t_otsu + 0.12, 2.5),
#     arrowprops=dict(arrowstyle="->", color="#d63031", lw=1.5),
#     fontsize=9.5,
#     fontweight="bold",
#     bbox=dict(boxstyle="round,pad=0.3", fc="#fdedec", ec="#d63031"),
# )

# ax_otsu.set_title(
#     "(A) Otsu's Thresholding: Between-Class Variance on Skewed Data",
#     fontsize=12,
#     fontweight="bold",
#     pad=10,
# )
# ax_otsu.set_xlabel("Strength Metric Value", fontsize=10, fontweight="bold")
# ax_otsu.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
# ax_otsu.set_xlim(-0.02, 1.02)
# ax_otsu.legend(loc="upper right", fontsize=8.5)

# # =============================================================================
# # (B) 2-컴포넌트 GMM: 주 봉우리(Head)와 긴 꼬리(Tail) 분리
# # =============================================================================
# ax_gmm = axes[0, 1]
# x_grid = np.linspace(0, 1, 500)

# X_data = data_skewed.reshape(-1, 1)
# gmm = GaussianMixture(n_components=2, random_state=42)
# gmm.fit(X_data)

# means = gmm.means_.flatten()
# variances = gmm.covariances_.flatten()
# weights = gmm.weights_.flatten()

# idx_sort = np.argsort(means)
# m1, m2 = float(means[idx_sort[0]]), float(means[idx_sort[1]])
# s1, s2 = float(np.sqrt(variances[idx_sort[0]])), float(
#     np.sqrt(variances[idx_sort[1]])
# )
# w1, w2 = float(weights[idx_sort[0]]), float(weights[idx_sort[1]])

# pdf1 = w1 * norm.pdf(x_grid, m1, s1)
# pdf2 = w2 * norm.pdf(x_grid, m2, s2)
# mixture_pdf = pdf1 + pdf2

# # 사후확률 교차점 계산
# A = 1.0 / (2.0 * s1**2) - 1.0 / (2.0 * s2**2)
# B = m2 / (s2**2) - m1 / (s1**2)
# C = (
#     m1**2 / (2.0 * s1**2)
#     - m2**2 / (2.0 * s2**2)
#     - np.log((w1 * s2) / (w2 * s1 + 1e-9) + 1e-9)
# )

# roots = np.roots([A, B, C])
# valid_roots = [r for r in roots if m1 <= r <= 1.0]
# t_gmm = float(valid_roots[0]) if valid_roots else float((m1 + m2) / 2.0)

# ax_gmm.hist(
#     data_skewed,
#     bins=35,
#     density=True,
#     color="#bdc3c7",
#     alpha=0.4,
#     edgecolor="white",
# )
# ax_gmm.plot(
#     x_grid,
#     pdf1,
#     color="#2980b9",
#     linestyle="--",
#     linewidth=2,
#     label="Comp 1 (Main Peak)",
# )
# ax_gmm.plot(
#     x_grid,
#     pdf2,
#     color="#e67e22",
#     linestyle="--",
#     linewidth=2,
#     label="Comp 2 (Long Tail)",
# )
# ax_gmm.plot(
#     x_grid,
#     mixture_pdf,
#     color="#2c3e50",
#     linewidth=2.5,
#     label="Mixture Density $p(x)$",
# )

# ax_gmm.axvline(
#     t_gmm,
#     color="#27ae60",
#     linestyle="-.",
#     linewidth=2.2,
#     label=f"Boundary $x^*$ ({t_gmm:.2f})",
# )
# idx_cross = np.argmin(np.abs(x_grid - t_gmm))
# ax_gmm.plot(
#     t_gmm, mixture_pdf[idx_cross], marker="s", markersize=8, color="#27ae60"
# )

# ax_gmm.annotate(
#     "Decision Boundary:\n$P(\\text{Head}|x^*) = P(\\text{Tail}|x^*)$",
#     xy=(t_gmm, mixture_pdf[idx_cross]),
#     xytext=(t_gmm + 0.12, 1.6),
#     arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5),
#     fontsize=9.5,
#     fontweight="bold",
#     bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#27ae60"),
# )

# ax_gmm.fill_between(
#     x_grid[x_grid <= t_gmm], 0, pdf1[x_grid <= t_gmm], color="#2980b9", alpha=0.1
# )
# ax_gmm.fill_between(
#     x_grid[x_grid > t_gmm], 0, pdf2[x_grid > t_gmm], color="#e67e22", alpha=0.1
# )

# ax_gmm.set_title(
#     "(B) 2-Component GMM: Head vs. Tail Component Decomposition",
#     fontsize=12,
#     fontweight="bold",
#     pad=10,
# )
# ax_gmm.set_xlabel("Strength Metric Value", fontsize=10, fontweight="bold")
# ax_gmm.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
# ax_gmm.set_xlim(-0.02, 1.02)
# ax_gmm.legend(loc="upper right", fontsize=8.5)

# # =============================================================================
# # (C) KDE: 단봉형 데이터의 우측 변곡점(Inflection Point) 탐색
# # =============================================================================
# ax_kde = axes[1, 0]
# kde = gaussian_kde(data_skewed, bw_method="silverman")
# kde_y = kde(x_grid)

# # 피크 탐색
# peak_idx = np.argmax(kde_y)
# x_peak = x_grid[peak_idx]

# # 2차 도함수(곡률 변화) 기반 변곡점 탐색 (f''(x) 부호가 바뀌는 지점)
# d2 = np.gradient(np.gradient(kde_y))
# inflection_indices = np.where(np.diff(np.sign(d2)))[0]
# # 피크 우측에 위치한 첫 번째 변곡점 선택 (급격한 하강에서 완만한 감쇄로 전환되는 지점)
# right_inflections = [i for i in inflection_indices if i > peak_idx]
# inflection_idx = (
#     right_inflections[0]
#     if right_inflections
#     else int(peak_idx + len(kde_y) * 0.25)
# )
# t_kde = x_grid[inflection_idx]

# ax_kde.hist(
#     data_skewed,
#     bins=35,
#     density=True,
#     color="#bdc3c7",
#     alpha=0.4,
#     edgecolor="white",
# )
# ax_kde.plot(
#     x_grid,
#     kde_y,
#     color="#2980b9",
#     linewidth=2.5,
#     label="Estimated Density $\hat{f}(x)$",
# )

# ax_kde.plot(
#     x_peak,
#     kde_y[peak_idx],
#     marker="^",
#     markersize=9,
#     color="#8e44ad",
#     label=f"Single Mode ({x_peak:.2f})",
# )
# ax_kde.axvline(
#     t_kde,
#     color="#3498db",
#     linestyle=":",
#     linewidth=2.5,
#     label=f"Inflection $T^*$ ({t_kde:.2f})",
# )
# ax_kde.plot(
#     t_kde, kde_y[inflection_idx], marker="v", markersize=9, color="#d63031"
# )

# ax_kde.annotate(
#     "Right Inflection Point:\n$f''(x) = 0$\n(Deceleration into Tail)",
#     xy=(t_kde, kde_y[inflection_idx]),
#     xytext=(t_kde + 0.12, 1.8),
#     arrowprops=dict(arrowstyle="->", color="#d63031", lw=1.5),
#     fontsize=9.5,
#     fontweight="bold",
#     bbox=dict(boxstyle="round,pad=0.3", fc="#fdedec", ec="#d63031"),
# )

# ax_kde.set_title(
#     "(C) KDE Inflection Search: Transition Boundary for Unimodal Skew",
#     fontsize=12,
#     fontweight="bold",
#     pad=10,
# )
# ax_kde.set_xlabel("Strength Metric Value", fontsize=10, fontweight="bold")
# ax_kde.set_ylabel("Density", fontsize=10, fontweight="bold")
# ax_kde.set_xlim(-0.02, 1.02)
# ax_kde.legend(loc="upper right", fontsize=8.5)

# # =============================================================================
# # (D) Kneedle: 단봉 누적 곡선에서의 최대 곡률 무릎점(Elbow Point)
# # =============================================================================
# ax_knee = axes[1, 1]

# sorted_vals = np.sort(data_skewed)
# n_samples = len(sorted_vals)
# t_seq = np.linspace(0.0, 1.0, n_samples)
# y_norm = (sorted_vals - sorted_vals[0]) / (np.ptp(sorted_vals) + 1e-9)

# # 대각선 대비 수직 편차
# diff_curve = np.abs(t_seq - y_norm)
# knee_idx = np.argmax(diff_curve)
# t_knee_metric = sorted_vals[knee_idx]  # 원본 지표 축 임곗값
# t_knee_rank = t_seq[knee_idx]  # 정규화 샘플 순위 (X축)
# y_knee_val = y_norm[knee_idx]  # 정규화 지표 값 (Y축)

# ax_knee.plot(
#     t_seq,
#     y_norm,
#     color="#8e44ad",
#     linewidth=2.5,
#     label="Sorted Cumulative Metric $y(t)$",
# )
# ax_knee.plot(
#     t_seq,
#     t_seq,
#     color="#7f8c8d",
#     linestyle="--",
#     linewidth=1.8,
#     label="Reference Diagonal ($y = t$)",
# )

# ax_knee.vlines(
#     t_knee_rank,
#     y_knee_val,
#     t_knee_rank,
#     color="#e74c3c",
#     linewidth=2.2,
#     linestyle="-",
#     label="Max Deviation $D_{\max}$",
# )
# ax_knee.plot(
#     t_knee_rank,
#     y_knee_val,
#     marker="o",
#     markersize=8,
#     color="#8e44ad",
#     label=f"Knee Point $T^*$ ({t_knee_metric:.2f})",
# )

# ax_knee.annotate(
#     f"Knee Point:\nMetric Thr = {t_knee_metric:.2f}\n(Max Curvature Elbow)",
#     xy=(t_knee_rank, y_knee_val),
#     xytext=(t_knee_rank - 0.35, y_knee_val + 0.35),
#     arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
#     fontsize=9.5,
#     fontweight="bold",
#     bbox=dict(boxstyle="round,pad=0.3", fc="#f5eef8", ec="#8e44ad"),
# )

# ax_knee.set_title(
#     "(D) Kneedle Algorithm: Elbow Detection on Unimodal Cumulative Curve",
#     fontsize=12,
#     fontweight="bold",
#     pad=10,
# )
# ax_knee.set_xlabel(
#     "Normalized Sample Rank ($t$)", fontsize=10, fontweight="bold"
# )
# ax_knee.set_ylabel(
#     "Normalized Metric Value ($y$)", fontsize=10, fontweight="bold"
# )
# ax_knee.set_xlim(-0.02, 1.02)
# ax_knee.set_ylim(-0.02, 1.02)
# ax_knee.legend(loc="upper left", fontsize=8.5)

# # -----------------------------------------------------------------------------
# # 2. 레이아웃 정돈 및 출력
# # -----------------------------------------------------------------------------
# plt.suptitle(
#     "Thresholding Behaviors on Right-Skewed Unimodal Distribution",
#     fontsize=15,
#     fontweight="bold",
#     y=0.99,
# )
# plt.tight_layout()
# plt.show()










