###########################################################################################################
# import libraries
###########################################################################################################
import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 프로젝트 루트(TSFM_Ops) 및 src 경로를 sys.path에 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.config import *
from src.model.PEFT.train_forecast import calculate_metrics

###########################################################################################################
# helper functions for path & data loading
###########################################################################################################

_ACTUALS_CACHE: Dict[Tuple[str, float], np.ndarray] = {}


def _resolve_path(rel_path: str) -> str:
    """실행 위치(프로젝트 루트 또는 src 폴더)에 관계없이 경로를 안전하게 해석합니다."""
    if os.path.isabs(rel_path) or os.path.exists(rel_path):
        return rel_path
    root_candidate = os.path.normpath(os.path.join(ROOT_DIR, rel_path))
    if os.path.exists(root_candidate):
        return root_candidate
    src_candidate = os.path.normpath(os.path.join(SRC_DIR, rel_path))
    if os.path.exists(src_candidate):
        return src_candidate
    return root_candidate


def load_test_actuals(data_name: str, ft_ratio: Optional[float] = None) -> np.ndarray:
    """
    원본 데이터셋 CSV에서 target_col을 로드한 뒤, ft_ratio 이후의 테스트 구간 실제값(Ground Truth)을 반환합니다.
    """
    if ft_ratio is None:
        ft_ratio = float(str(PARAMS.get('FT_RATIO', '0.7')).split(',')[0].strip())

    cache_key = (data_name, float(ft_ratio))
    if cache_key in _ACTUALS_CACHE:
        return _ACTUALS_CACHE[cache_key]

    resolved_data_name = data_name
    if resolved_data_name not in DATA_PATH:
        lower_map = {k.lower(): k for k in DATA_PATH.keys()}
        if resolved_data_name.lower() in lower_map:
            resolved_data_name = lower_map[resolved_data_name.lower()]
        else:
            raise KeyError(f"DATA_PATH에서 데이터셋 '{data_name}'을(를) 찾을 수 없습니다.")

    df_path = _resolve_path(DATA_PATH[resolved_data_name])
    tgt_col = DATASET[resolved_data_name]['target_col']
    df_raw = pd.read_csv(df_path)

    target = df_raw[tgt_col].fillna(0).values.astype(np.float32)
    ft_len = int(len(target) * float(ft_ratio))
    actuals = target[ft_len:]

    _ACTUALS_CACHE[cache_key] = actuals
    return actuals


###########################################################################################################
# filename parser for .npy prediction files
###########################################################################################################

# 파일명에서 날짜({DATE})와 본문(body) 분리:
#   - 신규 형식: {DATE}_{body}_preds.npy
#   - 기존 형식: {body}_preds_{DATE}.npy
_PREDS_PREFIX_DATE = re.compile(r"^(?P<date>\d+)_(?P<body>.+?)_preds\.npy$")
_PREDS_SUFFIX_DATE = re.compile(r"^(?P<body>.+?)_preds_(?P<date>\d+)\.npy$")

# 1. MoE 본문 패턴:
# TimesFM_MoE_[STL_EMD]_single_etth1_cl[512]_hl[64]_LoRA_r[8]_a[16]_d[0.1]_tgt_[qkv_proj, out]_lr[0.0001]_e[100]_bs[32]_st[32]
_MOE_PATTERN = re.compile(
    r"^(?P<tsfm>[^_]+)_MoE_\[(?P<decomp>[^\]]+)\]_(?P<inf_m>single|weighted|ensemble)_(?P<data>[^_]+)"
    r"_cl\[(?P<cl>\d+)\]_hl\[(?P<hl>\d+)\]_(?P<ft>[^_]+)"
    r"_r\[(?P<rank>\d+)\]_a\[(?P<alpha>\d+)\]_d\[(?P<dropout>[^\]]+)\]"
    r"_tgt_\[(?P<tgt>[^\]]+)\]_lr\[(?P<lr>[^\]]+)\]_e\[(?P<epochs>\d+)\]_bs\[(?P<bs>\d+)\]"
    r"(?:_st\[(?P<stride>\d+)\])?$"
)

# 2. LoRA 본문 패턴:
# TimesFM_etth1_cl[512]_hl[64]_LoRA_r[8]_a[16]_d[0.1]_tgt_[qkv_proj, out]_lr[0.0001]_e[100]_bs[32]_st[32]
_LORA_PATTERN = re.compile(
    r"^(?P<tsfm>[^_]+)_(?P<data>[^_]+)_cl\[(?P<cl>\d+)\]_hl\[(?P<hl>\d+)\]_(?P<ft>[^_]+)"
    r"_r\[(?P<rank>\d+)\]_a\[(?P<alpha>\d+)\]_d\[(?P<dropout>[^\]]+)\]"
    r"_tgt_\[(?P<tgt>[^\]]+)\]_lr\[(?P<lr>[^\]]+)\]_e\[(?P<epochs>\d+)\]_bs\[(?P<bs>\d+)\]"
    r"(?:_st\[(?P<stride>\d+)\])?$"
)

# 3. TimesFM 단독 본문 패턴:
# TimesFM_etth1_cl[512]_hl[64]
_TSFM_PATTERN = re.compile(
    r"^(?P<tsfm>[^_]+)_(?P<data>[^_]+)_cl\[(?P<cl>\d+)\]_hl\[(?P<hl>\d+)\]$"
)


def parse_prediction_filename(npy_path: str) -> Optional[Dict[str, Any]]:
    """
    예측 .npy 파일 경로에서 모델 종류, 데이터셋, cl, hl, 하이퍼파라미터 등의 메타데이터를 파싱합니다.
    신규 형식('{DATE}_..._preds.npy')과 기존 형식('..._preds_{DATE}.npy')을 모두 지원합니다.
    """
    fname = os.path.basename(npy_path)
    m_date = _PREDS_PREFIX_DATE.match(fname) or _PREDS_SUFFIX_DATE.match(fname)
    if not m_date:
        return None
    date_str = m_date.group('date')
    body = m_date.group('body')

    m_moe = _MOE_PATTERN.match(body)
    if m_moe:
        d = m_moe.groupdict()
        return {
            'category': 'MoE',
            'path': npy_path,
            'filename': fname,
            'stem': os.path.splitext(fname)[0],
            'tsfm_method': d['tsfm'],
            'data_name': d['data'],
            'cl': int(d['cl']),
            'hl': int(d['hl']),
            'decomp_method': d['decomp'],
            'inference_method': d['inf_m'],
            'ft_method': d['ft'],
            'rank': int(d['rank']),
            'alpha': int(d['alpha']),
            'dropout': float(d['dropout']),
            'target_modules': d['tgt'],
            'lr': float(d['lr']),
            'epochs': int(d['epochs']),
            'batch_size': int(d['bs']),
            'stride': int(d['stride']) if d.get('stride') else 32,
            'date': date_str,
            'label': f"{d['tsfm']}-MoE ({d['decomp']}, {d['inf_m']})",
        }

    m_lora = _LORA_PATTERN.match(body)
    if m_lora:
        d = m_lora.groupdict()
        return {
            'category': 'LoRA',
            'path': npy_path,
            'filename': fname,
            'stem': os.path.splitext(fname)[0],
            'tsfm_method': d['tsfm'],
            'data_name': d['data'],
            'cl': int(d['cl']),
            'hl': int(d['hl']),
            'decomp_method': None,
            'inference_method': None,
            'ft_method': d['ft'],
            'rank': int(d['rank']),
            'alpha': int(d['alpha']),
            'dropout': float(d['dropout']),
            'target_modules': d['tgt'],
            'lr': float(d['lr']),
            'epochs': int(d['epochs']),
            'batch_size': int(d['bs']),
            'stride': int(d['stride']) if d.get('stride') else 32,
            'date': date_str,
            'label': f"{d['tsfm']}+{d['ft']} (tgt=[{d['tgt']}])",
        }

    m_tsfm = _TSFM_PATTERN.match(body)
    if m_tsfm:
        d = m_tsfm.groupdict()
        return {
            'category': 'Base',
            'path': npy_path,
            'filename': fname,
            'stem': os.path.splitext(fname)[0],
            'tsfm_method': d['tsfm'],
            'data_name': d['data'],
            'cl': int(d['cl']),
            'hl': int(d['hl']),
            'decomp_method': None,
            'inference_method': None,
            'ft_method': None,
            'rank': None,
            'alpha': None,
            'dropout': None,
            'target_modules': None,
            'lr': None,
            'epochs': None,
            'batch_size': None,
            'stride': None,
            'date': date_str,
            'label': f"{d['tsfm']} (Base)",
        }

    return None


def _find_matching_context_experts_csv(npy_path: str) -> Optional[str]:
    """
    MoE 예측 .npy 파일과 짝을 이루는 context_experts CSV 파일이 존재하면 그 경로를 반환합니다.
    신규 형식('{DATE}_..._context_experts.csv')과 기존 형식('..._context_experts_{DATE}.csv')을 모두 지원합니다.
    """
    if npy_path.endswith("_preds.npy"):
        candidate = npy_path[: -len("_preds.npy")] + "_context_experts.csv"
        if os.path.isfile(candidate):
            return candidate
    candidate = re.sub(r"_preds_(\d+)\.npy$", r"_context_experts_\1.csv", npy_path)
    if candidate != npy_path and os.path.isfile(candidate):
        return candidate
    return None


###########################################################################################################
# plotting functions
###########################################################################################################


def plot_single_prediction(
    npy_path: str,
    save_dir: Optional[str] = None,
    zoom_steps: Optional[int] = None,
) -> Optional[str]:
    """
    단일 예측 .npy 파일을 로드하여 실제 Target(Actual)과 함께 시각화하고 .png로 저장합니다.
    MoE 모델의 경우 저장된 컨텍스트별 F_T, F_S, F_I 및 호출 expert 정보(_context_experts_*.csv)가 있으면
    하단 패널에 함께 시각화합니다.
    """
    meta = parse_prediction_filename(npy_path)
    if meta is None:
        return None

    preds = np.load(npy_path).astype(np.float32).flatten()
    actuals = load_test_actuals(meta['data_name'])

    min_len = min(len(actuals), len(preds))
    actuals = actuals[:min_len]
    preds = preds[:min_len]

    mae, mse, wape, smape, *_ = calculate_metrics(actuals, preds)
    hl = meta['hl']
    if zoom_steps is None:
        zoom_steps = min(min_len, max(hl * 4, 256))

    ctx_csv_path = _find_matching_context_experts_csv(npy_path) if meta['category'] == 'MoE' else None
    has_ctx_info = ctx_csv_path is not None

    n_rows = 3 if has_ctx_info else 2
    fig_height = 11 if has_ctx_info else 8
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, fig_height), constrained_layout=True)

    t_full = np.arange(min_len)

    # 1) 상단 패널: 전체 테스트 구간 Actual vs Prediction
    ax_full = axes[0]
    ax_full.plot(t_full, actuals, label="Actual (Ground Truth)", color="#1f2937", linewidth=1.3, alpha=0.85)
    ax_full.plot(t_full, preds, label=meta['label'], color="#e11d48", linewidth=1.3, alpha=0.9)

    # Horizon 경계선 표시 (너무 촘촘하지 않을 때만)
    if min_len // hl <= 40:
        for b_idx in range(hl, min_len, hl):
            ax_full.axvline(x=b_idx, color="#9ca3af", linestyle="--", linewidth=0.6, alpha=0.5)

    title_str = (
        f"[{meta['data_name']}] {meta['label']} | cl={meta['cl']}, hl={hl} "
        f"| MAE={mae:.4f}, MSE={mse:.4f}, WAPE={wape:.2f}%, sMAPE={smape:.2f}%"
    )
    if meta.get('target_modules'):
        title_str += f" | tgt=[{meta['target_modules']}]"
    ax_full.set_title(title_str, fontsize=11, fontweight="bold")
    ax_full.set_xlabel("Test Time Step (Full Horizon)")
    ax_full.set_ylabel("Value")
    ax_full.legend(loc="upper right", frameon=True)
    ax_full.grid(True, linestyle=":", alpha=0.5)

    # 2) 중단 패널: 마지막 zoom_steps 구간 확대(Zoom-in) 플롯
    ax_zoom = axes[1]
    z_start = max(0, min_len - zoom_steps)
    t_zoom = np.arange(z_start, min_len)
    ax_zoom.plot(t_zoom, actuals[z_start:], label="Actual (Ground Truth)", color="#1f2937", linewidth=1.6)
    ax_zoom.plot(t_zoom, preds[z_start:], label=f"{meta['label']} (Prediction)", color="#2563eb", linewidth=1.6, linestyle="--")

    first_boundary = ((z_start // hl) + 1) * hl
    for b_idx in range(first_boundary, min_len, hl):
        ax_zoom.axvline(x=b_idx, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.6)

    ax_zoom.set_title(f"Zoom-in View (Last {min_len - z_start} Steps: [{z_start}:{min_len}])", fontsize=10, fontweight="bold")
    ax_zoom.set_xlabel("Test Time Step")
    ax_zoom.set_ylabel("Value")
    ax_zoom.legend(loc="upper right", frameon=True)
    ax_zoom.grid(True, linestyle=":", alpha=0.5)

    # 3) 하단 패널 (MoE 컨텍스트 로그가 있는 경우): 컨텍스트별 F_T, F_S, F_I 및 호출된 Expert 시각화
    if has_ctx_info:
        ax_ctx = axes[2]
        ctx_df = pd.read_csv(ctx_csv_path)
        win_indices = ctx_df['window_idx'].values
        ax_ctx.plot(win_indices, ctx_df['F_T'].values, marker="o", markersize=4, label="F_T (Trend)", color="#d97706")
        ax_ctx.plot(win_indices, ctx_df['F_S'].values, marker="s", markersize=4, label="F_S (Seasonality)", color="#059669")
        if 'F_I' in ctx_df.columns and not ctx_df['F_I'].isna().all():
            ax_ctx.plot(win_indices, ctx_df['F_I'].values, marker="^", markersize=4, label="F_I (Intervention)", color="#7c3aed")

        # 윈도우 수가 적당하면 호출된 전문가 라벨 텍스트 표시
        if len(ctx_df) <= 35 and 'called_expert' in ctx_df.columns:
            for _, row in ctx_df.iterrows():
                w_i = int(row['window_idx'])
                exp_str = str(row['called_expert'])
                ax_ctx.annotate(
                    exp_str,
                    (w_i, max(float(row['F_T']), float(row['F_S']))),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    rotation=30,
                )

        ax_ctx.set_ylim(-0.05, 1.15)
        ax_ctx.set_title("Test Context Strengths (F_T, F_S, F_I) & Called Experts per Window", fontsize=10, fontweight="bold")
        ax_ctx.set_xlabel("Test Context Window Index")
        ax_ctx.set_ylabel("Strength [0, 1]")
        ax_ctx.legend(loc="upper right", frameon=True)
        ax_ctx.grid(True, linestyle=":", alpha=0.5)

    if save_dir is None:
        cat_key = meta['tsfm_method'] if meta['category'] == 'Base' else meta['category']
        save_dir = _resolve_path(RES_PATH['plot'].get(cat_key, f"./results/plot/{cat_key}/"))
    os.makedirs(save_dir, exist_ok=True)

    out_stem = meta['stem'] if meta['stem'].startswith(f"{meta['date']}_") else f"{meta['date']}_{meta['stem']}"
    out_png_path = os.path.join(save_dir, f"{out_stem}.png")
    fig.savefig(out_png_path, dpi=180)
    plt.close(fig)
    return out_png_path


def plot_baseline_comparisons(
    parsed_items: List[Dict[str, Any]],
    save_dir: Optional[str] = None,
    zoom_steps: Optional[int] = None,
) -> List[str]:
    """
    동일한 (data_name, cl, hl, date) 조건을 공유하는 TimesFM / LoRA / MoE 예측 결과(.npy)들을 그룹화하여,
    실제 Target(Actual)과 함께 하나의 비교 플롯에 겹쳐 그려 저장합니다.
    """
    saved_paths: List[str] = []

    # (tsfm_method, data_name, cl, hl, date) 별로 파일 분류
    by_setting: Dict[Tuple[str, str, int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in parsed_items:
        key = (item['tsfm_method'], item['data_name'], item['cl'], item['hl'], item['date'])
        by_setting[key].append(item)

    for (tsfm_method, data_name, cl, hl, date_str), group in by_setting.items():
        tsfm_items = [x for x in group if x['category'] not in ('LoRA', 'MoE')]
        lora_items = [x for x in group if x['category'] == 'LoRA']
        moe_items = [x for x in group if x['category'] == 'MoE']

        # 비교할 모델이 최소 2개 이상이거나 MoE 결과가 있을 때 비교 플롯 생성
        if len(group) < 2 and not moe_items:
            continue

        # target_modules 그룹별로 비교 (예: 'qkv_proj, out' vs 'ff0, ff1')
        tgt_groups = sorted({x['target_modules'] for x in (lora_items + moe_items) if x.get('target_modules')})
        if not tgt_groups:
            tgt_groups = [None]

        actuals_full = load_test_actuals(data_name)

        for tgt_mod in tgt_groups:
            candidates: List[Dict[str, Any]] = []
            candidates.extend(tsfm_items[:1])
            if tgt_mod is not None:
                candidates.extend([x for x in lora_items if x['target_modules'] == tgt_mod])
                candidates.extend([x for x in moe_items if x['target_modules'] == tgt_mod])
            else:
                candidates.extend(lora_items)
                candidates.extend(moe_items)

            if not candidates:
                continue

            loaded_series: List[Tuple[str, np.ndarray, Dict[str, Any], Tuple[float, float, float, float]]] = []
            min_len = len(actuals_full)

            for cand in candidates:
                arr = np.load(cand['path']).astype(np.float32).flatten()
                min_len = min(min_len, len(arr))
                loaded_series.append((cand['label'], arr, cand, (0.0, 0.0, 0.0, 0.0)))

            actuals = actuals_full[:min_len]
            updated_series = []
            for lbl, arr, cand, _ in loaded_series:
                arr_cut = arr[:min_len]
                mae, mse, wape, smape, *_ = calculate_metrics(actuals, arr_cut)
                updated_series.append((lbl, arr_cut, cand, (mae, mse, wape, smape)))

            eff_zoom = zoom_steps if zoom_steps is not None else min(min_len, max(hl * 4, 256))
            z_start = max(0, min_len - eff_zoom)

            fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
            t_full = np.arange(min_len)
            t_zoom = np.arange(z_start, min_len)

            palette = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#ca8a04"]

            # 상단: 전체 테스트 구간
            ax_full = axes[0]
            ax_full.plot(t_full, actuals, label="Actual (Ground Truth)", color="#111827", linewidth=1.6, alpha=0.9)

            # 하단: Zoom-in 구간
            ax_zoom = axes[1]
            ax_zoom.plot(t_zoom, actuals[z_start:], label="Actual (Ground Truth)", color="#111827", linewidth=1.8, alpha=0.95)

            for idx_s, (lbl, arr_cut, cand, (mae, mse, wape, smape)) in enumerate(updated_series):
                color = palette[idx_s % len(palette)]
                line_style = "--" if cand['category'] not in ('LoRA', 'MoE') else ("-." if cand['category'] == 'LoRA' else "-")
                legend_lbl = f"{lbl} (MAE={mae:.4f}, MSE={mse:.4f})"

                ax_full.plot(t_full, arr_cut, label=legend_lbl, color=color, linewidth=1.2, linestyle=line_style, alpha=0.85)
                ax_zoom.plot(t_zoom, arr_cut[z_start:], label=legend_lbl, color=color, linewidth=1.5, linestyle=line_style, alpha=0.9)

            tgt_suffix = f" | tgt=[{tgt_mod}]" if tgt_mod else ""
            ax_full.set_title(
                f"[{tsfm_method} Comparison - {data_name}] Full Test Series (cl={cl}, hl={hl}{tgt_suffix}, date={date_str})",
                fontsize=12,
                fontweight="bold",
            )
            ax_full.set_xlabel("Test Time Step")
            ax_full.set_ylabel("Value")
            ax_full.legend(loc="upper right", fontsize=9, frameon=True)
            ax_full.grid(True, linestyle=":", alpha=0.5)

            first_boundary = ((z_start // hl) + 1) * hl
            for b_idx in range(first_boundary, min_len, hl):
                ax_zoom.axvline(x=b_idx, color="#9ca3af", linestyle="--", linewidth=0.7, alpha=0.6)

            ax_zoom.set_title(
                f"[{tsfm_method} Comparison - {data_name}] Zoom-in View (Steps [{z_start}:{min_len}], Vertical dashed lines = hl={hl} boundaries)",
                fontsize=11,
                fontweight="bold",
            )
            ax_zoom.set_xlabel("Test Time Step")
            ax_zoom.set_ylabel("Value")
            ax_zoom.legend(loc="upper right", fontsize=9, frameon=True)
            ax_zoom.grid(True, linestyle=":", alpha=0.5)

            out_dir = save_dir if save_dir else _resolve_path(RES_PATH['plot'].get('MoE', './results/plot/MoE/'))
            os.makedirs(out_dir, exist_ok=True)

            tgt_tag = f"_tgt_[{tgt_mod}]" if tgt_mod else ""
            out_name = f"{date_str}_{tsfm_method}_Comparison_{data_name}_cl[{cl}]_hl[{hl}]{tgt_tag}.png"
            out_path = os.path.join(out_dir, out_name)
            fig.savefig(out_path, dpi=180)
            plt.close(fig)
            saved_paths.append(out_path)

    return saved_paths


def visualize_all_predictions(
    date_filter: Optional[str] = None,
    data_filter: Optional[str] = None,
    save_individual: bool = True,
    save_comparison: bool = True,
    zoom_steps: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    results/predictions/ 하위(TimesFM, LoRA, MoE)의 모든 .npy 파일들을 스캔하여
    개별 예측 플롯 및 모델 간 비교 플롯을 생성합니다.
    """
    pred_cfg = RES_PATH.get('predictions', {})
    search_dirs = []
    if isinstance(pred_cfg, dict):
        for sub_dir in pred_cfg.values():
            resolved = _resolve_path(sub_dir)
            if os.path.isdir(resolved):
                search_dirs.append(resolved)
    else:
        resolved = _resolve_path(str(pred_cfg))
        if os.path.isdir(resolved):
            search_dirs.append(resolved)

    parsed_items: List[Dict[str, Any]] = []
    for s_dir in search_dirs:
        for fname in sorted(os.listdir(s_dir)):
            if not fname.endswith(".npy"):
                continue
            full_path = os.path.join(s_dir, fname)
            meta = parse_prediction_filename(full_path)
            if meta is None:
                continue
            if date_filter and str(meta['date']) != str(date_filter).strip():
                continue
            if data_filter and meta['data_name'].lower() != data_filter.strip().lower():
                continue
            parsed_items.append(meta)

    print(f"🔍 Found {len(parsed_items)} prediction (.npy) files to visualize.")

    individual_plots: List[str] = []
    if save_individual:
        for idx, item in enumerate(parsed_items, 1):
            out_p = plot_single_prediction(item['path'], zoom_steps=zoom_steps)
            if out_p:
                individual_plots.append(out_p)
                print(f"  [{idx}/{len(parsed_items)}] Saved plot -> {out_p}")

    comparison_plots: List[str] = []
    if save_comparison and parsed_items:
        comparison_plots = plot_baseline_comparisons(parsed_items, zoom_steps=zoom_steps)
        for cp in comparison_plots:
            print(f"  📊 Saved comparison plot -> {cp}")

    print(
        f"✅ Visualization completed! "
        f"(Individual plots: {len(individual_plots)}, Comparison plots: {len(comparison_plots)})"
    )
    return individual_plots, comparison_plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize TSFM / LoRA / MoE .npy predictions with Ground Truth targets.")
    parser.add_argument("--date", type=str, default=None, help="특정 실험 날짜(예: 261104)만 필터링하여 시각화")
    parser.add_argument("--data", type=str, default=None, help="특정 데이터셋(예: Etth1)만 필터링하여 시각화")
    parser.add_argument("--zoom_steps", type=int, default=None, help="확대(Zoom-in) 서브플롯에 표시할 최근 스텝 수")
    parser.add_argument("--no_individual", action="store_true", help="개별 .npy 플롯 생성을 건너뛰고 비교 플롯만 생성")
    parser.add_argument("--no_comparison", action="store_true", help="비교 플롯 생성을 건너뛰고 개별 .npy 플롯만 생성")
    args = parser.parse_args()

    visualize_all_predictions(
        date_filter=args.date,
        data_filter=args.data,
        save_individual=not args.no_individual,
        save_comparison=not args.no_comparison,
        zoom_steps=args.zoom_steps,
    )
