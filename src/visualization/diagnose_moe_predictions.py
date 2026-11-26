###########################################################################################################
# Deep Diagnostic & Pinpoint Visualization Suite for TSFM / TSFM+LoRA / TSFM-MoE
###########################################################################################################
import argparse
import ast
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
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
from src.model.PEFT.MoE import assign_deterministic_sector
from src.util.data_analysis import calculate_time_series_strength
from src.visualization.plot_predictions import (
    _find_matching_context_experts_csv,
    _resolve_path,
    parse_prediction_filename,
)

plt.rcParams["axes.unicode_minus"] = False

MODEL_ORDER = [
    "TSFM",
    "TSFM+LoRA",
    "STL (Single)",
    "STL (Ensemble)",
    "STL-EMD (Single)",
    "STL-EMD (Ensemble)",
]

MODEL_STYLES: Dict[str, Dict[str, Any]] = {
    "TSFM": {"color": "#6b7280", "linestyle": ":", "linewidth": 1.5},
    "TSFM+LoRA": {"color": "#2563eb", "linestyle": "--", "linewidth": 1.8},
    "STL (Single)": {"color": "#ea580c", "linestyle": "-.", "linewidth": 1.3},
    "STL (Ensemble)": {"color": "#16a34a", "linestyle": "-", "linewidth": 1.9},
    "STL-EMD (Single)": {"color": "#db2777", "linestyle": "-.", "linewidth": 1.3},
    "STL-EMD (Ensemble)": {"color": "#7c3aed", "linestyle": "-", "linewidth": 1.9},
}


###########################################################################################################
# 1. Data Split, Scaling & Metadata Loaders
###########################################################################################################

_SPLIT_CACHE: Dict[Tuple[str, float], Dict[str, Any]] = {}
_TARGET_STRENGTH_CACHE: Dict[Tuple[str, int, int, int], Tuple[np.ndarray, np.ndarray]] = {}


def _resolve_dataset_key(data_name: str) -> str:
    if data_name in DATA_PATH:
        return data_name
    lower_map = {k.lower(): k for k in DATA_PATH.keys()}
    if data_name.lower() in lower_map:
        return lower_map[data_name.lower()]
    raise KeyError(f"DATA_PATH에서 데이터셋 '{data_name}'을(를) 찾을 수 없습니다.")


def load_dataset_split_info(data_name: str, ft_ratio: Optional[float] = None) -> Dict[str, Any]:
    """
    원본 데이터셋을 로드하여 학습 구간(tr_data), 전체 시리즈(target), 테스트 실제값(actuals),
    그리고 Scaled MAE/MSE 계산에 사용되는 학습셋 평균/표준편차(ref_mean, ref_std)를 반환합니다.
    """
    if ft_ratio is None:
        ft_ratio = float(str(PARAMS.get("FT_RATIO", "0.7")).split(",")[0].strip())

    ds_key = _resolve_dataset_key(data_name)
    cache_key = (ds_key, float(ft_ratio))
    if cache_key in _SPLIT_CACHE:
        return _SPLIT_CACHE[cache_key]

    df_path = _resolve_path(DATA_PATH[ds_key])
    tgt_col = DATASET[ds_key]["target_col"]
    df_raw = pd.read_csv(df_path)

    target = df_raw[tgt_col].fillna(0).values.astype(np.float32)
    ft_len = int(len(target) * float(ft_ratio))
    tr_data = target[:ft_len]
    actuals = target[ft_len:]

    ref_mean = float(np.mean(tr_data, dtype=np.float64))
    ref_std = float(np.std(tr_data, dtype=np.float64))
    if ref_std < 1e-8:
        ref_std = 1.0

    info = {
        "data_key": ds_key,
        "target": target,
        "tr_data": tr_data,
        "actuals": actuals,
        "ft_len": ft_len,
        "ref_mean": ref_mean,
        "ref_std": ref_std,
    }
    _SPLIT_CACHE[cache_key] = info
    return info


def _load_moe_performance_lookup(
    date_filter: Optional[str] = None,
    tsfm_filter: Optional[str] = None,
) -> Dict[Tuple[str, str, int, int, str, str], Dict[str, Any]]:
    """
    results/performance/MoE/ 내의 성능 CSV를 스캔하여
    (date, data_lower, cl, hl, decomp_norm, tgt_norm) -> {train_sector_counts, thresholds} 매핑을 구성합니다.
    """
    lookup: Dict[Tuple[str, str, int, int, str, str], Dict[str, Any]] = {}
    perf_dir = _resolve_path(RES_PATH["performance"].get("MoE", "./results/performance/MoE/"))
    if not os.path.isdir(perf_dir):
        return lookup

    for fname in sorted(os.listdir(perf_dir)):
        if not fname.endswith(".csv") or "context_experts" in fname:
            continue
        if "MoE_performance" not in fname:
            continue
        if tsfm_filter and f"_{tsfm_filter.strip().lower()}_moe_" not in fname.lower():
            continue

        # 파일명에서 date 추출 ({DATE}_TimesFM_MoE_performance.csv 또는 TimesFM_MoE_performance_{DATE}.csv)
        stem = os.path.splitext(fname)[0]
        parts = stem.split("_")
        file_date = parts[0] if parts[0].isdigit() else (parts[-1] if parts[-1].isdigit() else str(DATE))
        if date_filter and str(file_date) != str(date_filter).strip():
            continue

        fpath = os.path.join(perf_dir, fname)
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue

        for _, row in df.iterrows():
            d_low = str(row.get("data", "")).strip().lower()
            cl_v = int(row.get("cl", 0))
            hl_v = int(row.get("hl", 0))
            dm_norm = str(row.get("decomp_method", "")).strip().upper().replace("-", "_")
            tgt_raw = str(row.get("target_modules", "")).strip()
            try:
                tgt_list = ast.literal_eval(tgt_raw)
                tgt_norm = ", ".join(tgt_list) if isinstance(tgt_list, (list, tuple)) else tgt_raw
            except Exception:
                tgt_norm = tgt_raw.strip("[]'\"")

            tr_counts: Dict[str, int] = {}
            try:
                parsed_cnt = ast.literal_eval(str(row.get("train_sector_counts", "{}")))
                if isinstance(parsed_cnt, dict):
                    tr_counts = {str(k): int(v) for k, v in parsed_cnt.items()}
            except Exception:
                pass

            thr_tuple: Optional[Tuple[float, ...]] = None
            try:
                parsed_thr = ast.literal_eval(str(row.get("thresholds", "()")))
                if isinstance(parsed_thr, (list, tuple)):
                    thr_tuple = tuple(float(x) for x in parsed_thr)
            except Exception:
                pass

            key = (str(file_date), d_low, cl_v, hl_v, dm_norm, tgt_norm)
            lookup[key] = {
                "train_sector_counts": tr_counts,
                "thresholds": thr_tuple,
            }

    return lookup


def _canonical_moe_model_name(decomp: str, inf_m: str) -> str:
    dm_disp = "STL-EMD" if decomp.strip().upper().replace("-", "_") == "STL_EMD" else "STL"
    inf_disp = inf_m.strip().capitalize()
    return f"{dm_disp} ({inf_disp})"


def discover_experiment_groups(
    date_filter: Optional[str] = None,
    data_filter: Optional[str] = None,
    hl_filter: Optional[List[int]] = None,
    tsfm_filter: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, int, int, str, str], Dict[str, Any]]]:
    """
    results/predictions/ 하위 모든 .npy 파일을 스캔하여
    동일한 (date, data_lower, cl, hl, target_modules) 실험 단위로 묶어 반환합니다.
    """
    pred_cfg = RES_PATH.get("predictions", {})
    search_dirs = []
    if isinstance(pred_cfg, dict):
        for sub_dir in pred_cfg.values():
            resolved = _resolve_path(sub_dir)
            if os.path.isdir(resolved):
                search_dirs.append(resolved)

    all_items: List[Dict[str, Any]] = []
    for s_dir in search_dirs:
        for fname in sorted(os.listdir(s_dir)):
            if not fname.endswith(".npy"):
                continue
            meta = parse_prediction_filename(os.path.join(s_dir, fname))
            if meta is None:
                continue
            if date_filter and str(meta["date"]) != str(date_filter).strip():
                continue
            if tsfm_filter and str(meta["tsfm_method"]).lower() != tsfm_filter.strip().lower():
                continue
            if data_filter and meta["data_name"].lower() != data_filter.strip().lower():
                continue
            if hl_filter and int(meta["hl"]) not in hl_filter:
                continue
            all_items.append(meta)

    # 모델(tsfm_method) 필터가 지정되지 않았고 여러 TSFM 결과가 섞여 있으면 TSFM_METHOD(기본값) 우선 선택
    if tsfm_filter is None and all_items:
        unique_tsfms = sorted({str(x["tsfm_method"]) for x in all_items})
        if len(unique_tsfms) > 1:
            preferred = TSFM_METHOD if TSFM_METHOD in unique_tsfms else unique_tsfms[0]
            print(f"ℹ️ Multiple TSFM backbones detected {unique_tsfms} -> Selecting '{preferred}' (use --tsfm to override).")
            all_items = [x for x in all_items if str(x["tsfm_method"]) == preferred]
            tsfm_filter = preferred

    # 날짜 필터가 지정되지 않았고 여러 날짜가 섞여 있으면 가장 최신 날짜를 자동 선택
    if date_filter is None and all_items:
        unique_dates = sorted({str(x["date"]) for x in all_items})
        if len(unique_dates) > 1:
            latest_date = unique_dates[-1]
            print(f"ℹ️ Multiple experiment dates detected {unique_dates} -> Selecting latest date: {latest_date} (use --date to override).")
            all_items = [x for x in all_items if str(x["date"]) == latest_date]
            date_filter = latest_date

    perf_lookup = _load_moe_performance_lookup(date_filter=date_filter, tsfm_filter=tsfm_filter)

    # (date, data_lower, cl, hl) 기준으로 그룹화한 뒤 target_modules 별로 세부 그룹 구성
    by_base: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for item in all_items:
        b_key = (str(item["date"]), item["data_name"].lower(), int(item["cl"]), int(item["hl"]))
        by_base[b_key].append(item)

    data_order = [x.strip().lower() for x in DATA.split(",") if x.strip()]
    groups: List[Dict[str, Any]] = []

    def _sort_key(k: Tuple[str, str, int, int]):
        dt_s, dn_s, cl_s, hl_s = k
        d_idx = data_order.index(dn_s) if dn_s in data_order else 999
        return (dt_s, d_idx, dn_s, cl_s, hl_s)

    for (date_str, data_low, cl, hl) in sorted(by_base.keys(), key=_sort_key):
        bucket = by_base[(date_str, data_low, cl, hl)]
        tsfm_list = [x for x in bucket if x["category"] in ("Base", "TimesFM")]
        peft_list = [x for x in bucket if x["category"] in ("LoRA", "MoE")]

        tgt_set = sorted({x["target_modules"] for x in peft_list if x.get("target_modules")})
        if not tgt_set:
            tgt_set = ["qkv_proj, out"]

        for tgt_mod in tgt_set:
            models_map: Dict[str, Dict[str, Any]] = {}
            if tsfm_list:
                models_map["TSFM"] = tsfm_list[0]

            for x in peft_list:
                if x.get("target_modules") != tgt_mod:
                    continue
                if x["category"] == "LoRA":
                    models_map["TSFM+LoRA"] = x
                elif x["category"] == "MoE":
                    m_name = _canonical_moe_model_name(x["decomp_method"], x["inference_method"])
                    models_map[m_name] = x

            if len(models_map) < 2:
                continue

            groups.append(
                {
                    "date": date_str,
                    "data_name": data_low,
                    "cl": cl,
                    "hl": hl,
                    "target_modules": tgt_mod,
                    "models": models_map,
                }
            )

    return groups, perf_lookup


def _get_target_window_strengths(data_key: str, win_idx: int, p_start: int, p_end: int, actual_seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    테스트 정답 구간(actual_seg)의 실제 물리적 강도 벡터 (STL [2], STL_EMD [3])를 캐시와 함께 계산합니다.
    """
    c_key = (data_key, int(win_idx), int(p_start), int(p_end))
    if c_key in _TARGET_STRENGTH_CACHE:
        return _TARGET_STRENGTH_CACHE[c_key]

    if len(actual_seg) < 16:
        stl_v = np.array([np.nan, np.nan], dtype=np.float32)
        emd_v = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    else:
        ft_s, fs_s, _, ft_e, fs_e, fi_e = calculate_time_series_strength(actual_seg)
        stl_v = np.array([ft_s, fs_s], dtype=np.float32)
        emd_v = np.array([ft_e, fs_e, fi_e], dtype=np.float32)

    _TARGET_STRENGTH_CACHE[c_key] = (stl_v, emd_v)
    return stl_v, emd_v


###########################################################################################################
# 2. Core Window, Sector, Step-wise & Regime Drift Computation
###########################################################################################################


def collect_all_diagnostics(
    groups: List[Dict[str, Any]],
    perf_lookup: Dict[Tuple[str, str, int, int, str, str], Dict[str, Any]],
    include_regime_drift: bool = True,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int, int, str], Dict[str, np.ndarray]]]:
    """
    모든 실험 그룹에 대해:
      1) 윈도우×분해방식(STL, STL_EMD) 단위 진단 레코드 DataFrame (모델별 Scaled MAE/MSE, 학습 샘플 수 N_k, 강도, 레짐 일치 여부)
      2) (data_name, cl, hl, target_modules) -> {model_name: step_abs_scaled_err_matrix [N_full_win, hl]} 딕셔너리
    를 생성하여 반환합니다.
    """
    window_records: List[Dict[str, Any]] = []
    stepwise_matrices: Dict[Tuple[str, int, int, str], Dict[str, np.ndarray]] = {}

    total_g = len(groups)
    for g_idx, grp in enumerate(groups, 1):
        date_str = grp["date"]
        data_name = grp["data_name"]
        cl = grp["cl"]
        hl = grp["hl"]
        tgt_mod = grp["target_modules"]
        models_map = grp["models"]

        print(f"  [{g_idx:>2}/{total_g}] Diagnosing {data_name} (cl={cl}, hl={hl}, models={len(models_map)})...")

        split_info = load_dataset_split_info(data_name)
        actuals = split_info["actuals"]
        ref_std = split_info["ref_std"]

        # 각 모델의 1D 예측 배열 로드
        preds_1d: Dict[str, np.ndarray] = {}
        for m_name, meta in models_map.items():
            preds_1d[m_name] = np.load(meta["path"]).astype(np.float32).flatten()

        # 윈도우 경계 목록 생성 (actuals 기준 hl 간격)
        win_bounds: List[Tuple[int, int, int]] = []
        pos = 0
        w_i = 0
        while pos < len(actuals):
            end_pos = min(pos + hl, len(actuals))
            win_bounds.append((w_i, pos, end_pos))
            pos += hl
            w_i += 1

        # Step-wise 오차 행렬 (길이가 정확히 hl인 완전 윈도우만 집계)
        full_win_bounds = [(wi, ps, pe) for (wi, ps, pe) in win_bounds if (pe - ps) == hl]
        if full_win_bounds:
            step_dict: Dict[str, np.ndarray] = {}
            for m_name, p_arr in preds_1d.items():
                valid_rows = []
                for (_, ps, pe) in full_win_bounds:
                    if pe <= len(p_arr):
                        err_scaled = np.abs(actuals[ps:pe].astype(np.float64) - p_arr[ps:pe].astype(np.float64)) / ref_std
                        valid_rows.append(err_scaled)
                if valid_rows:
                    step_dict[m_name] = np.stack(valid_rows, axis=0)  # Shape: [M, hl]
            stepwise_matrices[(data_name, cl, hl, tgt_mod)] = step_dict

        # STL / STL_EMD 각각의 컨텍스트 라우팅 CSV 및 학습 샘플 수 로드
        for dm_norm, single_label, ens_label in [
            ("STL", "STL (Single)", "STL (Ensemble)"),
            ("STL_EMD", "STL-EMD (Single)", "STL-EMD (Ensemble)"),
        ]:
            ref_meta = models_map.get(single_label) or models_map.get(ens_label)
            if ref_meta is None:
                continue

            ctx_csv_path = _find_matching_context_experts_csv(ref_meta["path"])
            ctx_df = pd.read_csv(ctx_csv_path) if (ctx_csv_path and os.path.isfile(ctx_csv_path)) else None

            perf_entry = perf_lookup.get((date_str, data_name, cl, hl, dm_norm, tgt_mod), {})
            train_counts: Dict[str, int] = perf_entry.get("train_sector_counts", {})
            thresholds_tup: Optional[Tuple[float, ...]] = perf_entry.get("thresholds")

            for (wi, ps, pe) in win_bounds:
                act_seg = actuals[ps:pe].astype(np.float64)
                rem_len = len(act_seg)
                if rem_len == 0:
                    continue

                # 각 모델의 윈도우별 Scaled MAE / Scaled MSE 산출
                m_metrics: Dict[str, Tuple[float, float]] = {}
                for m_name, p_arr in preds_1d.items():
                    if pe <= len(p_arr):
                        p_seg = p_arr[ps:pe].astype(np.float64)
                        diff_s = (act_seg - p_seg) / ref_std
                        m_metrics[m_name] = (float(np.mean(np.abs(diff_s))), float(np.mean(diff_s ** 2)))
                    else:
                        m_metrics[m_name] = (np.nan, np.nan)

                assigned_sec = "UNKNOWN"
                called_exp_single = ""
                f_t, f_s, f_i = np.nan, np.nan, np.nan
                th_t, th_s, th_i = np.nan, np.nan, np.nan

                if ctx_df is not None and wi < len(ctx_df):
                    crow = ctx_df.iloc[wi]
                    assigned_sec = str(crow.get("assigned_sector", "UNKNOWN")).strip()
                    called_exp_single = str(crow.get("called_expert", "")).strip()
                    f_t = float(crow["F_T"]) if "F_T" in crow and pd.notna(crow["F_T"]) else np.nan
                    f_s = float(crow["F_S"]) if "F_S" in crow and pd.notna(crow["F_S"]) else np.nan
                    f_i = float(crow["F_I"]) if "F_I" in crow and pd.notna(crow["F_I"]) else np.nan
                    th_t = float(crow["theta_T"]) if "theta_T" in crow and pd.notna(crow["theta_T"]) else np.nan
                    th_s = float(crow["theta_S"]) if "theta_S" in crow and pd.notna(crow["theta_S"]) else np.nan
                    th_i = float(crow["theta_I"]) if "theta_I" in crow and pd.notna(crow["theta_I"]) else np.nan

                if thresholds_tup is None and pd.notna(th_t) and pd.notna(th_s):
                    thresholds_tup = (th_t, th_s) if dm_norm == "STL" else (th_t, th_s, th_i)

                train_n = train_counts.get(assigned_sec, np.nan) if train_counts else np.nan

                # 가설 3: 실제 미래 타깃 구간(act_seg)의 섹터와 컨텍스트 섹터 일치 여부(Regime Match vs Drift)
                target_sec = "N/A"
                regime_match: Optional[bool] = None
                if include_regime_drift and thresholds_tup is not None and rem_len >= 32 and assigned_sec != "UNKNOWN":
                    stl_tv, emd_tv = _get_target_window_strengths(split_info["data_key"], wi, ps, pe, act_seg.astype(np.float32))
                    t_vec = stl_tv if dm_norm == "STL" else emd_tv
                    if not np.any(np.isnan(t_vec)):
                        target_sec = assign_deterministic_sector(t_vec, thresholds_tup)
                        regime_match = bool(target_sec == assigned_sec)

                tsfm_mae, tsfm_mse = m_metrics.get("TSFM", (np.nan, np.nan))
                lora_mae, lora_mse = m_metrics.get("TSFM+LoRA", (np.nan, np.nan))
                s_mae, s_mse = m_metrics.get(single_label, (np.nan, np.nan))
                e_mae, e_mse = m_metrics.get(ens_label, (np.nan, np.nan))

                window_records.append(
                    {
                        "date": date_str,
                        "data": data_name,
                        "cl": cl,
                        "hl": hl,
                        "target_modules": tgt_mod,
                        "decomp_method": dm_norm,
                        "window_idx": wi,
                        "pred_start_idx": ps,
                        "pred_end_idx": pe,
                        "rem_len": rem_len,
                        "assigned_sector": assigned_sec,
                        "called_expert": called_exp_single,
                        "train_sector_n": train_n,
                        "F_T": f_t,
                        "F_S": f_s,
                        "F_I": f_i,
                        "theta_T": th_t,
                        "theta_S": th_s,
                        "theta_I": th_i,
                        "target_sector": target_sec,
                        "regime_match": regime_match,
                        "scaled_mae_TSFM": tsfm_mae,
                        "scaled_mae_LoRA": lora_mae,
                        "scaled_mae_MoE_Single": s_mae,
                        "scaled_mae_MoE_Ensemble": e_mae,
                        "scaled_mse_TSFM": tsfm_mse,
                        "scaled_mse_LoRA": lora_mse,
                        "scaled_mse_MoE_Single": s_mse,
                        "scaled_mse_MoE_Ensemble": e_mse,
                        "delta_Single_vs_LoRA": s_mae - lora_mae if (pd.notna(s_mae) and pd.notna(lora_mae)) else np.nan,
                        "delta_Ensemble_vs_LoRA": e_mae - lora_mae if (pd.notna(e_mae) and pd.notna(lora_mae)) else np.nan,
                    }
                )

    return pd.DataFrame(window_records), stepwise_matrices


###########################################################################################################
# 3. Summary Tables & Hypothesis Verification Builders
###########################################################################################################


def _sample_bucket_label(n_val: float) -> str:
    if pd.isna(n_val):
        return "Unknown"
    n_int = int(n_val)
    if n_int <= 10:
        return "1) 0~10 (Ultra-Sparse)"
    if n_int <= 30:
        return "2) 11~30 (Sparse)"
    if n_int <= 100:
        return "3) 31~100 (Moderate)"
    return "4) 101+ (Abundant)"


def build_summary_tables(
    win_df: pd.DataFrame,
    stepwise_matrices: Dict[Tuple[str, int, int, str], Dict[str, np.ndarray]],
) -> Dict[str, pd.DataFrame]:
    """
    가설 1(섹터 샘플 수), 가설 2(128-스텝 블록별 자기회귀 오차), 가설 3(레짐 일치 여부) 요약 DataFrame을 생성합니다.
    """
    tables: Dict[str, pd.DataFrame] = {}
    if win_df.empty:
        return tables

    # (A) 섹터별 상세 오차 분해표 (가중치: rem_len)
    sec_rows: List[Dict[str, Any]] = []
    grp_cols = ["data", "cl", "hl", "decomp_method", "assigned_sector"]
    for keys, sub in win_df.groupby(grp_cols, dropna=False):
        data_n, cl_v, hl_v, dm_v, sec_n = keys
        w = sub["rem_len"].values.astype(np.float64)
        w_sum = np.sum(w)
        if w_sum <= 0:
            continue

        def _wavg(col_name: str) -> float:
            vals = sub[col_name].values.astype(np.float64)
            mask = ~np.isnan(vals)
            if not np.any(mask):
                return np.nan
            return float(np.sum(vals[mask] * w[mask]) / np.sum(w[mask]))

        tr_n = sub["train_sector_n"].iloc[0]
        tsfm_m = _wavg("scaled_mae_TSFM")
        lora_m = _wavg("scaled_mae_LoRA")
        sing_m = _wavg("scaled_mae_MoE_Single")
        ens_m = _wavg("scaled_mae_MoE_Ensemble")

        sec_rows.append(
            {
                "data": data_n,
                "cl": cl_v,
                "hl": hl_v,
                "decomp_method": dm_v,
                "assigned_sector": sec_n,
                "train_sector_n": int(tr_n) if pd.notna(tr_n) else np.nan,
                "test_windows": len(sub),
                "test_steps": int(w_sum),
                "scaled_mae_TSFM": round(tsfm_m, 4) if pd.notna(tsfm_m) else np.nan,
                "scaled_mae_LoRA": round(lora_m, 4) if pd.notna(lora_m) else np.nan,
                "scaled_mae_MoE_Single": round(sing_m, 4) if pd.notna(sing_m) else np.nan,
                "scaled_mae_MoE_Ensemble": round(ens_m, 4) if pd.notna(ens_m) else np.nan,
                "delta_Single_vs_LoRA": round(sing_m - lora_m, 4) if (pd.notna(sing_m) and pd.notna(lora_m)) else np.nan,
                "delta_Ensemble_vs_LoRA": round(ens_m - lora_m, 4) if (pd.notna(ens_m) and pd.notna(lora_m)) else np.nan,
            }
        )
    sec_df = pd.DataFrame(sec_rows)
    tables["sector_breakdown"] = sec_df

    # (B) 가설 1 검증표: 섹터 학습 샘플 수 구간(Bucket)별 오차 차이 및 승률
    work_df = win_df.dropna(subset=["train_sector_n", "scaled_mae_LoRA"]).copy()
    if not work_df.empty:
        work_df["sample_bucket"] = work_df["train_sector_n"].apply(_sample_bucket_label)
        b_rows = []
        for (dm_v, b_lbl), b_sub in work_df.groupby(["decomp_method", "sample_bucket"]):
            d_sing = b_sub["delta_Single_vs_LoRA"].dropna()
            d_ens = b_sub["delta_Ensemble_vs_LoRA"].dropna()
            b_rows.append(
                {
                    "decomp_method": dm_v,
                    "sample_bucket": b_lbl,
                    "num_test_windows": len(b_sub),
                    "mean_train_n": round(float(b_sub["train_sector_n"].mean()), 1),
                    "mean_LoRA_MAE": round(float(b_sub["scaled_mae_LoRA"].mean()), 4),
                    "mean_Single_MAE": round(float(b_sub["scaled_mae_MoE_Single"].mean()), 4),
                    "mean_Ensemble_MAE": round(float(b_sub["scaled_mae_MoE_Ensemble"].mean()), 4),
                    "mean_delta_Single_vs_LoRA": round(float(d_sing.mean()), 4) if len(d_sing) else np.nan,
                    "mean_delta_Ensemble_vs_LoRA": round(float(d_ens.mean()), 4) if len(d_ens) else np.nan,
                    "single_win_rate_vs_LoRA_%": round(float((d_sing <= 0).mean() * 100), 1) if len(d_sing) else np.nan,
                    "ensemble_win_rate_vs_LoRA_%": round(float((d_ens <= 0).mean() * 100), 1) if len(d_ens) else np.nan,
                }
            )
        tables["sample_bucket_summary"] = pd.DataFrame(b_rows)

    # (C) 가설 2 검증표: 128-스텝 자기회귀 롤아웃 블록(Block 1~6)별 오차 추이
    blk_rows: List[Dict[str, Any]] = []
    for (data_n, cl_v, hl_v, tgt_m), m_dict in stepwise_matrices.items():
        n_blocks = int(np.ceil(hl_v / 128.0))
        for b_idx in range(n_blocks):
            s_start = b_idx * 128
            s_end = min((b_idx + 1) * 128, hl_v)
            row_entry: Dict[str, Any] = {
                "data": data_n,
                "cl": cl_v,
                "hl": hl_v,
                "rollout_block": f"Block {b_idx + 1} ({s_start + 1}~{s_end})",
                "step_start": s_start + 1,
                "step_end": s_end,
            }
            for m_name in MODEL_ORDER:
                if m_name in m_dict:
                    row_entry[m_name] = round(float(np.mean(m_dict[m_name][:, s_start:s_end])), 4)
                else:
                    row_entry[m_name] = np.nan
            blk_rows.append(row_entry)
    if blk_rows:
        tables["stepwise_block_summary"] = pd.DataFrame(blk_rows)

    # (D) 가설 3 검증표: Context vs Future Target 레짐 일치(Match) vs 불일치(Drift) 요약
    drift_df = win_df.dropna(subset=["regime_match", "scaled_mae_LoRA"]).copy()
    if not drift_df.empty:
        r_rows = []
        for (dm_v, hl_v, r_match), r_sub in drift_df.groupby(["decomp_method", "hl", "regime_match"]):
            d_sing = r_sub["delta_Single_vs_LoRA"].dropna()
            d_ens = r_sub["delta_Ensemble_vs_LoRA"].dropna()
            total_hl_win = len(drift_df[(drift_df["decomp_method"] == dm_v) & (drift_df["hl"] == hl_v)])
            r_rows.append(
                {
                    "decomp_method": dm_v,
                    "hl": hl_v,
                    "regime_status": "Regime Match (Context == Target)" if r_match else "Regime Drift (Context != Target)",
                    "num_windows": len(r_sub),
                    "share_within_hl_%": round(len(r_sub) / max(1, total_hl_win) * 100.0, 1),
                    "mean_LoRA_MAE": round(float(r_sub["scaled_mae_LoRA"].mean()), 4),
                    "mean_Single_MAE": round(float(r_sub["scaled_mae_MoE_Single"].mean()), 4),
                    "mean_Ensemble_MAE": round(float(r_sub["scaled_mae_MoE_Ensemble"].mean()), 4),
                    "mean_delta_Single_vs_LoRA": round(float(d_sing.mean()), 4) if len(d_sing) else np.nan,
                    "mean_delta_Ensemble_vs_LoRA": round(float(d_ens.mean()), 4) if len(d_ens) else np.nan,
                    "ensemble_win_rate_vs_LoRA_%": round(float((d_ens <= 0).mean() * 100), 1) if len(d_ens) else np.nan,
                }
            )
        tables["regime_drift_summary"] = pd.DataFrame(r_rows)

    return tables


###########################################################################################################
# 4. Visualization 1: Hypothesis 1 — Sector Sample Size & Sector Error Breakdown Plots
###########################################################################################################


def plot_sector_sample_diagnostics(
    sec_df: pd.DataFrame,
    bucket_df: Optional[pd.DataFrame],
    date_str: str,
    out_dir: str,
) -> List[str]:
    saved: List[str] = []
    if sec_df is None or sec_df.empty:
        return saved

    os.makedirs(out_dir, exist_ok=True)
    valid_sec = sec_df.dropna(subset=["train_sector_n", "delta_Single_vs_LoRA"]).copy()
    if not valid_sec.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)

        # Left: Scatter (Train Sector Sample Size N_k vs Delta Scaled MAE)
        ax_sc = axes[0]
        x_n = np.maximum(valid_sec["train_sector_n"].values.astype(np.float64), 0.8)
        ax_sc.scatter(
            x_n,
            valid_sec["delta_Single_vs_LoRA"].values,
            alpha=0.55,
            s=42,
            color="#e11d48",
            marker="o",
            label="MoE (Single) - TSFM+LoRA",
        )
        if "delta_Ensemble_vs_LoRA" in valid_sec.columns:
            ax_sc.scatter(
                x_n,
                valid_sec["delta_Ensemble_vs_LoRA"].values,
                alpha=0.7,
                s=42,
                color="#16a34a",
                marker="^",
                label="MoE (Ensemble) - TSFM+LoRA",
            )

        ax_sc.axhline(0.0, color="#111827", linestyle="-", linewidth=1.2, label="Zero Line (Below 0 = MoE Wins)")
        ax_sc.axvline(30.0, color="#d97706", linestyle="--", linewidth=1.2, label="Sparse Sector Threshold (N = 30)")
        ax_sc.set_xscale("log")
        ax_sc.set_xlabel("Training Sequences in Assigned Sector N_k (Log Scale)")
        ax_sc.set_ylabel("Delta Scaled MAE (MoE - TSFM+LoRA)")
        ax_sc.set_title("[Hypothesis 1] Sector Sample Size N_k vs. Error Gap (Outliers Concentrate in Small N_k)", fontweight="bold")
        ax_sc.grid(True, linestyle=":", alpha=0.5)
        ax_sc.legend(loc="upper right", fontsize=9)

        # 극단 이상치 상위 4개 라벨링
        worst_outliers = valid_sec.sort_values("delta_Single_vs_LoRA", ascending=False).head(4)
        for _, orow in worst_outliers.iterrows():
            if float(orow["delta_Single_vs_LoRA"]) > 0.015:
                ox = max(float(orow["train_sector_n"]), 0.8)
                oy = float(orow["delta_Single_vs_LoRA"])
                ax_sc.annotate(
                    f"{orow['data']}-H{int(orow['hl'])}\n[{orow['assigned_sector']}, N={int(orow['train_sector_n'])}]",
                    (ox, oy),
                    textcoords="offset points",
                    xytext=(6, -4),
                    fontsize=7.5,
                    color="#9f1239",
                    fontweight="bold",
                )

        # Right: Sample Bucket Bar Plot
        ax_bar = axes[1]
        if bucket_df is not None and not bucket_df.empty:
            agg_b = (
                bucket_df.groupby("sample_bucket", as_index=False)[
                    ["mean_delta_Single_vs_LoRA", "mean_delta_Ensemble_vs_LoRA", "ensemble_win_rate_vs_LoRA_%", "num_test_windows"]
                ]
                .mean()
                .sort_values("sample_bucket")
            )
            bx = np.arange(len(agg_b))
            width = 0.35
            b1 = ax_bar.bar(bx - width / 2, agg_b["mean_delta_Single_vs_LoRA"], width, color="#fb7185", label="Mean Delta MAE (Single - LoRA)")
            b2 = ax_bar.bar(bx + width / 2, agg_b["mean_delta_Ensemble_vs_LoRA"], width, color="#34d399", label="Mean Delta MAE (Ensemble - LoRA)")
            ax_bar.axhline(0.0, color="#111827", linewidth=1.1)
            ax_bar.set_xticks(bx)
            ax_bar.set_xticklabels(agg_b["sample_bucket"], rotation=12, fontsize=9)
            ax_bar.set_ylabel("Mean Delta Scaled MAE (Negative = MoE Better)")
            ax_bar.set_title("[Hypothesis 1] Mean Error Gap & Ensemble Win Rate by Sector Sample Bucket", fontweight="bold")
            ax_bar.grid(True, axis="y", linestyle=":", alpha=0.5)
            ax_bar.legend(loc="upper right", fontsize=9)

            for idx_b, (_, brow) in enumerate(agg_b.iterrows()):
                y_pos = max(0.0, float(brow["mean_delta_Single_vs_LoRA"]), float(brow["mean_delta_Ensemble_vs_LoRA"]))
                ax_bar.annotate(
                    f"Ens Win: {brow['ensemble_win_rate_vs_LoRA_%']:.1f}%",
                    (bx[idx_b], y_pos),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#065f46",
                )

        g_path = os.path.join(out_dir, f"{date_str}_Hypothesis1_SampleSize_vs_DeltaMAE.png")
        fig.savefig(g_path, dpi=180)
        plt.close(fig)
        saved.append(g_path)

    # 데이터셋별 섹터 오차 막대 그래프 (H 전체 통합 가중 평균)
    for data_name, d_sub in sec_df.groupby("data"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        for ax_i, dm_v in enumerate(["STL", "STL_EMD"]):
            ax = axes[ax_i]
            dm_sub = d_sub[d_sub["decomp_method"] == dm_v]
            if dm_sub.empty:
                ax.set_visible(False)
                continue

            # 동일 섹터를 모든 Horizon에 대해 test_steps 가중 평균
            s_agg_rows = []
            for sec_k, s_grp in dm_sub.groupby("assigned_sector"):
                sw = s_grp["test_steps"].values.astype(np.float64)
                sw_sum = max(1.0, np.sum(sw))
                s_agg_rows.append(
                    {
                        "sector": sec_k,
                        "train_n": int(np.nanmean(s_grp["train_sector_n"])) if s_grp["train_sector_n"].notna().any() else -1,
                        "test_w": int(s_grp["test_windows"].sum()),
                        "TSFM": float(np.nansum(s_grp["scaled_mae_TSFM"] * sw) / sw_sum),
                        "LoRA": float(np.nansum(s_grp["scaled_mae_LoRA"] * sw) / sw_sum),
                        "Single": float(np.nansum(s_grp["scaled_mae_MoE_Single"] * sw) / sw_sum),
                        "Ensemble": float(np.nansum(s_grp["scaled_mae_MoE_Ensemble"] * sw) / sw_sum),
                    }
                )
            s_agg = pd.DataFrame(s_agg_rows)
            sx = np.arange(len(s_agg))
            w_bar = 0.2
            ax.bar(sx - 1.5 * w_bar, s_agg["TSFM"], w_bar, label="TSFM", color="#9ca3af")
            ax.bar(sx - 0.5 * w_bar, s_agg["LoRA"], w_bar, label="TSFM+LoRA", color="#2563eb")
            ax.bar(sx + 0.5 * w_bar, s_agg["Single"], w_bar, label=f"{dm_v} (Single)", color="#f97316")
            ax.bar(sx + 1.5 * w_bar, s_agg["Ensemble"], w_bar, label=f"{dm_v} (Ensemble)", color="#16a34a")

            x_lbls = [
                f"{r['sector']}\n(Tr={r['train_n']}, Te={r['test_w']})" if r["train_n"] >= 0 else f"{r['sector']}\n(Te={r['test_w']})"
                for _, r in s_agg.iterrows()
            ]
            ax.set_xticks(sx)
            ax.set_xticklabels(x_lbls, rotation=20 if dm_v == "STL_EMD" else 0, fontsize=8.5)
            ax.set_ylabel("Weighted Scaled MAE across Horizons")
            ax.set_title(f"[{data_name.upper()} - {dm_v}] Sector-wise Scaled MAE & Sample Counts", fontweight="bold")
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", fontsize=8.5)

        d_path = os.path.join(out_dir, f"{date_str}_SectorBreakdown_{data_name}.png")
        fig.savefig(d_path, dpi=180)
        plt.close(fig)
        saved.append(d_path)

    return saved


###########################################################################################################
# 5. Visualization 2: Hypothesis 2 — Horizon Step-wise Error Profile (t = 1..H)
###########################################################################################################


def _smooth_curve(arr: np.ndarray, window: int = 9) -> np.ndarray:
    if len(arr) <= window or window <= 1:
        return arr
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=1, center=True).mean().values


def plot_stepwise_error_profiles(
    stepwise_matrices: Dict[Tuple[str, int, int, str], Dict[str, np.ndarray]],
    date_str: str,
    out_dir: str,
) -> List[str]:
    """
    각 예측 스텝 오프셋 t = 1..H 별 Scaled MAE 추이 곡선을 그려
    TimesFM의 단일 패치 출력 경계(128, 256, 384, 512, 640)에서의 자기회귀 오차 누적 양상을 검증합니다.
    """
    saved: List[str] = []
    if not stepwise_matrices:
        return saved

    os.makedirs(out_dir, exist_ok=True)

    # 1) 데이터셋별 2x2 Horizon Grid 플롯
    by_data: Dict[Tuple[str, int, str], Dict[int, Dict[str, np.ndarray]]] = defaultdict(dict)
    by_hl_global: Dict[int, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    for (data_n, cl_v, hl_v, tgt_m), m_dict in stepwise_matrices.items():
        by_data[(data_n, cl_v, tgt_m)][hl_v] = m_dict
        for m_name, mat in m_dict.items():
            by_hl_global[hl_v][m_name].append(np.mean(mat, axis=0))  # dataset mean curve [hl]

    for (data_n, cl_v, tgt_m), hl_map in by_data.items():
        hls_sorted = sorted(hl_map.keys())
        n_panels = len(hls_sorted)
        ncols = 2 if n_panels > 1 else 1
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.8 * ncols, 4.6 * nrows), constrained_layout=True, squeeze=False)
        flat_axes = axes.flatten()

        for idx_h, hl_v in enumerate(hls_sorted):
            ax = flat_axes[idx_h]
            m_dict = hl_map[hl_v]
            t_steps = np.arange(1, hl_v + 1)
            smooth_w = max(5, hl_v // 32)

            for m_name in MODEL_ORDER:
                if m_name not in m_dict:
                    continue
                mean_curve = np.mean(m_dict[m_name], axis=0)
                st = MODEL_STYLES[m_name]
                ax.plot(t_steps, mean_curve, color=st["color"], alpha=0.18, linewidth=0.8)
                ax.plot(
                    t_steps,
                    _smooth_curve(mean_curve, window=smooth_w),
                    label=f"{m_name} (avg={np.mean(mean_curve):.4f})",
                    color=st["color"],
                    linestyle=st["linestyle"],
                    linewidth=st["linewidth"],
                )

            # TimesFM model.o = 128 자기회귀 롤아웃 경계 표시
            ylo, yhi = ax.get_ylim()
            for b_step in range(128, hl_v, 128):
                ax.axvline(b_step, color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.65)
                ax.text(
                    b_step + 3,
                    ylo + 0.03 * (yhi - ylo),
                    f"AR#{b_step // 128 + 1} ({b_step})",
                    fontsize=7.5,
                    color="#b91c1c",
                )

            ax.set_title(f"[{data_n.upper()}] Step-wise Scaled MAE (H = {hl_v})", fontsize=10.5, fontweight="bold")
            ax.set_xlabel("Forecast Horizon Step t (1 .. H)")
            ax.set_ylabel("Step-wise Scaled MAE")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper left", fontsize=8, frameon=True)

        for extra_i in range(n_panels, len(flat_axes)):
            flat_axes[extra_i].set_visible(False)

        out_p = os.path.join(out_dir, f"{date_str}_StepwiseError_{data_n}_cl[{cl_v}].png")
        fig.savefig(out_p, dpi=180)
        plt.close(fig)
        saved.append(out_p)

    # 2) 전체 데이터셋 평균 Global Summary 플롯
    hls_global = sorted(by_hl_global.keys())
    if hls_global:
        ncols = 2 if len(hls_global) > 1 else 1
        nrows = int(np.ceil(len(hls_global) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8.0 * ncols, 4.8 * nrows), constrained_layout=True, squeeze=False)
        flat_axes = axes.flatten()

        for idx_h, hl_v in enumerate(hls_global):
            ax = flat_axes[idx_h]
            t_steps = np.arange(1, hl_v + 1)
            smooth_w = max(5, hl_v // 32)
            for m_name in MODEL_ORDER:
                curves = by_hl_global[hl_v].get(m_name, [])
                if not curves:
                    continue
                g_curve = np.mean(np.stack(curves, axis=0), axis=0)
                st = MODEL_STYLES[m_name]
                ax.plot(
                    t_steps,
                    _smooth_curve(g_curve, window=smooth_w),
                    label=f"{m_name} (avg={np.mean(g_curve):.4f})",
                    color=st["color"],
                    linestyle=st["linestyle"],
                    linewidth=st["linewidth"] + 0.2,
                )

            for b_step in range(128, hl_v, 128):
                ax.axvline(b_step, color="#dc2626", linestyle="--", linewidth=1.1, alpha=0.7)

            ax.set_title(
                f"[All Datasets Avg] Step-wise Scaled MAE (H = {hl_v} | Red dashed = 128-step AR Rollout)",
                fontsize=10.5,
                fontweight="bold",
            )
            ax.set_xlabel("Forecast Horizon Step t (1 .. H)")
            ax.set_ylabel("Mean Step-wise Scaled MAE")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper left", fontsize=8.5, frameon=True)

        for extra_i in range(len(hls_global), len(flat_axes)):
            flat_axes[extra_i].set_visible(False)

        g_out = os.path.join(out_dir, f"{date_str}_Hypothesis2_StepwiseError_GlobalSummary.png")
        fig.savefig(g_out, dpi=180)
        plt.close(fig)
        saved.append(g_out)

    return saved


###########################################################################################################
# 6. Visualization 3: Hypothesis 3 — Context vs. Target Regime Drift Plot
###########################################################################################################


def plot_regime_drift_diagnostics(
    drift_summary_df: Optional[pd.DataFrame],
    date_str: str,
    out_dir: str,
) -> Optional[str]:
    if drift_summary_df is None or drift_summary_df.empty:
        return None

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), constrained_layout=True)

    # Left: Horizon에 따른 Regime Match 비율(%) 변화
    ax_rate = axes[0]
    for dm_v, color, marker in [("STL", "#2563eb", "o"), ("STL_EMD", "#7c3aed", "s")]:
        m_sub = drift_summary_df[
            (drift_summary_df["decomp_method"] == dm_v)
            & (drift_summary_df["regime_status"].str.startswith("Regime Match"))
        ].sort_values("hl")
        if not m_sub.empty:
            ax_rate.plot(
                m_sub["hl"].astype(str),
                m_sub["share_within_hl_%"],
                marker=marker,
                markersize=7,
                linewidth=2.0,
                color=color,
                label=f"{dm_v} Regime Match Rate (%)",
            )
            for _, r in m_sub.iterrows():
                ax_rate.annotate(
                    f"{r['share_within_hl_%']:.1f}%",
                    (str(int(r["hl"])), float(r["share_within_hl_%"])),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=8.5,
                )

    ax_rate.set_ylim(0, 105)
    ax_rate.set_xlabel("Forecast Horizon H")
    ax_rate.set_ylabel("Context == Future Target Sector Match Rate (%)")
    ax_rate.set_title("[Hypothesis 3] Regime Stability Across Forecast Horizons", fontweight="bold")
    ax_rate.grid(True, linestyle=":", alpha=0.5)
    ax_rate.legend(loc="upper right")

    # Right: Regime Match vs Regime Drift 구간에서의 MoE Ensemble 승률 및 오차 격차
    ax_cmp = axes[1]
    agg_r = (
        drift_summary_df.groupby(["decomp_method", "regime_status"], as_index=False)[
            ["mean_delta_Single_vs_LoRA", "mean_delta_Ensemble_vs_LoRA", "ensemble_win_rate_vs_LoRA_%"]
        ]
        .mean()
    )
    labels = [f"{r['decomp_method']}\n({'Match' if 'Match' in r['regime_status'] else 'Drift'})" for _, r in agg_r.iterrows()]
    rx = np.arange(len(agg_r))
    w = 0.34
    ax_cmp.bar(rx - w / 2, agg_r["mean_delta_Single_vs_LoRA"], w, color="#fb7185", label="Mean Delta MAE (Single - LoRA)")
    ax_cmp.bar(rx + w / 2, agg_r["mean_delta_Ensemble_vs_LoRA"], w, color="#34d399", label="Mean Delta MAE (Ensemble - LoRA)")
    ax_cmp.axhline(0.0, color="#111827", linewidth=1.1)
    ax_cmp.set_xticks(rx)
    ax_cmp.set_xticklabels(labels, fontsize=9.5)
    ax_cmp.set_ylabel("Mean Delta Scaled MAE (Negative = MoE Better)")
    ax_cmp.set_title("[Hypothesis 3] Error Gap in Regime-Matched vs. Regime-Drifted Windows", fontweight="bold")
    ax_cmp.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax_cmp.legend(loc="upper left", fontsize=9)

    for idx_r, (_, rrow) in enumerate(agg_r.iterrows()):
        y_top = max(0.0, float(rrow["mean_delta_Single_vs_LoRA"]), float(rrow["mean_delta_Ensemble_vs_LoRA"]))
        ax_cmp.annotate(
            f"Ens Win: {rrow['ensemble_win_rate_vs_LoRA_%']:.1f}%",
            (rx[idx_r], y_top),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#065f46",
        )

    out_p = os.path.join(out_dir, f"{date_str}_Hypothesis3_RegimeDrift_Summary.png")
    fig.savefig(out_p, dpi=180)
    plt.close(fig)
    return out_p


###########################################################################################################
# 7. Visualization 4: Best & Worst Top-K Window Pinpoint Forecast Comparison Plots
###########################################################################################################


def plot_pinpoint_windows(
    groups: List[Dict[str, Any]],
    win_df: pd.DataFrame,
    out_dir: str,
    top_k: int = 2,
    context_tail: int = 192,
) -> List[str]:
    """
    각 (data, cl, hl) 설정마다 TSFM+LoRA 대비 MoE가:
      - 가장 크게 개선된 Best Top-K 윈도우 (상단)
      - 가장 크게 악화된 Worst Top-K 윈도우 (하단; 이상치 원인 규명용)
    를 자동 추출하여 [과거 Context 구간 + 미래 예측 구간(Actual & 6개 모델) + 섹터/강도/샘플수 메타데이터]를 시각화합니다.
    """
    saved: List[str] = []
    if win_df.empty:
        return saved

    os.makedirs(out_dir, exist_ok=True)

    for grp in groups:
        date_str = grp["date"]
        data_name = grp["data_name"]
        cl = grp["cl"]
        hl = grp["hl"]
        tgt_mod = grp["target_modules"]
        models_map = grp["models"]

        sub = win_df[
            (win_df["date"] == date_str)
            & (win_df["data"] == data_name)
            & (win_df["cl"] == cl)
            & (win_df["hl"] == hl)
            & (win_df["target_modules"] == tgt_mod)
        ]
        if sub.empty:
            continue

        # 윈도우별 대표 비교 지표 산출 (STL / STL_EMD 중 존재하는 행을 윈도우 번호로 인덱싱)
        stl_sub = sub[sub["decomp_method"] == "STL"].drop_duplicates(subset=["window_idx"]).set_index("window_idx")
        emd_sub = sub[sub["decomp_method"] == "STL_EMD"].drop_duplicates(subset=["window_idx"]).set_index("window_idx")
        all_win_ids = sorted(set(sub["window_idx"].unique()))
        if not all_win_ids:
            continue

        # 윈도우 랭킹: Best는 Ensemble 최소 delta 기준, Worst는 Single/Ensemble 최대 양수(+) 스파이크 기준
        rank_scores: List[Tuple[int, float, float]] = []
        for wi in all_win_ids:
            ens_deltas = []
            worst_deltas = []
            for tbl in (stl_sub, emd_sub):
                if wi in tbl.index:
                    r = tbl.loc[wi]
                    if pd.notna(r.get("delta_Ensemble_vs_LoRA")):
                        ens_deltas.append(float(r["delta_Ensemble_vs_LoRA"]))
                        worst_deltas.append(float(r["delta_Ensemble_vs_LoRA"]))
                    if pd.notna(r.get("delta_Single_vs_LoRA")):
                        worst_deltas.append(float(r["delta_Single_vs_LoRA"]))
            best_score = min(ens_deltas) if ens_deltas else 0.0
            worst_score = max(worst_deltas) if worst_deltas else 0.0
            rank_scores.append((wi, best_score, worst_score))

        best_sorted = [x[0] for x in sorted(rank_scores, key=lambda t: t[1])[:top_k]]
        worst_sorted = [x[0] for x in sorted(rank_scores, key=lambda t: t[2], reverse=True)[:top_k]]

        selected: List[Tuple[str, int]] = []
        seen_w = set()
        for rank_i, wi in enumerate(best_sorted, 1):
            selected.append((f"BEST #{rank_i} (MoE Beats Baseline)", wi))
            seen_w.add(wi)
        for rank_i, wi in enumerate(worst_sorted, 1):
            if wi not in seen_w:
                selected.append((f"WORST #{rank_i} (Outlier / Failure Case)", wi))
                seen_w.add(wi)

        if not selected:
            continue

        split_info = load_dataset_split_info(data_name)
        target_full = split_info["target"]
        ft_len = split_info["ft_len"]
        actuals = split_info["actuals"]
        ref_std = split_info["ref_std"]

        preds_1d: Dict[str, np.ndarray] = {
            m_name: np.load(meta["path"]).astype(np.float32).flatten()
            for m_name, meta in models_map.items()
        }

        n_rows = len(selected)
        fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4.3 * n_rows), constrained_layout=True, squeeze=False)

        for row_idx, (tag_title, wi) in enumerate(selected):
            ax = axes[row_idx, 0]
            ref_row = stl_sub.loc[wi] if wi in stl_sub.index else emd_sub.loc[wi]
            ps = int(ref_row["pred_start_idx"])
            pe = int(ref_row["pred_end_idx"])
            rem_len = pe - ps

            # 과거 Context 구간 슬라이싱 (target_full 상에서 예측 시작점 = ft_len + ps)
            abs_origin = ft_len + ps
            ctx_len_show = min(cl, context_tail, abs_origin)
            ctx_vals = target_full[abs_origin - ctx_len_show : abs_origin]
            act_vals = actuals[ps:pe]

            t_ctx = np.arange(-ctx_len_show, 0)
            t_fut = np.arange(0, rem_len)

            # 배경 음영 및 Context 시계열 플롯
            ax.axvspan(-ctx_len_show, 0, color="#f3f4f6", alpha=0.75)
            ax.axvline(0, color="#111827", linestyle="-", linewidth=1.3)

            # Context 마지막 값과 미래 첫 값을 자연스럽게 연결
            t_gt = np.concatenate(([t_ctx[-1]], t_fut)) if len(t_ctx) > 0 else t_fut
            v_gt = np.concatenate(([ctx_vals[-1]], act_vals)) if len(t_ctx) > 0 else act_vals
            ax.plot(t_ctx, ctx_vals, color="#374151", linewidth=1.5, label=f"Context (last {ctx_len_show} of {cl})")
            ax.plot(t_gt, v_gt, color="#111827", linewidth=2.1, label="Ground Truth (Target)")

            # 각 모델 예측선 및 해당 윈도우 Scaled MAE 표시
            for m_name in MODEL_ORDER:
                if m_name not in preds_1d:
                    continue
                p_arr = preds_1d[m_name]
                if pe > len(p_arr):
                    continue
                p_seg = p_arr[ps:pe]
                w_smae = float(np.mean(np.abs(act_vals.astype(np.float64) - p_seg.astype(np.float64)) / ref_std))
                st = MODEL_STYLES[m_name]
                t_p = np.concatenate(([t_ctx[-1]], t_fut)) if len(t_ctx) > 0 else t_fut
                v_p = np.concatenate(([ctx_vals[-1]], p_seg)) if len(t_ctx) > 0 else p_seg
                ax.plot(
                    t_p,
                    v_p,
                    color=st["color"],
                    linestyle=st["linestyle"],
                    linewidth=st["linewidth"],
                    alpha=0.92,
                    label=f"{m_name} (Scaled MAE={w_smae:.4f})",
                )

            # 128-스텝 자기회귀 롤아웃 경계선 표시
            for b_step in range(128, rem_len, 128):
                ax.axvline(b_step, color="#dc2626", linestyle=":", linewidth=1.0, alpha=0.65)

            # 섹터, 학습 샘플 수(N_k), 강도 지표 요약 헤더 구성
            info_parts = [f"[{data_name.upper()} | H={hl} | Win #{wi} ({ps}:{pe})] {tag_title}"]
            if wi in stl_sub.index:
                sr = stl_sub.loc[wi]
                sn_str = f"N={int(sr['train_sector_n'])}" if pd.notna(sr["train_sector_n"]) else "N=?"
                drift_s = ""
                if pd.notna(sr.get("regime_match")):
                    drift_s = ", Match" if bool(sr["regime_match"]) else f", Drift->{sr['target_sector']}"
                info_parts.append(
                    f"STL: {sr['assigned_sector']} ({sn_str}, FT={sr['F_T']:.2f}/FS={sr['F_S']:.2f}{drift_s})"
                )
            if wi in emd_sub.index:
                er = emd_sub.loc[wi]
                en_str = f"N={int(er['train_sector_n'])}" if pd.notna(er["train_sector_n"]) else "N=?"
                drift_e = ""
                if pd.notna(er.get("regime_match")):
                    drift_e = ", Match" if bool(er["regime_match"]) else f", Drift->{er['target_sector']}"
                info_parts.append(
                    f"STL-EMD: {er['assigned_sector']} ({en_str}, FT={er['F_T']:.2f}/FS={er['F_S']:.2f}/FI={er['F_I']:.2f}{drift_e})"
                )

            title_color = "#065f46" if tag_title.startswith("BEST") else "#9f1239"
            ax.set_title("  |  ".join(info_parts), fontsize=9.5, fontweight="bold", color=title_color)
            ax.set_xlabel("Relative Time Step (0 = Forecast Origin, Red dotted = 128-step AR Rollout)")
            ax.set_ylabel("Target Value")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=True)

        out_name = f"{date_str}_Pinpoint_{data_name}_cl[{cl}]_hl[{hl}]_tgt_[{tgt_mod}].png"
        out_p = os.path.join(out_dir, out_name)
        fig.savefig(out_p, dpi=180)
        plt.close(fig)
        saved.append(out_p)

    return saved


###########################################################################################################
# 8. Markdown Diagnostic Report Generator & Main Runner
###########################################################################################################


def _df_to_md_table(df: pd.DataFrame) -> str:
    """외부 tabulate 패키지 설치 여부와 무관하게 깔끔한 GitHub Markdown 표를 생성합니다."""
    if df is None or df.empty:
        return "_No data_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = [str(c) for c in df.columns]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body_lines = []
        for _, row in df.iterrows():
            vals = [str(row[c]) if pd.notna(row[c]) else "-" for c in df.columns]
            body_lines.append("| " + " | ".join(vals) + " |")
        return "\n".join([header, sep] + body_lines)


def write_diagnostic_markdown_report(
    date_str: str,
    win_df: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    report_path: str,
) -> str:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    lines: List[str] = [
        f"# TSFM-MoE Post-Hoc Diagnostic Verification Report (`{date_str}`)",
        "",
        "저장된 `.npy` 예측값과 `_context_experts.csv` 라우팅 기록을 바탕으로 3대 가설을 정량 검증한 요약 리포트입니다.",
        "",
    ]

    # 1) 가설 1 요약
    lines.append("## 1. 가설 1 검증: 섹터 학습 샘플 수($N_k$)별 과적합 여부")
    b_df = tables.get("sample_bucket_summary")
    if b_df is not None and not b_df.empty:
        lines.append("")
        lines.append(_df_to_md_table(b_df))
        lines.append("")

    # 상위 이상치 윈도우 Top 10
    if not win_df.empty:
        top_bad = (
            win_df.dropna(subset=["delta_Single_vs_LoRA"])
            .sort_values("delta_Single_vs_LoRA", ascending=False)
            .head(10)[
                [
                    "data",
                    "hl",
                    "decomp_method",
                    "window_idx",
                    "assigned_sector",
                    "train_sector_n",
                    "scaled_mae_LoRA",
                    "scaled_mae_MoE_Single",
                    "scaled_mae_MoE_Ensemble",
                    "delta_Single_vs_LoRA",
                ]
            ]
            .round(4)
        )
        lines.append("### Worst 10 Outlier Test Windows (`MoE Single` vs `TSFM+LoRA`)")
        lines.append("")
        lines.append(_df_to_md_table(top_bad))
        lines.append("")

    # 2) 가설 2 요약 (H=720 블록별 평균)
    blk_df = tables.get("stepwise_block_summary")
    if blk_df is not None and not blk_df.empty:
        h720_df = blk_df[blk_df["hl"] == blk_df["hl"].max()]
        if not h720_df.empty:
            num_cols = [c for c in MODEL_ORDER if c in h720_df.columns]
            g_blk = h720_df.groupby(["rollout_block", "step_start"], as_index=False)[num_cols].mean().sort_values("step_start")
            g_blk = g_blk.drop(columns=["step_start"]).round(4)
            lines.append(f"## 2. 가설 2 검증: 장기 예측(`H={int(blk_df['hl'].max())}`) 128-스텝 자기회귀 블록별 평균 Scaled MAE")
            lines.append("")
            lines.append(_df_to_md_table(g_blk))
            lines.append("")

    # 3) 가설 3 요약
    r_df = tables.get("regime_drift_summary")
    if r_df is not None and not r_df.empty:
        lines.append("## 3. 가설 3 검증: Context-Target 레짐 유지(Match) vs 레짐 전환(Drift)")
        lines.append("")
        lines.append(_df_to_md_table(r_df))
        lines.append("")

    content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    return report_path


def run_moe_diagnostics(
    date_filter: Optional[str] = None,
    data_filter: Optional[str] = None,
    hl_filter: Optional[List[int]] = None,
    tsfm_filter: Optional[str] = None,
    top_k: int = 2,
    include_regime_drift: bool = True,
) -> Dict[str, Any]:
    """
    저장된 예측(.npy) 및 컨텍스트 라우팅(.csv) 파일들을 종합 분석하여
    수치 테이블(.csv), 진단 리포트(.md), 4종 심층 시각화(.png)를 일괄 생성합니다.
    """
    groups, perf_lookup = discover_experiment_groups(
        date_filter=date_filter,
        data_filter=data_filter,
        hl_filter=hl_filter,
        tsfm_filter=tsfm_filter,
    )
    if not groups:
        print("⚠️ 분석할 예측 결과(.npy) 그룹을 찾지 못했습니다. results/predictions/ 경로와 --date 옵션을 확인해 주세요.")
        return {}

    active_date = groups[0]["date"]
    print(f"🔬 Starting TSFM-MoE Deep Diagnostics for date={active_date} ({len(groups)} experiment settings)...")

    win_df, stepwise_matrices = collect_all_diagnostics(
        groups=groups,
        perf_lookup=perf_lookup,
        include_regime_drift=include_regime_drift,
    )
    tables = build_summary_tables(win_df, stepwise_matrices)

    perf_diag_dir = _resolve_path("./results/performance/diagnosis/")
    plot_diag_dir = _resolve_path("./results/plot/diagnosis/")
    os.makedirs(perf_diag_dir, exist_ok=True)
    os.makedirs(plot_diag_dir, exist_ok=True)

    # CSV 저장
    saved_csvs: Dict[str, str] = {}
    win_csv_path = os.path.join(perf_diag_dir, f"{active_date}_window_level_diagnostics.csv")
    win_df.to_csv(win_csv_path, index=False)
    saved_csvs["window_level"] = win_csv_path

    for t_key, t_df in tables.items():
        c_path = os.path.join(perf_diag_dir, f"{active_date}_{t_key}.csv")
        t_df.to_csv(c_path, index=False)
        saved_csvs[t_key] = c_path

    # 플롯 생성
    print("📊 Generating Hypothesis 1 (Sector Sample Size & Sector Breakdown) plots...")
    sec_plots = plot_sector_sample_diagnostics(
        sec_df=tables.get("sector_breakdown", pd.DataFrame()),
        bucket_df=tables.get("sample_bucket_summary"),
        date_str=active_date,
        out_dir=os.path.join(plot_diag_dir, "sector_analysis"),
    )

    print("📈 Generating Hypothesis 2 (Step-wise t=1..H Autoregressive Error) plots...")
    step_plots = plot_stepwise_error_profiles(
        stepwise_matrices=stepwise_matrices,
        date_str=active_date,
        out_dir=os.path.join(plot_diag_dir, "stepwise_error"),
    )

    drift_plot = None
    if include_regime_drift and "regime_drift_summary" in tables:
        print("🔄 Generating Hypothesis 3 (Context vs. Target Regime Drift) plot...")
        drift_plot = plot_regime_drift_diagnostics(
            drift_summary_df=tables.get("regime_drift_summary"),
            date_str=active_date,
            out_dir=os.path.join(plot_diag_dir, "regime_drift"),
        )

    print(f"🎯 Generating Best & Worst Top-{top_k} Pinpoint Window Forecast plots...")
    pinpoint_plots = plot_pinpoint_windows(
        groups=groups,
        win_df=win_df,
        out_dir=os.path.join(plot_diag_dir, "pinpoint_windows"),
        top_k=top_k,
    )

    report_md_path = os.path.join(perf_diag_dir, f"{active_date}_MoE_Diagnostic_Report.md")
    write_diagnostic_markdown_report(active_date, win_df, tables, report_md_path)

    print("\n" + "=" * 72)
    print(f"✅ Deep Diagnostics Completed! (Date: {active_date})")
    print(f"  - Markdown Summary Report : {report_md_path}")
    print(f"  - Diagnostic CSV Tables   : {perf_diag_dir} ({len(saved_csvs)} files)")
    print(f"  - Sector Analysis Plots   : {len(sec_plots)} saved -> {os.path.join(plot_diag_dir, 'sector_analysis')}")
    print(f"  - Step-wise Error Plots   : {len(step_plots)} saved -> {os.path.join(plot_diag_dir, 'stepwise_error')}")
    if drift_plot:
        print(f"  - Regime Drift Plot       : 1 saved -> {drift_plot}")
    print(f"  - Pinpoint Window Plots   : {len(pinpoint_plots)} saved -> {os.path.join(plot_diag_dir, 'pinpoint_windows')}")
    print("=" * 72)

    if "sample_bucket_summary" in tables:
        print("\n[Preview] Hypothesis 1 — Sector Sample Size Bucket Summary:")
        print(tables["sample_bucket_summary"].to_string(index=False))

    return {
        "report_md": report_md_path,
        "csvs": saved_csvs,
        "sector_plots": sec_plots,
        "stepwise_plots": step_plots,
        "drift_plot": drift_plot,
        "pinpoint_plots": pinpoint_plots,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run post-hoc diagnostics & pinpoint window visualizations for TSFM / LoRA / TSFM-MoE."
    )
    parser.add_argument("--date", type=str, default=None, help="특정 실험 날짜 필터 (예: 261122). 미지정 시 최신 날짜 자동 선택")
    parser.add_argument("--tsfm", type=str, default=None, help="특정 TSFM 백본만 분석 (예: TimesFM, Chronos, Moirai)")
    parser.add_argument("--data", type=str, default=None, help="특정 데이터셋만 분석 (예: exchange, weather, ettm1)")
    parser.add_argument("--hl", type=str, default=None, help="특정 Horizon만 분석 (쉼표 구분, 예: '336,720')")
    parser.add_argument("--top_k", type=int, default=2, help="각 설정별 Best / Worst 핀포인트 윈도우 추출 개수 (기본값: 2)")
    parser.add_argument("--no_regime_drift", action="store_true", help="실제 미래 타깃 구간의 STL/EMD 레짐 일치 분석 생략 (빠른 실행용)")
    args = parser.parse_args()

    hl_list_arg = [int(x.strip()) for x in args.hl.split(",") if x.strip()] if args.hl else None

    run_moe_diagnostics(
        date_filter=args.date,
        data_filter=args.data,
        hl_filter=hl_list_arg,
        tsfm_filter=args.tsfm,
        top_k=args.top_k,
        include_regime_drift=not args.no_regime_drift,
    )
