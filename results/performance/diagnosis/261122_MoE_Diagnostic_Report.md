# TSFM-MoE Post-Hoc Diagnostic Verification Report (`261122`)

저장된 `.npy` 예측값과 `_context_experts.csv` 라우팅 기록을 바탕으로 3대 가설을 정량 검증한 요약 리포트입니다.

## 1. 가설 1 검증: 섹터 학습 샘플 수($N_k$)별 과적합 여부

| decomp_method | sample_bucket | num_test_windows | mean_train_n | mean_LoRA_MAE | mean_Single_MAE | mean_Ensemble_MAE | mean_delta_Single_vs_LoRA | mean_delta_Ensemble_vs_LoRA | single_win_rate_vs_LoRA_% | ensemble_win_rate_vs_LoRA_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STL | 2) 11~30 (Sparse) | 14 | 29.1 | 0.4466 | 0.5282 | 0.4835 | 0.0816 | 0.0369 | 35.7 | 35.7 |
| STL | 3) 31~100 (Moderate) | 259 | 70.8 | 0.3163 | 0.3165 | 0.3112 | 0.0002 | -0.0051 | 47.5 | 52.9 |
| STL | 4) 101+ (Abundant) | 1623 | 335.5 | 0.1971 | 0.2027 | 0.1981 | 0.0056 | 0.001 | 45.7 | 49.8 |
| STL_EMD | 1) 0~10 (Ultra-Sparse) | 18 | 5.7 | 0.3316 | 0.3488 | 0.3335 | 0.0172 | 0.002 | 33.3 | 50.0 |
| STL_EMD | 2) 11~30 (Sparse) | 103 | 22.7 | 0.3176 | 0.3236 | 0.3158 | 0.006 | -0.0019 | 45.6 | 50.5 |
| STL_EMD | 3) 31~100 (Moderate) | 301 | 53.1 | 0.168 | 0.1793 | 0.1697 | 0.0113 | 0.0017 | 37.5 | 45.8 |
| STL_EMD | 4) 101+ (Abundant) | 1474 | 290.2 | 0.2163 | 0.2233 | 0.2175 | 0.007 | 0.0012 | 44.2 | 47.8 |

### Worst 10 Outlier Test Windows (`MoE Single` vs `TSFM+LoRA`)

| data | hl | decomp_method | window_idx | assigned_sector | train_sector_n | scaled_mae_LoRA | scaled_mae_MoE_Single | scaled_mae_MoE_Ensemble | delta_Single_vs_LoRA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weather | 336 | STL_EMD | 34 | HT_LS_HI | 63 | 0.0233 | 1.1054 | 0.5571 | 1.082 |
| exchange | 720 | STL_EMD | 1 | LT_HS_LI | 27 | 0.1692 | 1.1305 | 0.5861 | 0.9613 |
| exchange | 336 | STL | 3 | LT_LS | 29 | 0.3192 | 0.9798 | 0.6464 | 0.6606 |
| ettm2 | 336 | STL_EMD | 36 | HT_HS_LI | 62 | 0.5957 | 0.9661 | 0.7809 | 0.3703 |
| ettm2 | 96 | STL_EMD | 135 | HT_LS_HI | 240 | 0.583 | 0.9209 | 0.7519 | 0.3379 |
| exchange | 720 | STL | 0 | LT_LS | 27 | 0.6287 | 0.9569 | 0.7928 | 0.3282 |
| etth2 | 96 | STL_EMD | 54 | HT_HS_HI | 122 | 0.3086 | 0.613 | 0.4591 | 0.3044 |
| exchange | 720 | STL | 1 | HT_HS | 27 | 0.1692 | 0.4702 | 0.2872 | 0.301 |
| ettm1 | 96 | STL_EMD | 137 | LT_HS_LI | 288 | 0.0809 | 0.3785 | 0.22 | 0.2976 |
| etth2 | 96 | STL | 17 | HT_HS | 120 | 0.2654 | 0.5593 | 0.4083 | 0.2939 |

## 2. 가설 2 검증: 장기 예측(`H=720`) 128-스텝 자기회귀 블록별 평균 Scaled MAE

| rollout_block | TSFM | TSFM+LoRA | STL (Single) | STL (Ensemble) | STL-EMD (Single) | STL-EMD (Ensemble) |
| --- | --- | --- | --- | --- | --- | --- |
| Block 1 (1~128) | 0.2464 | 0.234 | 0.2377 | 0.2351 | 0.2334 | 0.2328 |
| Block 2 (129~256) | 0.3815 | 0.3542 | 0.36 | 0.3554 | 0.38 | 0.3568 |
| Block 3 (257~384) | 0.4102 | 0.3658 | 0.3921 | 0.3749 | 0.416 | 0.3853 |
| Block 4 (385~512) | 0.4062 | 0.3435 | 0.395 | 0.3616 | 0.4127 | 0.3697 |
| Block 5 (513~640) | 0.4327 | 0.3662 | 0.4389 | 0.3981 | 0.4579 | 0.409 |
| Block 6 (641~720) | 0.4096 | 0.3384 | 0.4139 | 0.3685 | 0.4411 | 0.3841 |

## 3. 가설 3 검증: Context-Target 레짐 유지(Match) vs 레짐 전환(Drift)

| decomp_method | hl | regime_status | num_windows | share_within_hl_% | mean_LoRA_MAE | mean_Single_MAE | mean_Ensemble_MAE | mean_delta_Single_vs_LoRA | mean_delta_Ensemble_vs_LoRA | ensemble_win_rate_vs_LoRA_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STL | 96 | Regime Drift (Context != Target) | 699 | 71.3 | 0.1891 | 0.1945 | 0.1897 | 0.0054 | 0.0006 | 48.9 |
| STL | 96 | Regime Match (Context == Target) | 282 | 28.7 | 0.1957 | 0.1975 | 0.1948 | 0.0019 | -0.0008 | 53.5 |
| STL | 192 | Regime Drift (Context != Target) | 325 | 66.1 | 0.228 | 0.2289 | 0.2257 | 0.0009 | -0.0023 | 53.8 |
| STL | 192 | Regime Match (Context == Target) | 167 | 33.9 | 0.2243 | 0.2241 | 0.2226 | -0.0002 | -0.0018 | 53.3 |
| STL | 336 | Regime Drift (Context != Target) | 197 | 69.6 | 0.2535 | 0.2657 | 0.2566 | 0.0122 | 0.0031 | 51.8 |
| STL | 336 | Regime Match (Context == Target) | 86 | 30.4 | 0.2416 | 0.254 | 0.2446 | 0.0124 | 0.003 | 39.5 |
| STL | 720 | Regime Drift (Context != Target) | 97 | 72.9 | 0.2957 | 0.3083 | 0.299 | 0.0127 | 0.0033 | 42.3 |
| STL | 720 | Regime Match (Context == Target) | 36 | 27.1 | 0.2407 | 0.2669 | 0.2519 | 0.0262 | 0.0112 | 36.1 |
| STL_EMD | 96 | Regime Drift (Context != Target) | 782 | 79.7 | 0.1909 | 0.1965 | 0.191 | 0.0056 | 0.0001 | 48.5 |
| STL_EMD | 96 | Regime Match (Context == Target) | 199 | 20.3 | 0.1912 | 0.1982 | 0.193 | 0.007 | 0.0018 | 45.2 |
| STL_EMD | 192 | Regime Drift (Context != Target) | 396 | 80.5 | 0.2219 | 0.2269 | 0.2215 | 0.0051 | -0.0004 | 48.5 |
| STL_EMD | 192 | Regime Match (Context == Target) | 96 | 19.5 | 0.2469 | 0.2617 | 0.252 | 0.0148 | 0.0051 | 41.7 |
| STL_EMD | 336 | Regime Drift (Context != Target) | 218 | 77.0 | 0.2361 | 0.2477 | 0.2384 | 0.0116 | 0.0023 | 49.1 |
| STL_EMD | 336 | Regime Match (Context == Target) | 65 | 23.0 | 0.2961 | 0.3144 | 0.3039 | 0.0183 | 0.0078 | 36.9 |
| STL_EMD | 720 | Regime Drift (Context != Target) | 104 | 78.2 | 0.2821 | 0.2995 | 0.2865 | 0.0174 | 0.0044 | 49.0 |
| STL_EMD | 720 | Regime Match (Context == Target) | 29 | 21.8 | 0.2761 | 0.2746 | 0.2742 | -0.0015 | -0.0019 | 48.3 |
