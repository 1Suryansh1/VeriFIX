# All-11 EXPERIMENT Summary

| Run | Val | Patch | Time | Ct | H | L | H bucket | L bucket |
|---|---:|---:|---:|---:|---:|---:|---|---|
| v4_full_n06_a04_ct045_b1d1_2026_04_10 | 25 | 40 | 20.0 | 0.45 | 3/20 | 3/20 | VALIDATION_BUDGET_EXHAUSTED | VALIDATION_BUDGET_EXHAUSTED |
| v4_full_n06_a04_ct000_b1d1_2026_04_10 | 25 | 40 | 20.0 | 0.0 | 3/20 | 3/20 | VALIDATION_BUDGET_EXHAUSTED | VALIDATION_BUDGET_EXHAUSTED |
| v4_full_n02_a08_ct045_b1d1_2026_04_10 | 25 | 40 | 20.0 | 0.45 | 3/20 | 3/20 | VALIDATION_BUDGET_EXHAUSTED | VALIDATION_BUDGET_EXHAUSTED |
| v4_full_n02_a08_ct000_b1d1_2026_04_10 | 25 | 40 | 20.0 | 0.0 | 3/20 | 3/20 | VALIDATION_BUDGET_EXHAUSTED | VALIDATION_BUDGET_EXHAUSTED |
| v4_v3only_unchoked_policy500_ct000_b1d1_2026_04_10 | 25 | 500 | 20.0 | 0.0 | 3/20 | 3/20 | VALIDATION_BUDGET_EXHAUSTED | VALIDATION_BUDGET_EXHAUSTED |
| v4_v3only_unchoked_critic500_ct060_b1d1_2026_04_10 | 25 | 500 | 20.0 | 0.6 | 2/20 | 3/20 | CRITIC_GATE_OVERFILTERING | VALIDATION_BUDGET_EXHAUSTED |
| v4_v3sniper_V1_control_val025_2026_04_11_030623 | 25 | 500 | 20.0 | 0.0 | 3/20 | 3/20 | VALIDATION_BUDGET_EXHAUSTED | VALIDATION_BUDGET_EXHAUSTED |
| v4_v3sniper_V2_val050_2026_04_11_030623 | 50 | 500 | 20.0 | 0.0 | 5/20 | 5/20 | VALIDATION_BUDGET_EXHAUSTED | VALIDATION_BUDGET_EXHAUSTED |
| v4_v3sniper_V3_val100_2026_04_11_030623 | 100 | 500 | 20.0 | 0.0 | 6/20 | 6/20 | VALIDATION_BUDGET_EXHAUSTED | VALIDATION_BUDGET_EXHAUSTED |
| v4_v3sniper_V4_val150_2026_04_11_030623 | 150 | 500 | 25.0 | 0.0 | 8/20 | 8/20 | VALIDATED_NOT_PLAUSIBLE | VALIDATED_NOT_PLAUSIBLE |
| v4_v3sniper_V5_val200_2026_04_11_030623 | 200 | 500 | 30.0 | 0.0 | 9/20 | 9/20 | VALIDATED_NOT_PLAUSIBLE | VALIDATED_NOT_PLAUSIBLE |

## Key Deltas
- critic_full_n06: {'h_rep_delta': 0, 'l_rep_delta': 0, 'h_rate_delta': 0.0, 'l_rate_delta': 0.0, 'h_avg_val_delta': -5.25, 'l_avg_val_delta': 0.0}
- critic_full_n02: {'h_rep_delta': 0, 'l_rep_delta': 0, 'h_rate_delta': 0.0, 'l_rate_delta': 0.0, 'h_avg_val_delta': -5.4, 'l_avg_val_delta': 0.0}
- critic_unchoked: {'h_rep_delta': -1, 'l_rep_delta': 0, 'h_rate_delta': -0.05, 'l_rate_delta': 0.0, 'h_avg_val_delta': -18.5, 'l_avg_val_delta': 0.0}
- weight_ct000: {'h_rep_delta': 0, 'l_rep_delta': 0, 'h_rate_delta': 0.0, 'l_rate_delta': 0.0, 'h_avg_val_delta': -0.4, 'l_avg_val_delta': -0.4}
- weight_ct045: {'h_rep_delta': 0, 'l_rep_delta': 0, 'h_rate_delta': 0.0, 'l_rate_delta': 0.0, 'h_avg_val_delta': -0.55, 'l_avg_val_delta': -0.4}
- overnight_vs_v1:
  - v4_v3sniper_V1_control_val025_2026_04_11_030623: val=25 h=3 (d0), l=3 (d0)
  - v4_v3sniper_V2_val050_2026_04_11_030623: val=50 h=5 (d2), l=5 (d2)
  - v4_v3sniper_V3_val100_2026_04_11_030623: val=100 h=6 (d3), l=6 (d3)
  - v4_v3sniper_V4_val150_2026_04_11_030623: val=150 h=8 (d5), l=8 (d5)
  - v4_v3sniper_V5_val200_2026_04_11_030623: val=200 h=9 (d6), l=9 (d6)

## Best
- Hybrid: v4_v3sniper_V5_val200_2026_04_11_030623 -> 9/20
- Latent: v4_v3sniper_V5_val200_2026_04_11_030623 -> 9/20

## Barrier Break Runs (>3/20)
- v4_v3sniper_V2_val050_2026_04_11_030623
- v4_v3sniper_V3_val100_2026_04_11_030623
- v4_v3sniper_V4_val150_2026_04_11_030623
- v4_v3sniper_V5_val200_2026_04_11_030623