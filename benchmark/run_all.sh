#!/bin/bash
# Full experiment chain, run sequentially: two torch jobs must never contend for
# the 12 GB card (three concurrent jobs previously died with "Resource
# temporarily unavailable").
#
#   1  age-cutoff sweep + conformal refit for the 2 h time-varying regime
#   2  fine-tune CONTROL, vorticity term off  -- separates "the term helped"
#      from "training longer helped"
#   3  fine-tune with the vorticity term on
#   4  score all three checkpoints on the frozen benchmark
#
# Both fine-tune arms use the same seed and --deterministic_data, so they see
# byte-identical batches in identical order (verified empirically).
set -u

TRAIN_DIR="/workspace/ocean/Conditional DDPM"
PY=/venv/main/bin/python
BENCH=/workspace/DDPMLibrary/benchmark

COMMON=(
  --pickle /workspace/ocean/Datasets/pickles/data_raw_chrono.pickle
  --vcnn_ckpt /workspace/ocean/Models/vcnn_weights.pt
  --vcnn_module_dir /workspace/ocean/Models
  --mean_type vcnn --noise_max 0.1 --noise_cond 1 --use_dist 1 --data_std 0.10628
  --lags 13,25 --path_steps 120,200 --batch 16 --lr 2e-4 --base_ch 64 --min_snr 5.0
  --workers 2 --seed 20260830 --deterministic_data
  --epochs 10 --max_batches 300 --diag_every 5
  --init_from /workspace/DDPMLibrary/src/ddpm_library/assets/corrdiff_weights.pt
)

echo "=== [1/4] age cutoff + conformal refit ==="
"$PY" -u "$BENCH/age_calibration.py" > /workspace/age_cal.log 2>&1
echo "step1 exit=$?"

echo "=== [2/4] fine-tune control, lambda_vort=0 ==="
cd "$TRAIN_DIR" || exit 1
"$PY" -u train_corrdiff_vort.py "${COMMON[@]}" --lambda_vort 0.0 \
  --save_dir /workspace/vort_lam0 > /workspace/vort_lam0.log 2>&1
echo "step2 exit=$?"

echo "=== [3/4] fine-tune vorticity, lambda_vort=1 ==="
cd "$TRAIN_DIR" || exit 1
"$PY" -u train_corrdiff_vort.py "${COMMON[@]}" --lambda_vort 1.0 \
  --save_dir /workspace/vort_lam1 > /workspace/vort_lam1.log 2>&1
echo "step3 exit=$?"

echo "=== [4/4] evaluate all three checkpoints ==="
"$PY" -u "$BENCH/eval_vort.py" > /workspace/eval_vort.log 2>&1
echo "step4 exit=$?"

echo ALL_DONE
