#!/bin/bash
# Grid search over wd_time_scale_ratio and omega for Dion2
# Parallelizes across two GPU pairs: (0,1) and (2,3)

set -e

CONFIG="configs/dion2_160m.yaml"
WD_RATIOS=(0.025 0.05 0.1 0.2)
OMEGAS=(2 4 8)

# Build list of (wd_ratio, omega) pairs
PAIRS=()
for wd in "${WD_RATIOS[@]}"; do
    for omega in "${OMEGAS[@]}"; do
        PAIRS+=("${wd},${omega}")
    done
done

echo "Total runs: ${#PAIRS[@]}"
echo "GPU pairs: (0,1) and (2,3)"
echo "========================================="

run_job() {
    local gpus="$1"
    local wd_ratio="$2"
    local omega="$3"
    local job_name="wd${wd_ratio}_omega${omega}"

    echo "[$(date +%H:%M:%S)] START  ${job_name} on GPUs ${gpus}"

    CUDA_VISIBLE_DEVICES="${gpus}" \
    torchrun --standalone --nproc_per_node=2 \
        --master_port=$((29500 + RANDOM % 1000)) \
        train.py \
        --config "${CONFIG}" \
        --wd_time_scale_ratio "${wd_ratio}" \
        --omega "${omega}" \
        --wandb_job_name "${job_name}" \
        2>&1 | tee "logs_sweep/${job_name}.log"

    echo "[$(date +%H:%M:%S)] DONE   ${job_name}"
}

mkdir -p logs_sweep

idx=0
while [ $idx -lt ${#PAIRS[@]} ]; do
    # Launch up to 2 runs in parallel
    pids=()

    for gpu_pair in "0,1" "2,3"; do
        if [ $idx -ge ${#PAIRS[@]} ]; then
            break
        fi

        IFS=',' read -r wd omega <<< "${PAIRS[$idx]}"
        run_job "${gpu_pair}" "${wd}" "${omega}" &
        pids+=($!)
        idx=$((idx + 1))
    done

    # Wait for both runs in this batch to finish
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "========================================="
done

echo "All ${#PAIRS[@]} runs complete."
