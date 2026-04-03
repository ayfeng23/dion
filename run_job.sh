#!/bin/bash
set -euo pipefail
# Submit a dion training job to the bonete cluster via cjob.
#
# Usage:
#   ./run_job.sh <config> [--gpus N] [--duration SECONDS] [--cjob]
#
# Examples:
#   ./run_job.sh configs/normuon_160m.yaml --gpus 1 --cjob
#   ./run_job.sh configs/normuon_160m.yaml --gpus 2 --cjob
#   ./run_job.sh configs/normuon_160m.yaml              # local torchrun

CJOB="${CJOB:-/data/cluster-jobs/cjob}"
USER_SHORT="v-austinfeng"
DION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG=""
GPUS="1"
DURATION="14400"
MODE_CJOB=false
NO_WANDB=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)     GPUS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --cjob)     MODE_CJOB=true; shift ;;
    --no-wandb) NO_WANDB=true; shift ;;
    -*)         echo "Unknown flag: $1" >&2; exit 1 ;;
    *)
      if [[ -z "$CONFIG" ]]; then
        CONFIG="$1"; shift
      else
        echo "Unexpected argument: $1" >&2; exit 1
      fi
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Usage: run_job.sh <config.yaml> [--gpus N] [--duration SECONDS] [--cjob] [--no-wandb]" >&2
  exit 1
fi

# Derive a job name from the config filename (e.g. normuon_160m)
JOB_NAME="dion-$(basename "$CONFIG" .yaml)"

DAY_STAMP="$(TZ='America/New_York' date +%F)"
LOG_ROOT="/data/${USER_SHORT}/logs/${DAY_STAMP}"

if $MODE_CJOB; then
  mkdir -p "$LOG_ROOT"

  [[ -z "${WANDB_BASE_URL:-}" && -z "${WANDB_HOST:-}" ]] && \
    WANDB_BASE_URL="https://microsoft-research.wandb.io"

  CJOB_ARGS=(
    --name "$JOB_NAME"
    --upload "$DION_ROOT"
    --priority p1
    --gpus "$GPUS"
    --duration "$DURATION"
    --workstream dion
    --fetch-back-subdir . "${LOG_ROOT}/"
    --fetch-back-exclude source_dir
    --fetch-back-exclude volcano_scripts
  )

  [[ -n "${WANDB_BASE_URL:-}" ]] && CJOB_ARGS+=(--env "WANDB_BASE_URL=${WANDB_BASE_URL}")
  [[ -n "${WANDB_API_KEY:-}" ]]  && CJOB_ARGS+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")

  WANDB_FLAG=""
  if $NO_WANDB; then
    WANDB_FLAG="--no_wandb true"
  fi

  # USE_TORCHRUN=0 so volcano_command.sh runs our script directly (we invoke torchrun ourselves)
  CJOB_ARGS+=(--env "USE_TORCHRUN=0")

  set -x
  "$CJOB" enqueue "${CJOB_ARGS[@]}" \
    -- ./cluster_entry.sh "$CONFIG" "$GPUS" $WANDB_FLAG
else
  cd "$DION_ROOT"
  WANDB_FLAG=""
  if $NO_WANDB; then
    WANDB_FLAG="--no_wandb true"
  fi
  torchrun --standalone --nproc_per_node="$GPUS" train.py --config "$CONFIG" $WANDB_FLAG
fi
