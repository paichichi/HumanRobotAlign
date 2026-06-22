#!/usr/bin/env bash
#SBATCH --job-name=hralign_my_solution_one
#SBATCH --account=uoa04758
#SBATCH --partition=genoa,milan
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=18:00:00
#SBATCH --output=/nesi/project/uoa04758/xzha593/logs/RH20T/rvt/%x-%j.out
#SBATCH --error=/nesi/project/uoa04758/xzha593/logs/RH20T/rvt/%x-%j.err

set -euo pipefail

LOG_DIR=/nesi/project/uoa04758/xzha593/logs/RH20T/rvt
HRALIGN_ROOT=/nesi/project/uoa04758/xzha593/GitHub/HumanRobotAlign/rvt
ACTIVATE_SCRIPT=/nesi/project/uoa04758/xzha593/envs/activate_hralign.sh

mkdir -p "${LOG_DIR}"

# ===== resource monitor start =====
MONITOR_INTERVAL=${MONITOR_INTERVAL:-60}
MONITOR_LOG="${LOG_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.monitor.log"
SACCT_LOG="${LOG_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.sacct.log"

monitor_resources() {
  while true; do
    echo "============================================================"
    echo "[monitor] time=$(date)"
    echo "[monitor] host=$(hostname)"
    echo "[monitor] job_id=${SLURM_JOB_ID}"
    echo "[monitor] pwd=$(pwd)"
    echo

    echo "[free -h]"
    free -h || true
    echo

    echo "[top processes by RSS memory]"
    ps -u "$USER" -o pid,ppid,stat,pcpu,pmem,rss,vsz,comm,args --sort=-rss | head -30 || true
    echo

    echo "[nvidia-smi csv]"
    nvidia-smi \
      --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv || true
    echo

    echo "[slurm sstat]"
    sstat -j "${SLURM_JOB_ID}.batch" \
      --format=AveCPU,AveRSS,MaxRSS,AveVMSize,MaxVMSize \
      2>/dev/null || true
    echo

    sleep "${MONITOR_INTERVAL}"
  done
}

monitor_resources > "${MONITOR_LOG}" 2>&1 &
MONITOR_PID=$!

cleanup_monitor() {
  local exit_code=$?

  echo "============================================================" >> "${MONITOR_LOG}"
  echo "[monitor] stopping at $(date), exit_code=${exit_code}" >> "${MONITOR_LOG}"

  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
  fi

  echo "============================================================" > "${SACCT_LOG}"
  echo "[sacct final report] $(date)" >> "${SACCT_LOG}"

  sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,TotalCPU \
    >> "${SACCT_LOG}" 2>&1 || true

  exit "${exit_code}"
}

trap cleanup_monitor EXIT TERM INT
# ===== resource monitor end =====

source "${ACTIVATE_SCRIPT}"
cd "${HRALIGN_ROOT}"

export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "Job info"
echo "============================================================"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "PWD=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"

ls -lh train.py
ls -lh configs/vit_backbones/D4R_IN_rvt2.yaml
ls -lh mvt/configs/rvt2.yaml

nvidia-smi

echo "============================================================"
echo "Start training"
echo "============================================================"

python train.py \
  --exp_cfg_path configs/vit_backbones/D4R_IN_rvt2_2.yaml \
  --mvt_cfg_path mvt/configs/rvt2.yaml \
  --device 0,1
