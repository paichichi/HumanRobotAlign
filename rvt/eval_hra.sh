#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 1. Load environment
# ============================================================

source /home/xli990/paichichi/GitHub/hralign_eval_bundle/activate_eval_env.sh

export PROJECT_ROOT=/home/xli990/paichichi/GitHub/hralign_eval_bundle
export RVT_ROOT="${PROJECT_ROOT}/rvt"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

export PATH=/home/xli990/bin:${CONDA_PREFIX}/bin:${PATH}

export COPPELIASIM_ROOT=/home/xli990/software/CoppeliaSim
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
unset QT_QPA_PLATFORM
unset QT_PLUGIN_PATH

cd "${RVT_ROOT}"

# ============================================================
# 2. Basic config
# ============================================================

MODEL_NAME="model_14.pth"

DATA_ROOT="/home/xli990/paichichi/GitHub/X-ICM/data/unseen_tasks/test"

EPISODES=25
EPISODE_LENGTH=25
GPU_ID=0
RUNS=(1 2 3)

EXPERIMENTS=(
  "D4R_100days_of_hands_rvt2_bs6_lr1.25e-5"
  "D4R_ImageNet_rvt2_bs6_lr1.25e-5"
  "D4R_Kinetics_rvt2_bs6_lr1.25e-5"
  "D4R_SOUP_rvt2_bs6_lr1.25e-5"
  "HRP_Ego4D_rvt2_bs6_lr1.25e-5"
)

tasks=(
  "put_toilet_roll_on_stand"
  "put_knife_on_chopping_board"
  "close_fridge"
  "close_microwave"
  "close_laptop_lid"
  "phone_on_base"
  "toilet_seat_down"
  "lamp_off"
  "lamp_on"
  "put_books_on_bookshelf"
  "put_umbrella_in_umbrella_stand"
  "open_grill"
  "put_rubbish_in_bin"
  "take_usb_out_of_computer"
  "take_lid_off_saucepan"
  "take_plate_off_colored_dish_rack"
  "basketball_in_hoop"
  "scoop_with_spatula"
  "straighten_rope"
  "turn_oven_on"
  "beat_the_buzz"
  "water_plants"
  "unplug_charger"
)

# ============================================================
# 3. Run rollouts
# ============================================================

for exp_name in "${EXPERIMENTS[@]}"; do
  MODEL_FOLDER="runs/${exp_name}"

  # logs/hralign_eval/${RUN_NAME}/...
  RUN_NAME="${exp_name}_${MODEL_NAME%.pth}"
  LOG_ROOT="logs/hralign_eval/${RUN_NAME}"

  echo "============================================================"
  echo "Experiment: ${exp_name}"
  echo "Model folder: ${MODEL_FOLDER}"
  echo "Model name: ${MODEL_NAME}"
  echo "Log root: ${LOG_ROOT}"
  echo "============================================================"

  for task in "${tasks[@]}"; do
    for run_id in "${RUNS[@]}"; do
      RUN_DIR="${LOG_ROOT}/${task}/run${run_id}"
      mkdir -p "${RUN_DIR}"

      STDOUT_LOG="${RUN_DIR}/stdout.log"
      XVFB_LOG="${RUN_DIR}/xvfb.log"
      LOG_NAME="${RUN_NAME}_${task}_run${run_id}"

      echo "------------------------------------------------------------"
      echo "Task: ${task}"
      echo "Run: ${run_id}"
      echo "Log: ${STDOUT_LOG}"
      echo "------------------------------------------------------------"

      unset QT_QPA_PLATFORM
      unset QT_PLUGIN_PATH
      export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"

      CUDA_VISIBLE_DEVICES="${GPU_ID}" xvfb-run -a \
        -e "${XVFB_LOG}" \
        -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
        python -X faulthandler -u eval.py \
          --model-folder "${MODEL_FOLDER}" \
          --eval-datafolder "${DATA_ROOT}" \
          --tasks "${task}" \
          --eval-episodes "${EPISODES}" \
          --episode-length "${EPISODE_LENGTH}" \
          --log-name "${LOG_NAME}" \
          --device 0 \
          --headless \
          --model-name "${MODEL_NAME}" \
        2>&1 | tee "${STDOUT_LOG}"

    done
  done

  echo "Finished: ${RUN_NAME}"
done

echo "All done. Logs saved under logs/hralign_eval/"