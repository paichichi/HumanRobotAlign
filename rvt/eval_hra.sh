#!/usr/bin/env bash
set -euo pipefail

cd /home/xli990/paichichi/GitHub/HumanRobotAlign/rvt

# =====================
# Config
# =====================
MODEL_FOLDER="runs/UnadaptedR3M2RLBench"
MODEL_NAME="model_4.pth"
DATA_ROOT="/home/xli990/paichichi/GitHub/HumanRobotAlign/rvt/runs/data/test"

EPISODES=25
EPISODE_LENGTH=25
GPU_ID=0

RUN_NAME="debug_original_push_buttons"
LOG_ROOT="logs/hralign_eval/${RUN_NAME}"

tasks=(
  "push_buttons"
  "open_drawer"
  "sweep_to_dustpan_of_size"
)

# tasks=(
#   "put_toilet_roll_on_stand"
#   "put_knife_on_chopping_board"
#   "close_fridge"
#   "close_microwave"
#   "close_laptop_lid"
#   "phone_on_base"
#   "toilet_seat_down"
#   "lamp_off"
#   "lamp_on"
#   "put_books_on_bookshelf"
#   "put_umbrella_in_umbrella_stand"
#   "open_grill"
#   "put_rubbish_in_bin"
#   "take_usb_out_of_computer"
#   "take_lid_off_saucepan"
#   "take_plate_off_colored_dish_rack"
#   "basketball_in_hoop"
#   "scoop_with_spatula"
#   "straighten_rope"
#   "turn_oven_on"
#   "beat_the_buzz"
#   "water_plants"
#   "unplug_charger"
# )

# =====================
# Environment
# =====================
export COPPELIASIM_ROOT=/home/xli990/software/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$COPPELIASIM_ROOT/lib:$CONDA_PREFIX/lib:/usr/lib/nvidia
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export QT_PLUGIN_PATH=$COPPELIASIM_ROOT
export QT_QPA_PLATFORM=xcb

export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$LOG_ROOT"


cat > "${LOG_ROOT}/config.yaml" <<EOF
run_name: ${RUN_NAME}
model_folder: ${MODEL_FOLDER}
model_name: ${MODEL_NAME}
data_root: ${DATA_ROOT}
episodes: ${EPISODES}
episode_length: ${EPISODE_LENGTH}
gpu_id: ${GPU_ID}
tasks:
$(printf "  - %s\n" "${tasks[@]}")
EOF

# =====================
# Run tasks
# =====================
for task in "${tasks[@]}"; do
  echo "======================================"
  echo "Evaluating task: ${task}"
  echo "======================================"

  TASK_DIR="${LOG_ROOT}/${task}"
  mkdir -p "$TASK_DIR"

  STDOUT_LOG="${TASK_DIR}/stdout.log"
  EPISODE_CSV="${TASK_DIR}/episodes.csv"
  SUMMARY_CSV="${TASK_DIR}/summary.csv"

  LOG_NAME="${RUN_NAME}_${task}"

  CUDA_VISIBLE_DEVICES="$GPU_ID" xvfb-run -a \
    -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
    python -X faulthandler -u eval.py \
      --model-folder "$MODEL_FOLDER" \
      --eval-datafolder "$DATA_ROOT" \
      --tasks "$task" \
      --eval-episodes "$EPISODES" \
      --episode-length "$EPISODE_LENGTH" \
      --log-name "$LOG_NAME" \
      --device 0 \
      --headless \
      --model-name "$MODEL_NAME" \
    2>&1 | tee "$STDOUT_LOG"

  # 复制 HR-Align 原始 summary csv
  SRC_SUMMARY="${MODEL_FOLDER}/eval/${LOG_NAME}/${MODEL_NAME%.pth}/eval_results.csv"
  if [[ -f "$SRC_SUMMARY" ]]; then
    cp "$SRC_SUMMARY" "$SUMMARY_CSV"
  fi

  # 从 stdout 解析每个 episode 的 message
  python - "$STDOUT_LOG" "$EPISODE_CSV" <<'PY'
import csv
import re
import sys

stdout_log = sys.argv[1]
out_csv = sys.argv[2]

pattern = re.compile(
    r"^Evaluating (?P<task>.*?) \| Episode (?P<episode>\d+) \| "
    r"Score: (?P<score>.*?) \| Episode Length: (?P<episode_length>\d+) \| "
    r"Lang Goal: (?P<lang_goal>.*)$"
)

rows = []

with open(stdout_log, "r", errors="ignore") as f:
    for line in f:
        line = line.strip()
        m = pattern.match(line)
        if m:
            row = m.groupdict()
            row["message"] = line
            rows.append(row)

fieldnames = [
    "task",
    "episode",
    "score",
    "episode_length",
    "lang_goal",
    "message",
]

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] wrote {len(rows)} rows to {out_csv}")
PY

  echo "[DONE] ${task}"
  echo "episode csv: ${EPISODE_CSV}"
  echo "summary csv: ${SUMMARY_CSV}"
done

echo "All done. Logs saved to: ${LOG_ROOT}"