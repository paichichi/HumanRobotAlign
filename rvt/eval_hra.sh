#!/usr/bin/env bash
set -euo pipefail

cd /home/paichichi/projects/HumanRobotAlign/rvt

MODEL_FOLDER="runs/UnadaptedR3M_rvt2_bs6_lr1.25e-5"
MODEL_NAME="model_4_eval.pth"
DATA_ROOT="/home/paichichi/data/AGNOSTOS/unseen_tasks/test"

EPISODES=25
EPISODE_LENGTH=25
GPU_ID=0
RUNS=(1 2 3)

RUN_NAME="UnadaptedR3M_rvt2_bs6_lr1.25e-5"
LOG_ROOT="logs/${RUN_NAME}"

export COPPELIASIM_ROOT=/home/paichichi/software/CoppeliaSim_4_1_0
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$COPPELIASIM_ROOT/lib:${LD_LIBRARY_PATH:-}
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export QT_PLUGIN_PATH=$COPPELIASIM_ROOT
export QT_QPA_PLATFORM=xcb

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

for task in "${tasks[@]}"; do
  for run_id in "${RUNS[@]}"; do
    RUN_DIR="${LOG_ROOT}/${task}/run${run_id}"
    mkdir -p "$RUN_DIR"

    STDOUT_LOG="${RUN_DIR}/stdout.log"
    EPISODE_CSV="${RUN_DIR}/episodes.csv"
    SUMMARY_CSV="${RUN_DIR}/summary.csv"
    LOG_NAME="${RUN_NAME}_${task}_run${run_id}"

    echo "Evaluating ${task}, run ${run_id}"

    CUDA_VISIBLE_DEVICES="$GPU_ID" python -X faulthandler -u eval.py \
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

    SRC_SUMMARY="${MODEL_FOLDER}/eval/${LOG_NAME}/${MODEL_NAME%.pth}/eval_results.csv"
    [[ -f "$SRC_SUMMARY" ]] && cp "$SRC_SUMMARY" "$SUMMARY_CSV"

    python - "$STDOUT_LOG" "$EPISODE_CSV" <<'PY'
import csv
import re
import sys

pattern = re.compile(
    r"^Evaluating (?P<task>.*?) \| Episode (?P<episode>\d+) \| "
    r"Score: (?P<score>.*?) \| Episode Length: (?P<episode_length>\d+) \| "
    r"Lang Goal: (?P<lang_goal>.*)$"
)

rows = []
with open(sys.argv[1], "r", errors="ignore") as f:
    for line in f:
        match = pattern.match(line.strip())
        if match:
            row = match.groupdict()
            row["message"] = line.strip()
            rows.append(row)

with open(sys.argv[2], "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["task", "episode", "score", "episode_length", "lang_goal", "message"],
    )
    writer.writeheader()
    writer.writerows(rows)
PY
  done
done

echo "All done. Logs saved to: ${LOG_ROOT}"
