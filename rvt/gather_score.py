import os
import numpy as np
import pandas as pd
import statistics


seen_tasks = [
    "put_item_in_drawer",
    "reach_and_drag",
    "turn_tap",
    "slide_block_to_color_target",
    "open_drawer",
    "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter",
    "put_money_in_safe",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "light_bulb_in",
    "sweep_to_dustpan_of_size",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
]

unseen_tasks=[ 
    "put_toilet_roll_on_stand",
    "put_knife_on_chopping_board",
    "close_fridge",
    "close_microwave",
    "close_laptop_lid",
    "phone_on_base",
    "toilet_seat_down",
    "lamp_off",
    "lamp_on",
    "put_books_on_bookshelf", 
    "put_umbrella_in_umbrella_stand",
    "open_grill", 
    "put_rubbish_in_bin",

    "take_usb_out_of_computer",
    "take_lid_off_saucepan",
    "take_plate_off_colored_dish_rack",
    "basketball_in_hoop",
    "scoop_with_spatula", 
    "straighten_rope", 
    "turn_oven_on", 
    "beat_the_buzz",
	"water_plants", 
    "unplug_charger",
]


def get_scores(score_path):
    if not os.path.exists(score_path):
        print("not exist: ", score_path)
        return None

    try:
        df = pd.read_csv(score_path)

        if len(df) == 0:
            print("empty file: ", score_path)
            return None
        
        final_score = float(df.iloc[-1]["success rate"])

        return final_score

    except Exception as e:
        print("parse failed:", score_path)
        print(e)
        return None


def gather_score_single(exp_path, seed=0, prefix=""):
    if prefix=="":
        task_names=[task_name for task_name in os.listdir(exp_path) if ".txt" not in task_name]
    elif prefix=='seen_':
        task_names=seen_tasks
    elif prefix=="unseen_":
        task_names=unseen_tasks
    else:
       pass
    scores=[]
    for task_name in task_names:
        score_file=exp_path+f"/{task_name}/run{seed+1}/summary.csv"
        score_i = get_scores(score_file)
        scores.append(score_i)
    f=open(exp_path+f"/{prefix}scores_run{seed+1}.txt", 'w')
    for idx,score in enumerate(scores):
        f.write("%s: %.2f\n"%(task_names[idx],scores[idx]))
    f.write("averaged success rate: %.2f"%statistics.mean(scores))
    f.close()
    return 


def parse_score_file(file_path):
    """
    Parse the score file and return a dictionary of task names and their success rates.
    Also return the averaged success rate at the end of the file.
    """
    scores = {}
    averaged_success_rate = None
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("averaged success rate"):
                # Extract the averaged success rate
                averaged_success_rate = float(line.split(":")[-1].strip())
            else:
                # Extract task name and success rate
                task_name, success_rate = line.split(":")
                scores[task_name.strip()] = float(success_rate.strip())
    return scores, averaged_success_rate

def summarize_scores(folder_path, prefix=""):
    """
    Summarize all score files in the given folder, calculate the mean and std for each task,
    and write the results into an output file.
    """
    task_scores = {}
    averaged_success_rates = []
    task_scores_level_1=[]
    task_scores_level_2=[]
    # Read all files in the folder
    for file_name in os.listdir(folder_path):
        task_scores_level_1_single=[]
        task_scores_level_2_single=[]
        if file_name.endswith(".txt") and file_name.startswith(prefix+"scores_"):
            file_path = os.path.join(folder_path, file_name)
            scores, averaged_success_rate = parse_score_file(file_path)
            # print(file_path, scores, averaged_success_rate)
            # Accumulate scores

            # Collect averaged success rates
            if averaged_success_rate is not None:
                averaged_success_rates.append(averaged_success_rate)

            for task, score in scores.items():
                if task not in task_scores:
                    task_scores[task] = []
                task_scores[task].append(score)
            
                if prefix=="unseen_":
                    if task in unseen_tasks[:13]:
                        task_scores_level_1_single.append(score)
                    elif task in unseen_tasks[13:]:
                        task_scores_level_2_single.append(score)
                    else:
                        pass
            
            task_scores_level_1.append(np.mean(task_scores_level_1_single))
            task_scores_level_2.append(np.mean(task_scores_level_2_single))
                
    # Prepare the summary content
    summary_lines = []
    for task, scores in task_scores.items():
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        summary_lines.append(f"{task}: {avg_score:.2f} ({std_score:.2f})")
    
    # Calculate the mean of the averaged success rates
    mean_averaged_success_rate = np.mean(averaged_success_rates)
    std_averaged_success_rate = np.std(averaged_success_rates)

    # if prefix=="unseen_":
    #     summary_lines.append(f"level-1 success rate: {np.mean(task_scores_level_1):.2f} ({np.std(task_scores_level_1):.2f})")
    #     summary_lines.append(f"level-2 success rate: {np.mean(task_scores_level_2):.2f} ({np.std(task_scores_level_2):.2f})")
    if prefix == "unseen_":
        # Current metric:
        # std across seed-level Level-1 / Level-2 averages.
        level_1_mean = np.mean(task_scores_level_1)
        level_1_std_across_seed_avg = np.std(task_scores_level_1)
    
        level_2_mean = np.mean(task_scores_level_2)
        level_2_std_across_seed_avg = np.std(task_scores_level_2)
    
        # New metric:
        # average of per-task std within Level-1 / Level-2.
        level_1_task_stds = [
            np.std(task_scores[task])
            for task in unseen_tasks[:13]
            if task in task_scores
        ]
    
        level_2_task_stds = [
            np.std(task_scores[task])
            for task in unseen_tasks[13:]
            if task in task_scores
        ]
    
        level_1_avg_task_std = np.mean(level_1_task_stds)
        level_2_avg_task_std = np.mean(level_2_task_stds)
    
        summary_lines.append(
            f"level-1 success rate: {level_1_mean:.2f} ({level_1_std_across_seed_avg:.2f})"
        )
        summary_lines.append(
            f"level-1 avg task std: {level_1_avg_task_std:.2f}"
        )
    
        summary_lines.append(
            f"level-2 success rate: {level_2_mean:.2f} ({level_2_std_across_seed_avg:.2f})"
        )
        summary_lines.append(
            f"level-2 avg task std: {level_2_avg_task_std:.2f}"
        )
    summary_lines.append(f"averaged success rate: {mean_averaged_success_rate:.2f} ({std_averaged_success_rate:.2f})")
    
    # Write to the output file
    with open(folder_path+f"/{prefix}summary.txt", 'w') as file:
        file.write("\n".join(summary_lines))


# exp_path="logs/hralign_eval/UnadaptedR3M2RLBench_reproduced"
# exp_path="logs/hralign_eval/UnadaptedR3M2RLBench_unfair_comparison_test"
# exp_path="logs/hralign_eval/AdaptedR3M2RLBench_unfair_comparison_test"
# exp_path="logs/hralign_eval/UnadaptedR3M2RLBench_model_3_test"
exp_path="logs/UnadaptedR3M_rvt2_bs6_lr1.25e-5"

prefix="unseen_"
# prefix="seen_"

for seed_id in range(3):
    gather_score_single(exp_path, seed=seed_id, prefix=prefix)
summarize_scores(exp_path, prefix=prefix)


