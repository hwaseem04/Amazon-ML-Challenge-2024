#!/bin/bash

# List of start and part values
starts=(0 5051 10101 15151 20201 25251 30301 35351 40401 45451 50501 55551 60601 65651 70701 75751 80801 85851 90901 95951 101001 106051 111101 116151 121201 126251)
parts=(5050 10100 15150 20200 25250 30300 35350 40400 45450 50500 55550 60600 65650 70700 75750 80800 85850 90900 95950 101000 106050 111100 116150 121200 126250 131188)

# List of available GPUs
gpus=(0 1 2 4 6 7)

# Initialize array to keep track of how many tmux sessions are assigned to each GPU
gpu_sessions=(0 0 0 0 0 0)

# Loop through the indices of the arrays
for i in "${!starts[@]}"; do
  # Find the first GPU that has less than 5 tmux sessions assigned to it
  for gpu_idx in "${!gpus[@]}"; do
    if [ "${gpu_sessions[$gpu_idx]}" -lt 5 ]; then
      gpu_id=${gpus[$gpu_idx]}
      gpu_sessions[$gpu_idx]=$((gpu_sessions[$gpu_idx] + 1))
      break
    fi
  done

  # Create a tmux session name like "part1", "part2", etc.
  session_name="part$((i + 100))"
  
  # Start a new tmux session, detached
  tmux new-session -d -s "$session_name"
  
  # Send the command to set CUDA_VISIBLE_DEVICES and run the Python script
  tmux send-keys -t "$session_name" "export CUDA_VISIBLE_DEVICES=$gpu_id" C-m
  tmux send-keys -t "$session_name" "conda activate ugrip" C-m
  tmux send-keys -t "$session_name" "python3 inference.py --s ${starts[i]} --part ${parts[i]}" C-m
done

echo "All tmux sessions have been created with GPU assignment."
