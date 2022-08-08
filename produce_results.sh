#!/bin/bash

# Activate environment
source post_processing_env/bin/activate

# Run post processing on transfer learning and multi-task learning experiments
# TODO

# Create simulator timing plots

# Create correlation plot

# Create PES plot

# Create TL 2D and 4D plot

# Create toymodel HF->UHF and LF->UHF plot

# Create multi-task learning 2D and 4D plot (LF->HF)

# Create multi-task learning 2D and 4D plot (HF->UHF)



# Create multi-fidelity sampling strategy plots
for strategy_idx in 1 3 6; do
    python3 scripts/analyse/plot_sampling_strategies.py \
    --experiment 2UHF_ICM1_ELCB${strategy_idx} --plot_sampling_strategy \
    --show_plots
done

# Create plot for average cost savings in 2D and 4D experiments


deactivate
