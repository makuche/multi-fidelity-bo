#!/bin/bash

APPLY_POSTPROCESSING=true
PLOT_TIMINGS_CORRELATION=true
PLOT_SAMPLING_STRATEGIES=true
PLOT_TL_RESULTS=true
PLOT_MT_TOYMODEL_RESULTS=true
PLOT_MT_RESULTS=true
PLOT_EFFICIENCY=true

# Activate environment
THESIS_DIR=$(cd $(dirname "${BASH_SOURCE:-$0}")/.. && pwd)
SCRIPTS_DIR=$THESIS_DIR/scripts
source $THESIS_DIR/post_processing_env/bin/activate
echo "Activated post-processing environment"
sleep .5

# Run post processing on transfer learning and multi-task learning experiments
APPLY_POSTPROCESSING=false
if $APPLY_POSTPROCESSING
then
    approaches=("transfer_learning" "multi_task_learning")
    for approach in "${approaches[@]}"; do
        echo "Running post-processing on $approach"
        python3 $SCRIPTS_DIR/preprocess/parse_raw_data.py --setup $approach
        echo "Done with $approach"
    done
fi

# Create simulator timing plots (page 45) and correlation plot (51),
# print simulator timings, pearson correlation coefficients and covariances
if $PLOT_TIMINGS_CORRELATION
then
    python3 $SCRIPTS_DIR/analyse/plot_correlation_statistics.py
fi

# Create PES plot

# Create TL 2D and 4D plot (57)
if $PLOT_TL_RESULTS
then
    for dim in "2D" "4D"; do
    echo "Plotting $dim transfer learning results"
        python3 $SCRIPTS_DIR/analyse/plot_TL_results_boxplot.py \
        --dimension $dim
    done
fi

# Create toymodel HF->UHF and LF->UHF plot (60,61)
if $PLOT_MT_TOYMODEL_RESULTS
then
    for fidelities in "uhf_hf" "uhf_lf"; do
    echo "Plotting toymodel for fidelity combination $fidelities"
        python3 $SCRIPTS_DIR/analyse/plot_utilities_toymodel.py \
        --fidelities $fidelities
    done
fi

# Create multi-task learning 2D and 4D plots (62,63,64)
if $PLOT_MT_RESULTS
then
    for dimension in "2D" "4D"; do
        for target_fidelity in "uhf" "hf"; do
        echo "Plotting MT results for fidelity combination $fidelities"
            python3 $SCRIPTS_DIR/analyse/plot_MT_results_boxplot.py \
            --highest_fidelity $target_fidelity --print_summary \
            --dimension $dimension --print_non_converged
        done
    done
fi

# Create multi-fidelity sampling strategy plots (72,73)
if $PLOT_SAMPLING_STRATEGIES
then
    for strategy_idx in 1 3 6; do
        echo "Plotting sampling strategy $strategy_idx"
        python3 $SCRIPTS_DIR/analyse/plot_sampling_strategies.py \
        --experiment 2UHF_ICM1_ELCB${strategy_idx} --plot_sampling_strategy
    done
fi

# TL and MT comparison (74,75)
if $PLOT_EFFICIENCY
then
    for dimension in "2D" "4D"; do
        echo "Comparing TL and MT for $dimension experiments"
        python3 $SCRIPTS_DIR/analyse/plot_efficiency.py \
        --dimension $dimension
    done
fi

# Deactivate environment
deactivate
