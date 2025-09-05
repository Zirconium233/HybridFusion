import os
import itertools
import time
import datetime as dt
from typing import Dict, Any, List

import torch
import pandas as pd
import numpy as np
# Add the current directory to the import path (can be ignored when running in Image/ directory with VS Code)
import sys
CUR_DIR = os.path.dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

import train_ycbcr as train  # Directly call method2.main and implement parameter sweeping by overwriting its global variables

# Common global controls (can be overwritten by command line or upper-level scripts)
EPOCHS = 4              # Fix to 4 epochs each time (focus on epochs 1/2)
TEST_FREQ = 1           # Evaluate once per epoch for easy tracking of epochs 1/2
PROJECT_ROOT = "./checkpoints/search"
METRIC_MODE = "mu"      # Uniformly use mean weight for evaluation, aligned with your settings in method2.py
DATASET_TRAIN = "MSRS"  # method2 is fixed to use MSRS for training

# Whether to run the full 240 parameter combinations. Set to False for a quick small-grid verification process
RUN_FULL = True

# Full search range (approximately 4×5×3×4 = 240 combinations)
# - max_ratio:      [1, 4, 16, 40]
# - grad_ratio:     [2, 32, 40, 64, 90]
# - ssim_shared:    [2, 10, 32]  -> Assign to both ssim_ratio and ssim_ir_ratio simultaneously
# - consist_ratio:  [1, 12, 24, 40]
# Keep other parameters as default: ir_compose=2.0, color_ratio=2.0, window=48, max_mode="l1", consist_mode="l1"
FULL_GRID = {
    # Slightly increase some default values to match the strong loss configuration in train_ycbcr
    "max_ratio":     [10.0, 4.0, 16.0, 40.0],
    "grad_ratio":    [2.0, 32.0, 40.0, 64.0, 90.0],
    # Use the lower initial ssim value (1.0) from train_ycbcr
    "ssim_shared":   [1.0, 10.0, 32.0],
    # Adjust the minimum value from 1.0 to 2.0 (closer to the conservative value you recommended)
    "consist_ratio": [2.0, 12.0, 24.0, 40.0],
}

# Quick small-range verification (approximately 2×2×1×2 = 4 combinations)
QUICK_GRID = {
    "max_ratio":     [16.0],
    "grad_ratio":    [32.0, 64.0],
    "ssim_shared":   [10.0],
    "consist_ratio": [12.0],
}

SELECTED_GRID = FULL_GRID if RUN_FULL else QUICK_GRID

# Other fixed weights for each experiment (can be included in the grid if needed)
FIXED_LOSS_KW = {
    "ir_compose": 2.0,
    "color_ratio": 2.0,
    "ssim_window_size": 48,
    "max_mode": "l1",
    "consist_mode": "l1",
}

# Record structure: Each experiment exp_id -> {cfg, history(list[{'epoch':int,'results':{dataset:metrics}}])}
class EvalHook:
    def __init__(self):
        self.history = []
    
    def __call__(self, epoch: int, results: Dict[str, Dict[str, float]]):
        # Append epoch-wise evaluation results to history (convert nested dicts to regular dicts for serialization)
        self.history.append({"epoch": int(epoch), "results": {k: dict(v) for k, v in results.items()}})

def set_method2_globals(cfg: Dict[str, Any], exp_dir: str):
    """Overwrite global variables of method2 based on the configuration cfg"""
    # Training and evaluation controls
    train.EPOCHS = cfg.get("epochs", EPOCHS)
    train.TEST_FREQ = cfg.get("test_freq", TEST_FREQ)
    train.METRIC_MODE = cfg.get("metric_mode", METRIC_MODE)
    train.PROJECT_DIR = exp_dir
    # Disable saving to speed up parameter sweeping
    train.SAVE_IMAGES_TO_DIR = False
    train.SAVE_MODELS = False
    train.SAVE_FREQ = 0
    # Overwrite FusionLoss weights
    train.LOSS_MAX_RATIO = float(cfg["max_ratio"])
    train.LOSS_CONSIST_RATIO = float(cfg["consist_ratio"])
    train.LOSS_GRAD_RATIO = float(cfg["grad_ratio"])
    # Prioritize ssim_shared (assign to both ssim_ratio and ssim_ir_ratio), fallback to independent fields if not present
    if "ssim_shared" in cfg:
        shared = float(cfg["ssim_shared"])
        train.LOSS_SSIM_RATIO = shared
        train.LOSS_SSIM_IR_RATIO = shared
    else:
        train.LOSS_SSIM_RATIO = float(cfg["ssim_ratio"])
        train.LOSS_SSIM_IR_RATIO = float(cfg["ssim_ir_ratio"])
    # Apply fixed loss parameters
    train.LOSS_IR_COMPOSE = float(FIXED_LOSS_KW["ir_compose"])
    train.LOSS_COLOR_RATIO = float(FIXED_LOSS_KW["color_ratio"])
    train.LOSS_SSIM_WINDOW = int(FIXED_LOSS_KW["ssim_window_size"])
    train.LOSS_MAX_MODE = FIXED_LOSS_KW["max_mode"]
    train.LOSS_CONSIST_MODE = FIXED_LOSS_KW["consist_mode"]

def grid_iter(grid: Dict[str, List[Any]]):
    """Generate all combinations of parameters from the grid (Cartesian product)"""
    keys = list(grid.keys())
    # Iterate over all possible value combinations of the grid parameters
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))

def run_one_experiment(exp_id: int, cfg: Dict[str, Any], out_root: str):
    """Run a single experiment with the given experiment ID, configuration, and output root directory"""
    # Create experiment-specific directory (format exp_001, exp_002, etc.)
    exp_dir = os.path.join(out_root, f"exp_{exp_id:03d}")
    os.makedirs(exp_dir, exist_ok=True)
    # Overwrite method2's global variables with the current experiment configuration
    set_method2_globals(cfg, exp_dir)
    # Inject evaluation callback to record training history
    hook = EvalHook()
    train.EVAL_CALLBACK = hook
    # Start experiment
    print(f"\n==== EXP {exp_id} start ====\nConfig: {cfg}\nDir: {exp_dir}")
    start = time.time()
    train.main()
    elapsed = time.time() - start
    print(f"==== EXP {exp_id} done in {elapsed/60:.2f} min ====")
    # Return experiment results (configuration, history, and elapsed time)
    return {"cfg": cfg, "history": hook.history, "elapsed_sec": elapsed}

def history_to_dataframe(exp_id: int, history: List[Dict[str, Any]]):
    """Convert experiment history (list of epoch results) to a pandas DataFrame for easy analysis"""
    rows = []
    for item in history:
        ep = item["epoch"]
        # Unpack results for each dataset
        for ds, metrics in item["results"].items():
            row = {"exp_id": exp_id, "epoch": ep, "dataset": ds}
            row.update(metrics)  # Add all metrics (e.g., VIF, Qabf, SSIM, Reward) to the row
            rows.append(row)
    return pd.DataFrame(rows)

# New: Dataset weights for evaluation (70% for MSRS, 30% split equally among the other 5 datasets)
DATASET_WEIGHTS = {
    "MSRS": 0.70,
    "M3FD": 0.06,
    "RS":   0.06,
    "PET":  0.06,
    "SPECT":0.06,
    "CT":   0.06,
}
DATASET_LIST = list(DATASET_WEIGHTS.keys())

def summarize_topk(df_all: pd.DataFrame, params_table: pd.DataFrame, epoch_target: int, topk: int = 10):
    """
    Summarize the top-k experiments at the target epoch, sorted by weighted average Reward.
    Args:
        df_all: DataFrame containing all evaluation metrics
        params_table: DataFrame containing experiment parameters
        epoch_target: Target epoch to summarize (e.g., 1 or 2)
        topk: Number of top experiments to return
    Returns:
        DataFrame of top-k experiments with parameters and key metrics
    """
    # Filter data to the target epoch
    df_e = df_all[df_all["epoch"] == epoch_target].copy()
    if df_e.empty:
        return pd.DataFrame()
    
    # Pivot table: Rows = exp_id, Columns = (metric, dataset), Values = metric values
    piv = df_e.pivot_table(index="exp_id", columns="dataset",
                           values=["Reward", "VIF", "Qabf", "SSIM"], aggfunc="mean")
    
    # Calculate weighted average Reward for each experiment (handle missing datasets by re-normalizing weights)
    def weighted_reward_row(row):
        total_reward = 0.0
        total_weight = 0.0
        for ds in DATASET_LIST:
            reward_key = ("Reward", ds)
            # Only include datasets with valid (non-NaN) Reward values
            if reward_key in row.index and not np.isnan(row[reward_key]):
                weight = DATASET_WEIGHTS.get(ds, 0.0)
                total_reward += float(row[reward_key]) * weight
                total_weight += weight
        # Return weighted average (or NaN if no valid datasets)
        return (total_reward / total_weight) if (total_weight > 0) else np.nan
    
    # Add weighted average Reward as a new column
    piv[("Reward", "MeanOverSets")] = piv.apply(weighted_reward_row, axis=1)
    # Sort experiments by weighted average Reward (descending order)
    piv = piv.sort_values(by=("Reward", "MeanOverSets"), ascending=False)
    
    # Flatten multi-level column names (e.g., ("Reward", "MSRS") → "Reward_MSRS")
    piv.columns = [f"{metric}_{dataset}" if dataset != "" else f"{metric}" 
                   for metric, dataset in piv.columns]
    
    # Merge parameter table with metric table (link via exp_id)
    topk_df = params_table.merge(piv.reset_index(), on="exp_id", how="right")
    return topk_df.head(topk)

def main():
    """Main function: Initialize experiment environment, run all parameter combinations, and summarize results"""
    # Create root directory for experiments
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    # Generate timestamp for the current run (format: YYYYMMDD_HHMMSS)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_ROOT, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Generate all parameter combinations from the selected grid
    configs = list(grid_iter(SELECTED_GRID))
    print(f"Total configs: {len(configs)} (RUN_FULL={RUN_FULL})")

    # Run experiments one by one
    runs = []
    for exp_idx, cfg in enumerate(configs, 1):
        # Add fixed global settings to the current configuration
        cfg = dict(cfg)
        cfg["epochs"] = EPOCHS
        cfg["metric_mode"] = METRIC_MODE
        # Run single experiment and record results
        exp_result = run_one_experiment(exp_idx, cfg, out_dir)
        runs.append(exp_result)

    # Aggregate all experiment results into DataFrames
    all_metric_rows = []
    for exp_idx, run in enumerate(runs, 1):
        # Convert experiment history to DataFrame
        metric_df = history_to_dataframe(exp_idx, run["history"])
        # Add elapsed time for the experiment
        metric_df["elapsed_sec"] = run["elapsed_sec"]
        all_metric_rows.append(metric_df)
    # Combine all metric data into a single DataFrame
    df_all_metrics = pd.concat(all_metric_rows, ignore_index=True) if all_metric_rows else pd.DataFrame()

    # Create parameter table (map exp_id to its configuration)
    params_rows = []
    for exp_idx, run in enumerate(runs, 1):
        param_row = {"exp_id": exp_idx}
        param_row.update(run["cfg"])
        params_rows.append(param_row)
    df_params = pd.DataFrame(params_rows)

    # Save results (PyTorch pth for full data, CSV for easy viewing)
    save_data = {"params": df_params, "metrics": df_all_metrics}
    torch.save(save_data, os.path.join(out_dir, "all_results.pth"))
    df_params.to_csv(os.path.join(out_dir, "params.csv"), index=False)
    df_all_metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    print(f"\n[Saved] results -> {out_dir}")

    # Print top-10 experiments for epochs 1 and 2 (focus on early convergence performance)
    for target_epoch in [1, 2]:
        top10_df = summarize_topk(df_all_metrics, df_params, epoch_target=target_epoch, topk=10)
        print(f"\n===== Top-10 @ epoch {target_epoch} (sorted by mean Reward over datasets) =====")
        if top10_df.empty:
            print("No records available.")
        else:
            # Define columns to display (prioritize key parameters first)
            # Show ssim_shared if present; otherwise show ssim_ratio/ssim_ir_ratio
            if "ssim_shared" in top10_df.columns:
                key_param_cols = ["exp_id", "max_ratio", "grad_ratio", "ssim_shared", "consist_ratio"]
            else:
                key_param_cols = ["exp_id", "max_ratio", "grad_ratio", "ssim_ratio", "ssim_ir_ratio", "consist_ratio"]
            # Keep only existing columns and append remaining metric columns
            display_cols = [col for col in key_param_cols if col in top10_df.columns] + \
                          [col for col in top10_df.columns if col not in key_param_cols]
            # Print top-10 results (hide index for readability)
            print(top10_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()