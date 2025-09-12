import os
import time
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Use train_ycbcr as the scheduled script. Do not change its internal default hyperparameters here,
# but override a few runtime controls to ensure correct epoch coverage and speed for repeated runs.
import train_ycbcr as train

# Number of independent runs (default: 10)
RUNS = 100
# We will collect metrics at these epochs (ensure train runs at least up to the max)
TARGET_EPOCHS = [2, 10]
OUT_DIR = "./checkpoints/std_runs"
os.makedirs(OUT_DIR, exist_ok=True)


class EvalHook:
    """Collect evaluation history from train_ycbcr (epoch -> {dataset: metrics})."""
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def __call__(self, epoch: int, results: Dict[str, Dict[str, float]]):
        # Store serializable representation
        self.history.append({"epoch": int(epoch), "results": {k: dict(v) for k, v in results.items()}})


def history_to_dataframe(history: List[Dict[str, Any]], run_id: int) -> pd.DataFrame:
    """Convert a single run history into a flat DataFrame (run_id, epoch, dataset, metrics...)."""
    rows = []
    for item in history:
        ep = item["epoch"]
        for ds, metrics in item["results"].items():
            row = {"run_id": run_id, "epoch": ep, "dataset": ds}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_train_for_repeated_runs(target_max_epoch: int, out_dir_for_run: str):
    """
    Ensure train_ycbcr will run enough epochs to include all TARGET_EPOCHS and reduce heavy IO.
    We intentionally override only runtime controls (not training loss hyperparams).
    """
    # Make sure training reaches the maximum target epoch
    train.EPOCHS = max(train.EPOCHS, int(target_max_epoch))
    # Force evaluation frequency to 1 so we capture metrics at requested epochs
    train.TEST_FREQ = 1
    # Disable saving artifacts to speed up repeated experiments
    train.SAVE_MODELS = False
    train.SAVE_IMAGES_TO_DIR = False
    train.SAVE_FREQ = 0
    # Use a per-run project directory to avoid collisions (no heavy saving since disabled)
    train.PROJECT_DIR = out_dir_for_run


def collect_runs(runs: int) -> List[Dict[str, Any]]:
    """Run train.main multiple times, injecting EvalHook to collect per-epoch metrics."""
    all_runs = []
    target_max = max(TARGET_EPOCHS) if TARGET_EPOCHS else 0
    for i in range(1, runs + 1):
        print(f"\n=== START RUN {i}/{runs} ===")
        run_out = os.path.join(OUT_DIR, f"run_{i:03d}")
        os.makedirs(run_out, exist_ok=True)

        # Prepare train module runtime settings for this run
        prepare_train_for_repeated_runs(target_max_epoch=target_max, out_dir_for_run=run_out)

        hook = EvalHook()
        train.EVAL_CALLBACK = hook  # train.main will call this during evaluation

        start_t = time.time()
        try:
            train.main()
            elapsed = time.time() - start_t
            print(f"=== RUN {i} finished in {elapsed/60:.2f} min ===")
        except Exception as e:
            # On failure, print brief message and continue (history may be empty)
            print(f"=== RUN {i} failed: {e} ===")
        # Save per-run history for later inspection
        try:
            df_run = history_to_dataframe(hook.history, i)
            if not df_run.empty:
                df_run.to_csv(os.path.join(run_out, "history.csv"), index=False)
        except Exception:
            pass

        all_runs.append({"run_id": i, "history": hook.history})
    return all_runs


def summarize_runs(all_runs: List[Dict[str, Any]], target_epochs: List[int], out_dir: str):
    """Aggregate all runs, compute mean and std per dataset & metric at target epochs, and save CSVs."""
    # Combine per-run DataFrames
    dfs = []
    for r in all_runs:
        df = history_to_dataframe(r["history"], r["run_id"])
        if not df.empty:
            dfs.append(df)
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        df_all = pd.DataFrame()

    # Save raw per-run per-epoch data
    df_all.to_csv(os.path.join(out_dir, "all_runs_metrics.csv"), index=False)

    summaries = {}
    for epoch in target_epochs:
        df_e = df_all[df_all["epoch"] == int(epoch)].copy()
        if df_e.empty:
            print(f"[WARN] No records found for epoch {epoch}")
            summaries[epoch] = pd.DataFrame()
            continue

        # Identify metric columns (exclude run_id, epoch, dataset)
        metric_cols = [c for c in df_e.columns if c not in ("run_id", "epoch", "dataset")]
        rows = []
        datasets = sorted(df_e["dataset"].unique())
        for ds in datasets:
            sub = df_e[df_e["dataset"] == ds]
            row = {"dataset": ds, "n_runs": int(sub["run_id"].nunique())}
            for m in metric_cols:
                vals = sub[m].astype(float).values
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    row[f"{m}_mean"] = np.nan
                    row[f"{m}_std"] = np.nan
                else:
                    row[f"{m}_mean"] = float(np.mean(vals))
                    row[f"{m}_std"] = float(np.std(vals, ddof=0))
            rows.append(row)
        summary_df = pd.DataFrame(rows)
        summaries[epoch] = summary_df
        summary_df.to_csv(os.path.join(out_dir, f"summary_epoch_{epoch}.csv"), index=False)
        print(f"[Saved] summary_epoch_{epoch}.csv (datasets: {', '.join(datasets)})")

    # Combined table for convenience
    combined_rows = []
    for epoch, df_sum in summaries.items():
        if df_sum.empty:
            continue
        for _, r in df_sum.iterrows():
            base = {"epoch": int(epoch), "dataset": r["dataset"], "n_runs": int(r["n_runs"])}
            for k, v in r.items():
                if k in ("dataset", "n_runs"):
                    continue
                base[k] = v
            combined_rows.append(base)
    if combined_rows:
        combined_df = pd.DataFrame(combined_rows)
        combined_df.to_csv(os.path.join(out_dir, "summary_combined.csv"), index=False)
        print(f"[Saved] summary_combined.csv")

    return summaries


def main():
    print(f"Running train.main {RUNS} times, collecting metrics at epochs {TARGET_EPOCHS}")
    all_runs = collect_runs(RUNS)
    summaries = summarize_runs(all_runs, TARGET_EPOCHS, OUT_DIR)

    # Print a brief summary: Reward mean ± std per dataset for each target epoch
    for epoch in TARGET_EPOCHS:
        df = summaries.get(epoch)
        print(f"\n--- Epoch {epoch} summary (Reward mean ± std) ---")
        if df is None or df.empty:
            print("No data")
            continue
        if "Reward_mean" in df.columns and "Reward_std" in df.columns:
            for _, r in df.iterrows():
                print(f"{r['dataset']}: {r['Reward_mean']:.6f} ± {r['Reward_std']:.6f}")
        else:
            print("Reward metric not present for this epoch.")


if __name__ == "__main__":
    main()