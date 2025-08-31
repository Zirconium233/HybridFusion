import os
import re
import argparse
import json
import torch
import pandas as pd
import numpy as np
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

# Use paper-friendly English fonts (fallback to DejaVu if Times is missing)
# plt.rcParams["font.family"] = "Times New Roman, DejaVu Serif"
plt.rcParams["axes.unicode_minus"] = True


def find_latest_run_dir(root="./checkpoints/search_method2"):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dir not found: {root}")
    subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subdirs:
        raise FileNotFoundError(f"No subdirs in: {root}")
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for d in subdirs:
        if os.path.basename(d).startswith("run_"):
            return d
    return subdirs[0]


def _load_ranking_csv(path: str):
    # Expect columns: max_ratio,grad_ratio,consist_ratio,ssim_shared,Reward
    df = pd.read_csv(path)
    required = {"max_ratio", "grad_ratio", "consist_ratio", "ssim_shared", "Reward"}
    if not required.issubset(df.columns):
        raise ValueError(f"ranking csv missing columns, need {required}, got {set(df.columns)}")
    m = re.search(r"epoch(\d+)", os.path.basename(path))
    epoch = int(m.group(1)) if m else -1
    df["epoch"] = epoch
    df["dataset"] = "AVG"
    df["exp_id"] = np.arange(len(df))
    params_df = df[["exp_id", "max_ratio", "grad_ratio", "consist_ratio", "ssim_shared"]].copy()
    metrics_df = df[["exp_id", "epoch", "dataset", "Reward"]].copy()
    return params_df, metrics_df


def load_results(input_path: str):
    """
    Return params_df, metrics_df
    - input can be:
      - path/to/all_results.pth
      - run dir containing params.csv/metrics.csv or all_results.pth
      - ranking_epochX.csv (aggregated across datasets)
    """
    if os.path.isfile(input_path):
        if input_path.endswith(".pth"):
            obj = torch.load(input_path, map_location="cpu", weights_only=False)
            return obj["params"].copy(), obj["metrics"].copy()
        if os.path.basename(input_path).startswith("ranking_epoch") and input_path.endswith(".csv"):
            return _load_ranking_csv(input_path)
        raise FileNotFoundError(f"Unsupported file: {input_path}")

    if os.path.isdir(input_path):
        pth = os.path.join(input_path, "all_results.pth")
        if os.path.isfile(pth):
            return load_results(pth)
        params_csv = os.path.join(input_path, "params.csv")
        metrics_csv = os.path.join(input_path, "metrics.csv")
        if os.path.isfile(params_csv) and os.path.isfile(metrics_csv):
            return pd.read_csv(params_csv), pd.read_csv(metrics_csv)
        raise FileNotFoundError(f"params.csv/metrics.csv not found in {input_path}")

    raise FileNotFoundError(f"Cannot load: {input_path}")


def ensure_output_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def merge_params_metrics(params_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.merge(params_df, on="exp_id", how="left")
    if "ssim_shared" not in df.columns:
        if "ssim_ratio" in df.columns and "ssim_ir_ratio" in df.columns:
            df["ssim_shared"] = (df["ssim_ratio"].astype(float) + df["ssim_ir_ratio"].astype(float)) / 2.0
        else:
            df["ssim_shared"] = np.nan
    return df


def make_heatmap(df_epoch: pd.DataFrame, x_key: str, y_key: str, value_key: str, out_pdf: str, title: str = "", dataset_weights: Dict[str, float] = None):
    # Average over other dims and datasets (weighted)
    keep_cols = [x_key, y_key, "dataset", value_key]
    for c in ["max_ratio", "grad_ratio", "consist_ratio", "ssim_shared"]:
        if c not in keep_cols and c in df_epoch.columns:
            keep_cols.append(c)
    df_small = df_epoch[keep_cols].copy()

    # Apply dataset weights
    weights = dataset_weights or {}
    df_small["__w__"] = df_small["dataset"].map(weights).fillna(0.0)
    df_small["__vw__"] = df_small[value_key].astype(float) * df_small["__w__"]

    g = (
        df_small.groupby([x_key, y_key])[["__vw__", "__w__"]]
        .sum()
        .reset_index()
    )
    # Avoid division by zero
    g[value_key] = g["__vw__"] / g["__w__"].replace(0, pd.NA)
    g = g.drop(columns=["__vw__", "__w__"]).dropna(subset=[value_key])

    pivot = g.pivot(index=y_key, columns=x_key, values=value_key)
    try:
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    except Exception:
        pass

    # Wider figure and smaller annotations for paper
    plt.figure(figsize=(max(8, 0.9 * len(pivot.columns)), max(4.5, 0.6 * len(pivot.index))))
    sns.heatmap(
        pivot,
        annot=True,
        annot_kws={"fontsize": 7},
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": value_key},
        linewidths=0.4,
        linecolor="white",
    )
    plt.title(title or f"{value_key} heatmap: {y_key} vs {x_key}")
    # Remove underscores in axis labels for paper style
    xlabel = x_key.replace("_", " ")
    ylabel = y_key.replace("_", " ")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, format="pdf")
    plt.close()
    return pivot


def center_positions(n: int):
    center = (n - 1) / 2.0
    return [i for i in sorted(range(n), key=lambda i: (abs(i - center), i))]


def order_keys_center(keys, score_map):
    keys_sorted = sorted(keys, key=lambda k: score_map.get(k, -1e9), reverse=True)
    n = len(keys_sorted)
    out = [None] * n
    pos_order = center_positions(n)
    for idx, k in enumerate(keys_sorted):
        out[pos_order[idx]] = k
    for i in range(n):
        if out[i] is None:
            out[i] = keys_sorted[min(i, len(keys_sorted) - 1)]
    return out


def pair_label(prefix_a: str, va, prefix_b: str, vb):
    # Paper-friendly label without underscores, short and clean
    return f"{prefix_a}{va}, {prefix_b}{vb}"


def pairpair_heatmap(
    df_epoch: pd.DataFrame,
    out_pdf: str,
    value_key: str = "Reward",
    # Swap axes: put (consist, ssim) on X (columns), (max, grad) on Y (rows)
    x_pair=("consist_ratio", "ssim_shared"),
    y_pair=("max_ratio", "grad_ratio"),
    title: str = "",
    annotate: bool = True,
    dataset_weights: Dict[str, float] = None,
):
    cols_needed = list(x_pair) + list(y_pair) + [value_key]
    df4 = df_epoch[cols_needed + ["dataset"]].copy()

    # Weighted aggregation over datasets
    weights = dataset_weights or {}
    df4["__w__"] = df4["dataset"].map(weights).fillna(0.0)
    df4["__vw__"] = df4[value_key].astype(float) * df4["__w__"]
    agg = (
        df4.groupby(list(x_pair) + list(y_pair))[["__vw__", "__w__"]]
        .sum()
        .reset_index()
    )
    agg[value_key] = agg["__vw__"] / agg["__w__"].replace(0, pd.NA)
    df4g = agg.drop(columns=["__vw__", "__w__"]).dropna(subset=[value_key])

    # Build pair keys with paper-friendly labels
    df4g["x_key"] = df4g.apply(lambda r: pair_label("C=", r[x_pair[0]], "S=", r[x_pair[1]]), axis=1)
    df4g["y_key"] = df4g.apply(lambda r: pair_label("M=", r[y_pair[0]], "G=", r[y_pair[1]]), axis=1)

    # Column/row average scores to center best in the middle
    col_scores = df4g.groupby("x_key")[value_key].mean().to_dict()
    row_scores = df4g.groupby("y_key")[value_key].mean().to_dict()

    x_keys = order_keys_center(sorted(df4g["x_key"].unique()), col_scores)
    y_keys = order_keys_center(sorted(df4g["y_key"].unique()), row_scores)

    mat = df4g.pivot(index="y_key", columns="x_key", values=value_key).reindex(index=y_keys, columns=x_keys)

    # Wider than tall
    fig_w = max(10, 0.8 * len(x_keys))
    fig_h = max(8, 0.45 * len(y_keys))
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        mat,
        annot=annotate,
        annot_kws={"fontsize": 6.8},
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": value_key},
        linewidths=0.4,
        linecolor="white",
    )
    # Clean English title/labels for paper
    plt.title(title or f"{value_key} at selected epoch | (max, grad) on rows vs (consist, ssim) on columns")
    plt.xlabel("consistency ratio, SSIM ratio (columns)")
    plt.ylabel("max ratio, grad ratio (rows)")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=400, format="pdf")
    plt.close()
    return mat, x_keys, y_keys


def find_global_best(df_epoch: pd.DataFrame, value_key: str = "Reward", dataset_weights: Dict[str, float] = None):
    keys = ["max_ratio", "grad_ratio", "consist_ratio", "ssim_shared"]
    missing = [k for k in keys if k not in df_epoch.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # Weighted aggregate over datasets
    weights = dataset_weights or {}
    dfw = df_epoch.copy()
    dfw["__w__"] = dfw["dataset"].map(weights).fillna(0.0)
    dfw["__vw__"] = dfw[value_key].astype(float) * dfw["__w__"]

    g = (
        dfw.groupby(keys)[["__vw__", "__w__"]]
        .sum()
        .reset_index()
    )
    g[value_key] = g["__vw__"] / g["__w__"].replace(0, pd.NA)
    g = g.drop(columns=["__vw__", "__w__"]).dropna(subset=[value_key])
    g = g.sort_values(by=value_key, ascending=False)
    return g.iloc[0].to_dict(), g


def main():
    parser = argparse.ArgumentParser(description="Analyze search results and draw 2D/2x2 heatmaps (paper-ready)")
    parser.add_argument(
        "-i", "--input",
        default="last",
        help='Input: all_results.pth / run_dir / ranking_epoch*.csv. "last" -> latest run under ./checkpoints/search_method2'
    )
    parser.add_argument("-e", "--epoch", type=int, default=2, help="Epoch to analyze (default 2)")
    parser.add_argument("-o", "--out_dir", default=None, help="Output dir, default run_dir/analysis_plus")
    parser.add_argument("-m", "--metric", default="Reward", help="Metric field to use (default Reward)")
    parser.add_argument("--no_annot", action="store_true", help="Disable value annotation in heatmaps")
    args = parser.parse_args()

    # Resolve input
    if args.input == "last":
        run_dir = find_latest_run_dir("./checkpoints/search_method2")
        params_df, metrics_df = load_results(run_dir)
    else:
        if os.path.isfile(args.input) and args.input.endswith(".csv") and os.path.basename(args.input).startswith("ranking_epoch"):
            params_df, metrics_df = _load_ranking_csv(args.input)
            run_dir = os.path.dirname(os.path.dirname(args.input))
        else:
            params_df, metrics_df = load_results(args.input)
            run_dir = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)

    df = merge_params_metrics(params_df, metrics_df)

    # Filter epoch
    if "epoch" in df.columns and args.epoch is not None and args.epoch >= 0:
        df_e = df[df["epoch"] == args.epoch].copy()
    else:
        df_e = df.copy()

    if df_e.empty:
        raise RuntimeError(f"No records for epoch={args.epoch}. Avail epochs: {sorted(df['epoch'].unique().tolist()) if 'epoch' in df.columns else 'N/A'}")

    out_dir = args.out_dir or os.path.join(run_dir, "analysis_plus")
    ensure_output_dir(out_dir)

    # Compute new composite metric per record:
    # Score = (VIF + 1.5*Qabf + SSIM + MI/4) / 4
    def _to_float(series_or_val, idx):
        s = df_e.get(series_or_val, None) if isinstance(series_or_val, str) else series_or_val
        if s is None:
            return pd.Series(0.0, index=idx)
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    idx = df_e.index
    vif = _to_float("VIF", idx)
    qabf = _to_float("Qabf", idx)
    ssim = _to_float("SSIM", idx)
    mi = _to_float("MI", idx)  # if MI missing, treated as 0
    df_e["Score"] = (vif + 1.5 * qabf + ssim + mi / 5.0) / 4.0

    # Dataset weights: MSRS -> 0.6, others (assume two) -> 0.2 each
    ds_unique = df_e["dataset"].unique().tolist() if "dataset" in df_e.columns else []
    dataset_weights = {ds: (0.5 if ds == "MSRS" else 0.25) for ds in ds_unique}

    # Use Score as the analysis metric
    metric = "Score"

    # 1) Three classic 2D heatmaps (weighted over datasets) -> save as PDF
    pairs = [
        ("max_ratio", "grad_ratio"),
        ("max_ratio", "consist_ratio"),
        ("grad_ratio", "consist_ratio"),
    ]
    saved_figs = []
    for x, y in pairs:
        pdf = os.path.join(out_dir, f"heatmap_{y}-vs-{x}_epoch{args.epoch}.pdf")
        title = f"{metric} (epoch {args.epoch}) | {y} vs {x} | weighted over datasets"
        _ = make_heatmap(df_e, x_key=x, y_key=y, value_key=metric, out_pdf=pdf, title=title, dataset_weights=dataset_weights)
        saved_figs.append(pdf)

    # 2) Big pair-pair heatmap with swapped axes and paper-friendly labels -> PDF (weighted)
    big_pdf = os.path.join(out_dir, f"heatmap_pairpair_epoch{args.epoch}.pdf")
    title_big = f"{metric} (epoch {args.epoch}) | (max, grad) on rows vs (consist, ssim) on columns (weighted)"
    _, x_keys, y_keys = pairpair_heatmap(
        df_e,
        out_pdf=big_pdf,
        value_key=metric,
        x_pair=("consist_ratio", "ssim_shared"),
        y_pair=("max_ratio", "grad_ratio"),
        title=title_big,
        annotate=(not args.no_annot),
        dataset_weights=dataset_weights,
    )
    saved_figs.append(big_pdf)

    # 3) Global best over 4D (weighted over datasets)
    best_cfg, table_all = find_global_best(df_e, value_key=metric, dataset_weights=dataset_weights)
    best_json = {
        "epoch": int(args.epoch),
        "metric": metric,
        "best": best_cfg,
        "images": saved_figs,
        "x_keys_order": x_keys,
        "y_keys_order": y_keys,
    }
    with open(os.path.join(out_dir, f"best_epoch{args.epoch}.json"), "w") as f:
        json.dump(best_json, f, indent=2)

    # Save ranking table for this epoch (PDF plots already saved)
    table_path = os.path.join(out_dir, f"ranking_full_epoch{args.epoch}.csv")
    table_all.sort_values(by=metric, ascending=False).to_csv(table_path, index=False)

    print(f"[analysis] out_dir: {out_dir}")
    print("Saved figures (PDF):")
    for p in saved_figs:
        print(f" - {p}")
    print(f"\nGlobal best (weighted over datasets) @ epoch={args.epoch}: {best_cfg}")
    print(f"Full ranking saved to: {table_path}")


if __name__ == "__main__":
    main()