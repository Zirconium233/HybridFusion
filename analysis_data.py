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


def find_latest_run_dir(root="./checkpoints/search"):
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


# 新增：与 search.py/ train_ycbcr 保持一致的默认数据集权重
DEFAULT_DATASET_WEIGHTS = {
    "MSRS": 0.70,
    "M3FD": 0.06,
    "RS":   0.06,
    "PET":  0.06,
    "SPECT":0.06,
    "CT":   0.06,
}


def make_heatmap(df_epoch: pd.DataFrame, x_key: str, y_key: str, value_key: str, out_pdf: str, title: str = "", dataset_weights: Dict[str, float] = None):
    # Average over other dims and datasets (weighted)
    keep_cols = [x_key, y_key, "dataset", value_key]
    for c in ["max_ratio", "grad_ratio", "consist_ratio", "ssim_shared"]:
        if c not in keep_cols and c in df_epoch.columns:
            keep_cols.append(c)
    df_small = df_epoch[keep_cols].copy()

    # Apply dataset weights (use provided or default)
    weights = dataset_weights or DEFAULT_DATASET_WEIGHTS
    df_small["__w__"] = df_small["dataset"].map(weights).fillna(0.0)
    df_small["__vw__"] = df_small[value_key].astype(float) * df_small["__w__"]

    g = (
        df_small.groupby([x_key, y_key])[["__vw__", "__w__"]]
        .sum()
        .reset_index()
    )
    # Avoid division by zero: where total weight is zero, fallback to simple average
    g["__w__"] = g["__w__"].replace(0, pd.NA)
    g[value_key] = g["__vw__"] / g["__w__"]
    # If any rows have NaN due to zero weight, try unweighted mean fallback
    if g[value_key].isna().any():
        fallback = (
            df_small.groupby([x_key, y_key])[value_key]
            .mean()
            .reset_index(name=f"{value_key}_fallback")
        )
        g = g.merge(fallback, on=[x_key, y_key], how="left")
        g[value_key] = g[value_key].fillna(g[f"{value_key}_fallback"])
        g = g.drop(columns=[f"{value_key}_fallback"])

    g = g.drop(columns=["__vw__", "__w__"]).dropna(subset=[value_key])

    pivot = g.pivot(index=y_key, columns=x_key, values=value_key)
    try:
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    except Exception:
        pass

    # 动态注释字体与图像尺寸微调（适应大/小网格）
    ncols = max(1, pivot.shape[1])
    nrows = max(1, pivot.shape[0])
    fig_w = max(8, 0.9 * ncols)
    fig_h = max(4.5, 0.55 * nrows)
    annot_font = max(5, int(80 / max(1, ncols * nrows)**0.5))

    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        annot=True,
        annot_kws={"fontsize": annot_font},
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": value_key},
        linewidths=0.4,
        linecolor="white",
    )
    plt.title(title or f"{value_key} heatmap: {y_key} vs {x_key}")
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
    """
    返回长度为 n 的位置序列，使得第一个分配位置为中心，随后向两侧扩散：
    例如 n=5 -> [2,3,1,4,0] (0-based index), n=4 -> [1,2,0,3]
    便于把最高分的元素放在中心附近。
    """
    if n <= 0:
        return []
    mid = (n - 1) // 2
    pos = [mid]
    step = 1
    while len(pos) < n:
        right = mid + step
        if right < n:
            pos.append(right)
        left = mid - step
        if left >= 0 and len(pos) < n:
            pos.append(left)
        step += 1
    return pos


def order_keys_center(keys, score_map):
    """
    keys: 可迭代的 key 列表
    score_map: dict key->score 用于排序优先级（score 越高越优先放到中心）
    返回重新排列的 keys 列表，使高分项靠近中心位置。
    """
    keys = list(keys)
    n = len(keys)
    if n <= 1:
        return keys
    # 计算每个 key 的分数，缺失的视为最低
    scores = {k: float(score_map.get(k, -1e9)) for k in keys}
    # 按分数降序排列 keys
    sorted_keys = sorted(keys, key=lambda k: scores[k], reverse=True)
    # 生成中心扩散位置
    positions = center_positions(n)
    # 结果占位
    result = [None] * n
    for i, k in enumerate(sorted_keys):
        pos = positions[i]
        result[pos] = k
    # 若有 None（很少见），用剩余 keys 填充
    if any(x is None for x in result):
        remaining = [k for k in keys if k not in result]
        it = iter(remaining)
        for i in range(n):
            if result[i] is None:
                result[i] = next(it)
    return result

def _reposition_top_toward_center(keys, score_series, n_move=2):
    """
    将得分最高的 n_move 个 key 尽量靠近中心（保持其它顺序相对稳定）。
    """
    keys = list(keys)
    if len(keys) <= 2 or n_move <= 0:
        return keys
    scores = {k: float(score_series.get(k, -1e9)) for k in keys}
    top_keys = sorted(keys, key=lambda k: scores[k], reverse=True)[:n_move]
    n = len(keys)
    center = (n - 1) // 2
    target_positions = [center]
    step = 1
    while len(target_positions) < len(top_keys):
        if center + step < n:
            target_positions.append(center + step)
        if len(target_positions) >= len(top_keys):
            break
        if center - step >= 0:
            target_positions.append(center - step)
        step += 1
    remaining = [k for k in keys if k not in top_keys]
    new_order = [None] * n
    ri = 0
    for i in range(n):
        if i in target_positions:
            continue
        if ri < len(remaining):
            new_order[i] = remaining[ri]; ri += 1
    for tk, pos in zip(top_keys, target_positions):
        new_order[pos] = tk
    for i in range(n):
        if new_order[i] is None:
            if ri < len(remaining):
                new_order[i] = remaining[ri]; ri += 1
    return new_order

def _axis_strength(mat: pd.DataFrame, axis: int = 0, w_mean: float = 0.5, w_max: float = 0.5) -> pd.Series:
    """
    计算每个行/列的强度：mean 与 max 的加权，避免“平均高但峰值不高”或“峰值高但整体弱”。
    axis=0 -> 按列统计；axis=1 -> 按行统计
    """
    if axis == 0:
        m_mean = mat.mean(axis=0).fillna(-1e9)
        m_max  = mat.max(axis=0).fillna(-1e9)
    else:
        m_mean = mat.mean(axis=1).fillna(-1e9)
        m_max  = mat.max(axis=1).fillna(-1e9)
    return w_mean * m_mean + w_max * m_max

def pair_label(prefix_a: str, va, prefix_b: str, vb):
    # Paper-friendly label without underscores, short and clean
    return f"{prefix_a}{va}, {prefix_b}{vb}"


def pairpair_heatmap(
    df_epoch: pd.DataFrame,
    out_pdf: str,
    value_key: str = "Reward",
    # 默认仍然用 (consist, ssim) 与 (max, grad) 作为两个 pair，
    # 但我们会把 x_pair 放到行（index），y_pair 放到列（columns），
    # 以便让之前的竖向高亮变为横向高亮；同时对行列做居中排序。
    x_pair=("consist_ratio", "ssim_shared"),
    y_pair=("max_ratio", "grad_ratio"),
    title: str = "",
    annotate: bool = True,
    dataset_weights: Dict[str, float] = None,
):
    cols_needed = list(x_pair) + list(y_pair) + [value_key]
    df4 = df_epoch[cols_needed + ["dataset"]].copy()

    # Weighted aggregation over datasets (use provided or default)
    weights = dataset_weights or DEFAULT_DATASET_WEIGHTS
    df4["__w__"] = df4["dataset"].map(weights).fillna(0.0)
    df4["__vw__"] = df4[value_key].astype(float) * df4["__w__"]
    agg = (
        df4.groupby(list(x_pair) + list(y_pair))[["__vw__", "__w__"]]
        .sum()
        .reset_index()
    )
    agg["__w__"] = agg["__w__"].replace(0, pd.NA)
    agg[value_key] = agg["__vw__"] / agg["__w__"]

    # fallback to unweighted mean where necessary
    if agg[value_key].isna().any():
        fallback = (
            df4.groupby(list(x_pair) + list(y_pair))[value_key]
            .mean()
            .reset_index(name=f"{value_key}_fallback")
        )
        agg = agg.merge(fallback, on=list(x_pair) + list(y_pair), how="left")
        agg[value_key] = agg[value_key].fillna(agg[f"{value_key}_fallback"])
        agg = agg.drop(columns=[f"{value_key}_fallback"])

    df4g = agg.drop(columns=["__vw__", "__w__"]).dropna(subset=[value_key])

    # Build pair keys with readable labels
    df4g["x_key"] = df4g.apply(lambda r: pair_label("C=", r[x_pair[0]], "S=", r[x_pair[1]]), axis=1)
    df4g["y_key"] = df4g.apply(lambda r: pair_label("M=", r[y_pair[0]], "G=", r[y_pair[1]]), axis=1)

    # Compute average scores per x_key and y_key for ordering
    col_scores = df4g.groupby("y_key")[value_key].mean().to_dict()   # will become columns
    row_scores = df4g.groupby("x_key")[value_key].mean().to_dict()   # will become rows

    # Order keys so that high-score keys尽量居中
    x_keys_ordered = order_keys_center(sorted(df4g["x_key"].unique()),
                                       df4g.groupby("x_key")[value_key].mean().to_dict())  # rows
    y_keys_ordered = order_keys_center(sorted(df4g["y_key"].unique()),
                                       df4g.groupby("y_key")[value_key].mean().to_dict())  # cols

    # 初始矩阵（行 = x_key，列 = y_key）
    mat = df4g.pivot(index="x_key", columns="y_key", values=value_key).reindex(index=x_keys_ordered, columns=y_keys_ordered)

    # 双轴更激进的“向中心靠拢”：按强度选前30%一起往中间放置，做两次细化
    try:
        for _ in range(2):  # 两次细化
            # 列（y轴）强度
            col_strength = _axis_strength(mat, axis=0, w_mean=0.5, w_max=0.5)
            n_cols_move = max(2, int(round(len(y_keys_ordered) * 0.30)))
            y_keys_ordered = _reposition_top_toward_center(y_keys_ordered, col_strength, n_move=n_cols_move)

            # 行（x轴）强度
            row_strength = _axis_strength(mat, axis=1, w_mean=0.5, w_max=0.5)
            n_rows_move = max(2, int(round(len(x_keys_ordered) * 0.30)))
            x_keys_ordered = _reposition_top_toward_center(x_keys_ordered, row_strength, n_move=n_rows_move)

            # 重新索引
            mat = mat.reindex(index=x_keys_ordered, columns=y_keys_ordered)
    except Exception:
        pass

    # 调大注释字体且左右留白（不贴边）
    fig_w = max(10, 0.9 * len(y_keys_ordered))
    fig_h = max(7, 0.52 * len(x_keys_ordered))
    annot_font = max(9, int(95 / max(1, len(x_keys_ordered) * len(y_keys_ordered))**0.5))

    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        mat,
        annot=annotate,
        annot_kws={"fontsize": annot_font},
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": value_key},
        linewidths=0.4,
        linecolor="white",
        square=False,
    )
    plt.title(title or f"{value_key} (epoch) | ({x_pair[0]},{x_pair[1]}) on rows vs ({y_pair[0]},{y_pair[1]}) on columns")
    plt.xlabel(f"{y_pair[0]}, {y_pair[1]}")
    plt.ylabel(f"{x_pair[0]}, {x_pair[1]}")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    # 左右留一定间距，避免贴边
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.18)
    plt.savefig(out_pdf, dpi=300, format="pdf")
    plt.close()
    return mat, y_keys_ordered, x_keys_ordered


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
        help='Input: all_results.pth / run_dir / ranking_epoch*.csv. "last" -> latest run under ./checkpoints/search'
    )
    parser.add_argument("-e", "--epoch", type=int, default=2, help="Epoch to analyze (default 2)")
    parser.add_argument("-o", "--out_dir", default=None, help="Output dir, default run_dir/analysis_plus")
    parser.add_argument("-m", "--metric", default="Reward", help="Metric field to use (default Reward)")
    parser.add_argument("--no_annot", action="store_true", help="Disable value annotation in heatmaps")
    args = parser.parse_args()

    # Resolve input
    if args.input == "last":
        run_dir = find_latest_run_dir("./checkpoints/search")
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
    # Score = (VIF + 1.5*Qabf + SSIM) / 4
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

    # Dataset weights: prefer DEFAULT_DATASET_WEIGHTS but restrict to datasets present
    ds_unique = df_e["dataset"].unique().tolist() if "dataset" in df_e.columns else []
    if ds_unique:
        dataset_weights = {ds: DEFAULT_DATASET_WEIGHTS.get(ds, 0.0) for ds in ds_unique}
        # Normalize so weights sum to 1 across present datasets (keeps relative importance)
        ssum = sum(dataset_weights.values())
        if ssum > 0:
            dataset_weights = {k: v / ssum for k, v in dataset_weights.items()}
    else:
        dataset_weights = DEFAULT_DATASET_WEIGHTS.copy()

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