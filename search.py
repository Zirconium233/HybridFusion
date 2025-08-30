import os
import itertools
import time
import datetime as dt
from typing import Dict, Any, List

import torch
import pandas as pd

# 将当前目录加入 import 路径（VS Code 运行在 Image/ 下时可忽略）
import sys
CUR_DIR = os.path.dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

import train as train  # 直接调度 method2.main，通过覆盖其全局变量实现扫参

# 共同全局控制（可被命令行或上层脚本覆盖）
EPOCHS = 4              # 固定每次4 epoch（重点关注 1/2）
TEST_FREQ = 1           # 每个epoch评测一次，便于抓取1/2
PROJECT_ROOT = "./checkpoints/search_method2"
METRIC_MODE = "mu"      # 评测统一使用均值权重，和你在 method2.py 的对齐
DATASET_TRAIN = "MSRS"  # method2 已固定用 MSRS 训练

# 是否跑全量240组。False 则跑快速小网格验证流程
RUN_FULL = True

# 全量搜索范围（约 4×5×3×4 = 240 组）
# - max_ratio:      [1, 4, 16, 40]
# - grad_ratio:     [2, 32, 40, 64, 90]
# - ssim_shared:    [2, 10, 32]  -> 同时赋给 ssim_ratio 与 ssim_ir_ratio
# - consist_ratio:  [1, 12, 24, 40]
# 其余保持默认：ir_compose=2.0, color_ratio=2.0, window=48, max_mode="l1", consist_mode="l1"
FULL_GRID = {
    "max_ratio":     [1.0, 4.0, 16.0, 40.0],
    "grad_ratio":    [2.0, 32.0, 40.0, 64.0, 90.0],
    "ssim_shared":   [2.0, 10.0, 32.0],
    "consist_ratio": [1.0, 12.0, 24.0, 40.0],
}

# 快速小范围验证（约 2×2×1×2 = 4 组）
QUICK_GRID = {
    "max_ratio":     [16.0],
    "grad_ratio":    [32.0, 64.0],
    "ssim_shared":   [10.0],
    "consist_ratio": [12.0],
}

SELECTED_GRID = FULL_GRID if RUN_FULL else QUICK_GRID

# 每次实验的其他固定权重（如需也可纳入网格）
FIXED_LOSS_KW = {
    "ir_compose": 2.0,
    "color_ratio": 2.0,
    "ssim_window_size": 48,
    "max_mode": "l1",
    "consist_mode": "l1",
}

# 记录结构：每个实验exp_id -> {cfg, history(list[{'epoch':int,'results':{dataset:metrics}}])}
class EvalHook:
    def __init__(self):
        self.history = []
    def __call__(self, epoch: int, results: Dict[str, Dict[str, float]]):
        self.history.append({"epoch": int(epoch), "results": {k: dict(v) for k, v in results.items()}})

def set_method2_globals(cfg: Dict[str, Any], exp_dir: str):
    """根据 cfg 覆盖 method2 的全局变量"""
    # 训练与评测控制
    train.EPOCHS = cfg.get("epochs", EPOCHS)
    train.TEST_FREQ = cfg.get("test_freq", TEST_FREQ)
    train.METRIC_MODE = cfg.get("metric_mode", METRIC_MODE)
    train.PROJECT_DIR = exp_dir
    # 关闭保存以加速扫参
    train.SAVE_IMAGES_TO_DIR = False
    train.SAVE_MODELS = False
    train.SAVE_FREQ = 0
    # 覆盖 FusionLoss 权重
    train.LOSS_MAX_RATIO = float(cfg["max_ratio"])
    train.LOSS_CONSIST_RATIO = float(cfg["consist_ratio"])
    train.LOSS_GRAD_RATIO = float(cfg["grad_ratio"])
    # ssim_shared 优先（同时赋给 ssim_ratio 与 ssim_ir_ratio），否则回退到独立字段
    if "ssim_shared" in cfg:
        shared = float(cfg["ssim_shared"])
        train.LOSS_SSIM_RATIO = shared
        train.LOSS_SSIM_IR_RATIO = shared
    else:
        train.LOSS_SSIM_RATIO = float(cfg["ssim_ratio"])
        train.LOSS_SSIM_IR_RATIO = float(cfg["ssim_ir_ratio"])
    train.LOSS_IR_COMPOSE = float(FIXED_LOSS_KW["ir_compose"])
    train.LOSS_COLOR_RATIO = float(FIXED_LOSS_KW["color_ratio"])
    train.LOSS_SSIM_WINDOW = int(FIXED_LOSS_KW["ssim_window_size"])
    train.LOSS_MAX_MODE = FIXED_LOSS_KW["max_mode"]
    train.LOSS_CONSIST_MODE = FIXED_LOSS_KW["consist_mode"]

def grid_iter(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))

def run_one_experiment(exp_id: int, cfg: Dict[str, Any], out_root: str):
    exp_dir = os.path.join(out_root, f"exp_{exp_id:03d}")
    os.makedirs(exp_dir, exist_ok=True)
    set_method2_globals(cfg, exp_dir)
    # 注入回调
    hook = EvalHook()
    train.EVAL_CALLBACK = hook
    # 运行
    print(f"\n==== EXP {exp_id} start ====\nConfig: {cfg}\nDir: {exp_dir}")
    start = time.time()
    train.main()
    elapsed = time.time() - start
    print(f"==== EXP {exp_id} done in {elapsed/60:.2f} min ====")
    return {"cfg": cfg, "history": hook.history, "elapsed_sec": elapsed}

def history_to_dataframe(exp_id: int, history: List[Dict[str, Any]]):
    rows = []
    for item in history:
        ep = item["epoch"]
        for ds, metrics in item["results"].items():
            row = {"exp_id": exp_id, "epoch": ep, "dataset": ds}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)

def summarize_topk(df_all: pd.DataFrame, params_table: pd.DataFrame, epoch_target: int, topk: int = 10):
    df_e = df_all[df_all["epoch"] == epoch_target].copy()
    if df_e.empty:
        return pd.DataFrame()
    # 只用 Reward 排序；同时保留各数据集的主要指标（VIF/Qabf/SSIM/Reward）
    piv = df_e.pivot_table(index="exp_id", columns="dataset",
                           values=["Reward", "VIF", "Qabf", "SSIM"], aggfunc="mean")
    # 平均 Reward 作为排名依据
    reward_cols = [c for c in piv.columns if isinstance(c, tuple) and c[0] == "Reward"]
    piv[("Reward", "MeanOverSets")] = piv[reward_cols].mean(axis=1)
    piv = piv.sort_values(by=("Reward", "MeanOverSets"), ascending=False)
    # 展平列名
    piv.columns = [f"{a}_{b}" if b != "" else f"{a}" for a, b in piv.columns]
    # 合并参数
    out = params_table.merge(piv.reset_index(), on="exp_id", how="right")
    return out.head(topk)

def main():
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_ROOT, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # 生成配置列表
    configs = list(grid_iter(SELECTED_GRID))
    print(f"Total configs: {len(configs)} (RUN_FULL={RUN_FULL})")

    runs = []
    for i, cfg in enumerate(configs, 1):
        # 附加 epochs 设置（全部 10 epoch，可按需覆盖）
        cfg = dict(cfg)
        cfg["epochs"] = EPOCHS
        cfg["metric_mode"] = METRIC_MODE
        result = run_one_experiment(i, cfg, out_dir)
        runs.append(result)

    # 汇总为 DataFrame
    all_rows = []
    for i, r in enumerate(runs, 1):
        df = history_to_dataframe(i, r["history"])
        df["elapsed_sec"] = r["elapsed_sec"]
        all_rows.append(df)
    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    # 参数表
    params_rows = []
    for i, r in enumerate(runs, 1):
        row = {"exp_id": i}
        row.update(r["cfg"])
        params_rows.append(row)
    params_df = pd.DataFrame(params_rows)

    # 保存所有结果（pth + csv）
    save_obj = {"params": params_df, "metrics": df_all}
    torch.save(save_obj, os.path.join(out_dir, "all_results.pth"))
    params_df.to_csv(os.path.join(out_dir, "params.csv"), index=False)
    df_all.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    print(f"\n[Saved] results -> {out_dir}")

    # 打印 epoch 1/2 的 Top10（只关注早期收敛表现）
    for ep in [1, 2]:
        topk = summarize_topk(df_all, params_df, epoch_target=ep, topk=10)
        print(f"\n===== Top-10 @ epoch {ep} (sorted by mean Reward over datasets) =====")
        if topk.empty:
            print("No records.")
        else:
            # 只显示关键参数列与各数据集主要指标
            # 优先显示 ssim_shared，如不存在则显示 ssim_ratio/ssim_ir_ratio
            if "ssim_shared" in topk.columns:
                cols_front = ["exp_id", "max_ratio", "grad_ratio", "ssim_shared", "consist_ratio"]
            else:
                cols_front = ["exp_id", "max_ratio", "grad_ratio", "ssim_ratio", "ssim_ir_ratio", "consist_ratio"]
            # 其余指标列自动保留
            show_cols = [c for c in cols_front if c in topk.columns] + \
                        [c for c in topk.columns if c not in cols_front]
            print(topk[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()