import os
import glob
import torch
from typing import Dict, List
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dataset import ImageFusionDataset
from model.policy_net import PolicyNet
from model.traditional_fusion import LaplacianPyramidFusion

# -------------------------------
# 硬编码参数（可按需修改）
# -------------------------------
# 模型来源（多种训练条件）
# key 为条件名；value 为每个任务的检查点目录（脚本会在该目录下尝试 final/epoch_X 形式）
MODEL_SOURCES: Dict[str, Dict[str, str]] = {
    "Med_ycbcr_500": {
        "PET": "./checkpoints/Med_ycbcr_500/PET",
        "SPECT": "./checkpoints/Med_ycbcr_500/SPECT",
        "CT": "./checkpoints/Med_ycbcr_500/CT",
    },
}
# 模型文件优先顺序（按下列顺序寻找）
CKPT_PREFER = ["final/policy_net.pth", "epoch_500/policy_net.pth", "epoch_200/policy_net.pth",
               "epoch_100/policy_net.pth", "epoch_50/policy_net.pth", "epoch_10/policy_net.pth", "epoch_2/policy_net.pth"]

# 推理 dtype（与 train_med.py 一致建议 bf16；如遇不兼容可改为 float32）
INFER_DTYPE = torch.bfloat16  # torch.float32 / torch.float16 / torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集（与 train_med.py 对齐）
MED_DATASETS = {
    "PET": {
        "dir_A": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/PET-MRI/PET",
        "dir_B": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/PET-MRI/MRI",
    },
    "CT": {
        "dir_A": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/CT-MRI/CT",
        "dir_B": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/CT-MRI/MRI",
    },
    "SPECT": {
        "dir_A": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/SPECT-MRI/SPECT",
        "dir_B": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/SPECT-MRI/MRI",
    },
}

# 评测目标定义：每个任务训练后在这些目标上保存图像
EVAL_TARGETS: Dict[str, List[str]] = {
    "PET":   ["PET", "SPECT"],  # 自测 + 跨任务
    "SPECT": ["SPECT", "PET"],  # 自测 + 跨任务
    "CT":    ["CT"],            # 仅自测
}

# Loader
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True

# 保存目录
OUT_ROOT = "./save_images/med_ycbcr"

torch.backends.cudnn.benchmark = False


# -------------------------------
# 工具函数
# -------------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def make_3ch(x: torch.Tensor) -> torch.Tensor:
    return x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)

def make_1ch(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    return x.mean(dim=1, keepdim=True)

def find_ckpt(task_dir: str) -> str:
    # 优先 final，其次按列表寻找 epoch_X；若都不存在，尝试取最大 epoch_* 目录
    for rel in CKPT_PREFER:
        p = os.path.join(task_dir, rel)
        if os.path.isfile(p):
            return p
    # 兜底：选择 task_dir 下最高的 epoch_* 路径
    cands = sorted(glob.glob(os.path.join(task_dir, "epoch_*", "policy_net.pth")))
    if len(cands) > 0:
        return cands[-1]
    return ""


def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)

def to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5
def to_m11(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 1) * 2.0 - 1.0
def rgb_to_ycbcr(x_rgb_m11: torch.Tensor):
    x = to_01(x_rgb_m11)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return to_m11(y), to_m11(cb), to_m11(cr)
def ycbcr_to_rgb(y_m11: torch.Tensor, cb_m11: torch.Tensor, cr_m11: torch.Tensor) -> torch.Tensor:
    y  = to_01(y_m11)
    cb = to_01(cb_m11) - 0.5
    cr = to_01(cr_m11) - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = torch.cat([r, g, b], dim=1).clamp(0, 1)
    return to_m11(rgb)

# -------------------------------
# 单个数据集推理与保存
# -------------------------------
@torch.no_grad()
def infer_and_save_for_target(
    policy_net: PolicyNet,
    fusion_kernel: LaplacianPyramidFusion,
    target_name: str,
    out_dir: str,
):
    paths = MED_DATASETS[target_name]
    print(f"  [Target] {target_name} -> A:{paths['dir_A']}  B:{paths['dir_B']}")

    ds = ImageFusionDataset(
        dir_A=paths["dir_A"], dir_B=paths["dir_B"], dir_C=None,
        is_train=False, is_getpatch=False, augment=False
    )
    print(f"    Dataset size: {len(ds)}")
    dl = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=max(1, NUM_WORKERS // 2), pin_memory=PIN_MEMORY, drop_last=False
    )

    # 子文件夹
    dA = os.path.join(out_dir, "A")
    dB = os.path.join(out_dir, "B")
    dWmu = os.path.join(out_dir, "W_mu")
    dWsa = os.path.join(out_dir, "W_sample")
    dFmu = os.path.join(out_dir, "F_mu")
    dFsa = os.path.join(out_dir, "F_sample")
    for d in [dA, dB, dWmu, dWsa, dFmu, dFsa]:
        ensure_dir(d)

    idx_global = 0
    pbar = tqdm(dl, desc=f"[Infer] {target_name}", leave=False)
    for batch in pbar:
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise RuntimeError("Dataset should return (A,B[,C]).")
        A, B = batch[0], batch[1]

        # dtype/device
        A = to_ch_last(A.to(device=DEVICE, dtype=INFER_DTYPE))
        B = to_ch_last(B.to(device=DEVICE, dtype=INFER_DTYPE))

        # 通道处理：PET/SPECT 走 Y 融合；CT 直接灰度融合（不做 YCbCr）
        B1 = make_1ch(B)
        if target_name == "CT" or A.shape[1] != 3:
            Y = make_1ch(A)
            Cb = Cr = None
        else:
            A3 = make_3ch(A)
            Y, Cb, Cr = rgb_to_ycbcr(A3)

        # 策略输出
        mu, logvar = policy_net(Y, B1)
        std = torch.exp(0.5 * logvar)
        sampled_w = torch.clamp(mu + torch.randn_like(std) * std, 0.0, 1.0)

        # 融合
        F_Y_mu = fusion_kernel(Y, B1, mu)
        F_Y_sa = fusion_kernel(Y, B1, sampled_w)
        if Cb is not None:
            F_mu = ycbcr_to_rgb(F_Y_mu, Cb, Cr)
            F_sa = ycbcr_to_rgb(F_Y_sa, Cb, Cr)
        else:
            F_mu = F_Y_mu.repeat(1, 3, 1, 1)
            F_sa = F_Y_sa.repeat(1, 3, 1, 1)

        # 转 CPU 一次性写盘（减少 GPU->CPU 同步次数）
        # A 保存原图（CT 可为单通道）
        A_01_cpu = to_01((A if A.shape[1] == 1 else A).float()).cpu()
        B_01_cpu = to_01(B1.float()).cpu()
        W_mu_cpu = mu.float().clamp(0, 1).cpu()
        W_sa_cpu = sampled_w.float().clamp(0, 1).cpu()
        F_mu_cpu = to_01(F_mu.float()).cpu()
        F_sa_cpu = to_01(F_sa.float()).cpu()

        bs = A_01_cpu.shape[0]
        for i in range(bs):
            stem = f"{idx_global:06d}.png"
            save_image(A_01_cpu[i], os.path.join(dA, stem))
            save_image(B_01_cpu[i], os.path.join(dB, stem))
            save_image(W_mu_cpu[i], os.path.join(dWmu, stem))
            save_image(W_sa_cpu[i], os.path.join(dWsa, stem))
            save_image(F_mu_cpu[i], os.path.join(dFmu, stem))
            save_image(F_sa_cpu[i], os.path.join(dFsa, stem))
            idx_global += 1


# -------------------------------
# 主流程：对多种条件与任务保存跨任务/自测结果
# -------------------------------
@torch.no_grad()
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    for cond_name, task_dirs in MODEL_SOURCES.items():
        print(f"\n[Condition] {cond_name}")

        for task, task_dir in task_dirs.items():
            # 查找 ckpt
            ckpt_path = find_ckpt(task_dir)
            if not ckpt_path:
                print(f"  [Skip] {task}: 未找到模型（在 {task_dir} 下未找到 policy_net.pth）")
                continue

            print(f"  [Load] {task} <- {ckpt_path}")
            # 模型和融合核
            # 统一两通道（Y 与 MRI/IR）；CT 也为灰度+MRI
            policy_net = PolicyNet(in_channels=2, out_channels=2).to(device=DEVICE, dtype=INFER_DTYPE)
            state = torch.load(ckpt_path, map_location="cpu")
            policy_net.load_state_dict(state, strict=True)
            policy_net.eval()

            fusion_kernel = LaplacianPyramidFusion(num_levels=4).to(device=DEVICE, dtype=INFER_DTYPE)

            # 评测目标（PET/SPECT 互测，CT 仅自测）
            targets = EVAL_TARGETS.get(task, [task])
            for tgt in targets:
                out_dir = os.path.join(OUT_ROOT, cond_name, f"{task}_eval_{tgt}")
                ensure_dir(out_dir)
                infer_and_save_for_target(policy_net, fusion_kernel, tgt, out_dir)

    print("\nAll conditions finished.")


if __name__ == '__main__':
    main()