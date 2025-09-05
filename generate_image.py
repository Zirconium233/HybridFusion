import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dataset import ImageFusionDataset
from model.policy_net import PolicyNet
from model.traditional_fusion import LaplacianPyramidFusion

# -------------------------------
# 可调参数（硬编码）
# -------------------------------
# 模型与推理
PROJECT_DIR = "./checkpoints/stochastic_policy_ycbcr_final"
CKPT_EPOCH = 10  # 加载第 10 个 epoch 的模型
MIXED_PRECISION_DTYPE = torch.float32  # 可选 torch.float32/torch.float16/torch.bfloat16

# 数据与保存 (generate image的速度瓶颈在PNG的CPU压缩和磁盘IO的For循环)
OUT_ROOT = "./save_images/Inference_image"
DATASETS = {
    "MSRS": {
        "dir_A": "./data/MSRS-main/MSRS-main/test/vi",
        "dir_B": "./data/MSRS-main/MSRS-main/test/ir",
    },
    "M3FD": {
        "dir_A": "./data/M3FD_Fusion/Vis",
        "dir_B": "./data/M3FD_Fusion/Ir",
    },
    "RS": {
        "dir_A": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/crop_LR_visible",
        "dir_B": "./data/road-scene-infrared-visible-images-master/road-scene-infrared-visible-images-master/cropinfrared",
    },
    "PET": {
        "dir_A": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/PET-MRI/PET",
        "dir_B": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/PET-MRI/MRI",
    },
    "SPECT": {
        "dir_A": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/SPECT-MRI/SPECT",
        "dir_B": "./data/Med/Havard-Medical-Image-Fusion-Datasets-main/SPECT-MRI/MRI",
    },
}
DATASET_ORDER = ["MSRS", "M3FD", "RS", "PET", "SPECT"]

# Loader
BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True

torch.backends.cudnn.benchmark = False

# -------------------------------
# 工具
# -------------------------------
def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)

def to_01(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
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
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -------------------------------
# 主逻辑
# -------------------------------
@torch.no_grad()
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 加载模型
    ckpt_path = os.path.join(PROJECT_DIR, f"epoch_{CKPT_EPOCH}", "policy_net.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"未找到模型权重: {ckpt_path}")

    # 使用 Y 与 IR 融合（2 通道）
    policy_net = PolicyNet(in_channels=2, out_channels=2).to(device=device, dtype=MIXED_PRECISION_DTYPE)
    state = torch.load(ckpt_path, map_location="cpu")
    policy_net.load_state_dict(state, strict=True)
    policy_net.eval()

    fusion_kernel = LaplacianPyramidFusion(num_levels=4).to(device=device, dtype=MIXED_PRECISION_DTYPE)

    # 2) 遍历数据集
    for name in DATASET_ORDER:
        paths = DATASETS[name]
        print(f"\n[Dataset] {name} -> A:{paths['dir_A']}  B:{paths['dir_B']}")

        ds = ImageFusionDataset(
            dir_A=paths["dir_A"],
            dir_B=paths["dir_B"],
            dir_C=None,
            is_train=False,
            is_getpatch=False,
            augment=False,
        )
        dl = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=max(1, NUM_WORKERS // 2), pin_memory=PIN_MEMORY, drop_last=False
        )
        # 输出目录
        base_out = os.path.join(OUT_ROOT, name)
        out_A = os.path.join(base_out, "A")
        out_B = os.path.join(base_out, "B")
        out_W_mu = os.path.join(base_out, "W_mu")         # mu 权重
        out_W_sample = os.path.join(base_out, "W_sample") # sample 权重
        out_F_mu = os.path.join(base_out, "F_mu")
        out_F_sample = os.path.join(base_out, "F_sample")
        for d in [out_A, out_B, out_W_mu, out_W_sample, out_F_mu, out_F_sample]:
            ensure_dir(d)

        # 3) 推理与保存
        global_idx = 0
        pbar = tqdm(dl, desc=f"[Infer] {name}")
        for batch in pbar:
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                raise RuntimeError("Dataset should return (A,B[,C]).")
            A, B = batch[0], batch[1]
            # 对齐 device / dtype
            A = to_ch_last(A.to(device=device, dtype=MIXED_PRECISION_DTYPE))
            B = to_ch_last(B.to(device=device, dtype=MIXED_PRECISION_DTYPE))

            # 将 A 转 YCbCr，仅融合 Y；B 取 1 通道
            if A.shape[1] == 3:
                Y, Cb, Cr = rgb_to_ycbcr(A)
            else:
                # 兼容极端情况（如灰度 A）
                Y = A.mean(dim=1, keepdim=True)
                Cb = Cr = None
            if B.shape[1] != 1:
                B = B.mean(dim=1, keepdim=True)

            # 策略网络输出（Y, B）
            mu, logvar = policy_net(Y, B)
            std = torch.exp(0.5 * logvar)
            sampled_w = torch.clamp(mu + torch.randn_like(std) * std, 0.0, 1.0)

            # 融合（分别用 mu 和 sample 的权重）在 Y 上执行，并复原 RGB
            F_Y_mu = fusion_kernel(Y, B, mu)
            F_Y_sample = fusion_kernel(Y, B, sampled_w)
            if Cb is not None:
                F_mu = ycbcr_to_rgb(F_Y_mu, Cb, Cr)
                F_sample = ycbcr_to_rgb(F_Y_sample, Cb, Cr)
            else:
                F_mu = F_Y_mu.repeat(1, 3, 1, 1)
                F_sample = F_Y_sample.repeat(1, 3, 1, 1)
            # 保存本批次
            bs = A.shape[0]
            # A/B 原图保存：输入是 [-1,1]，保存到 [0,1]
            # 注意：A 保存原图（RGB/灰度），B 保存为 1 通道
            A_01 = to_01(A.float())
            B_01 = to_01(B.float())
            # 权重直接在 [0,1]（mu、sample）
            W_mu_01 = mu.float().clamp(0, 1)
            W_sample_01 = sampled_w.float().clamp(0, 1)
            # 融合图 [-1,1] -> [0,1] 再存
            F_mu_01 = to_01(F_mu.float())
            F_sample_01 = to_01(F_sample.float())

            for i in range(bs):
                stem = f"{global_idx:06d}.png"
                # A/B
                save_image(A_01[i], os.path.join(out_A, stem))
                save_image(B_01[i], os.path.join(out_B, stem))
                # 权重（灰度单通道为 1xHxW，save_image 支持）
                save_image(W_mu_01[i], os.path.join(out_W_mu, stem))
                save_image(W_sample_01[i], os.path.join(out_W_sample, stem))
                # 融合
                save_image(F_mu_01[i], os.path.join(out_F_mu, stem))
                save_image(F_sample_01[i], os.path.join(out_F_sample, stem))
                global_idx += 1

        print(f"[Done] {name} 保存到: {base_out}")

    print("\nAll datasets finished.")


if __name__ == "__main__":
    run_inference()