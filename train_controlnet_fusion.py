#!/usr/bin/env python
# coding=utf-8
"""
ControlNet 4通道(3+1)条件输入训练脚本（用于图像融合：RGB可见光 + IR）。
- 条件输入：concat([A_RGB(3c), B_gray(1c)]) -> [B, 4, H, W]，范围[0,1]
- 目标输出：伪标签 C_RGB，范围[-1,1]，用于扩散训练的噪声回归
- 文本提示：统一为空串（弱化文本提示，让ControlNet主导）
- 分辨率：默认 480x640（与 MSRS 原图一致），需为8的倍数
- 依赖：diffusers>=0.26, accelerate, transformers, torchvision
"""

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Dict

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# 本地数据集
from Image.dataset import ImageFusionDataset

# 要求最低版本（与参考脚本保持一致）
check_min_version("0.26.0")

logger = get_logger(__name__)

# 数据集路径索引（可扩展多个数据集）
DATASETS: Dict[str, Dict[str, Dict[str, str]]] = {
    "MSRS": {
        "train": {
            "dir_A": "./data/MSRS-main/MSRS-main/train/vi",
            "dir_B": "./data/MSRS-main/MSRS-main/train/ir",
            "dir_C": "./data/MSRS-main/MSRS-main/train/label",
        },
        "test": {
            "dir_A": "./data/MSRS-main/MSRS-main/test/vi",
            "dir_B": "./data/MSRS-main/MSRS-main/test/ir",
        },
    }
}
TEST_SET_NAMES = ["MSRS"]


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ControlNet with 4-channel conditioning for image fusion.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Stable Diffusion base model path or HF model id (e.g. runwayml/stable-diffusion-v1-5).",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path or HF model id to an existing 4ch ControlNet. If not set, initialize from UNet with 4 channels.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    # 数据集
    parser.add_argument("--dataset_name", type=str, default="MSRS", help=f"One of: {list(DATASETS.keys())}")
    parser.add_argument("--train_dir_A", type=str, default=None, help="Override train A dir (RGB).")
    parser.add_argument("--train_dir_B", type=str, default=None, help="Override train B dir (IR gray).")
    parser.add_argument("--train_dir_C", type=str, default=None, help="Override train C dir (pseudo-label RGB).")

    # 分辨率（保持原始 480x640）
    parser.add_argument("--height", type=int, default=480, help="Image height (must be divisible by 8).")
    parser.add_argument("--width", type=int, default=640, help="Image width (must be divisible by 8).")

    # 训练超参
    parser.add_argument("--output_dir", type=str, default="controlnet-4ch-fusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # 工程与加速
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help='path or "latest"')
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")

    args = parser.parse_args()

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("height and width must be divisible by 8.")
    return args


class FusionControlNetDataset(torch.utils.data.Dataset):
    """
    读取 A(可见RGB), B(IR灰度), C(伪标签RGB)，输出：
      - pixel_values: C, [-1,1], float32, [3,H,W]
      - conditioning_pixel_values: concat([A(0..1), B(0..1)]), [4,H,W]
      - input_ids: 空文本的token ids
    """
    def __init__(self, dir_A, dir_B, dir_C, height=480, width=640, tokenizer=None, is_train=True):
        assert dir_C is not None and os.path.isdir(dir_C), "训练需要提供伪标签C目录"
        self.inner = ImageFusionDataset(
            dir_A=dir_A, dir_B=dir_B, dir_C=dir_C,
            is_train=is_train, is_getpatch=False, patch_size=128,
            augment=False, transform=None, scale=1
        )
        self.height = height
        self.width = width
        self.tokenizer = tokenizer

        self.resize_A = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR)
        self.resize_B = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR)
        self.resize_C = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR)
        self.to_tensor = transforms.ToTensor()

        # 预先构造空文本输入
        if tokenizer is not None:
            self.empty_input_ids = tokenizer(
                [""], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids[0]
        else:
            self.empty_input_ids = None

    def __len__(self):
        return len(self.inner)

    @staticmethod
    def to_neg1_pos1(t: torch.Tensor) -> torch.Tensor:
        return t.mul_(2.0).add_(-1.0)

    def __getitem__(self, idx):
        # 返回 PIL: A(RGB), B(L), C(RGB)
        A, B, C = self.inner[idx]

        # 条件图像 [0,1]
        A_t = self.to_tensor(self.resize_A(A.convert("RGB")))     # [3,H,W], 0..1
        B_t = self.to_tensor(self.resize_B(B))                    # [1,H,W], 0..1
        cond = torch.cat([A_t, B_t], dim=0)                       # [4,H,W]

        # 目标像素 [-1,1]
        C_t = self.to_tensor(self.resize_C(C.convert("RGB")))     # [3,H,W], 0..1
        pixel_values = self.to_neg1_pos1(C_t)                     # [-1,1]

        sample = {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": cond,
        }

        if self.empty_input_ids is not None:
            sample["input_ids"] = self.empty_input_ids.clone()
        return sample


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples], dim=0).float().contiguous()
    conditioning_pixel_values = torch.stack([e["conditioning_pixel_values"] for e in examples], dim=0).float().contiguous()
    batch = {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
    }
    if "input_ids" in examples[0]:
        input_ids = torch.stack([e["input_ids"] for e in examples], dim=0)
        batch["input_ids"] = input_ids
    return batch


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def main():
    args = parse_args()

    # 日志与加速
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        import transformers, diffusers as _diff
        transformers.utils.logging.set_verbosity_warning()
        _diff.utils.logging.set_verbosity_info()
    else:
        import transformers, diffusers as _diff
        transformers.utils.logging.set_verbosity_error()
        _diff.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # 输出目录
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # 数据路径解析
    if args.train_dir_A and args.train_dir_B and args.train_dir_C:
        dir_A, dir_B, dir_C = args.train_dir_A, args.train_dir_B, args.train_dir_C
    else:
        if args.dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset_name={args.dataset_name}")
        ds_cfg = DATASETS[args.dataset_name]["train"]
        dir_A, dir_B, dir_C = ds_cfg["dir_A"], ds_cfg["dir_B"], ds_cfg["dir_C"]

    # 加载分词器与文本编码器
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # 模型
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # ControlNet：4 通道条件
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing ControlNet weights...")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        # 尝试检查配置
        cond_ch = getattr(controlnet.config, "conditioning_channels", None)
        if cond_ch is not None and cond_ch != 4:
            logger.warning(f"Loaded ControlNet conditioning_channels={cond_ch}, expected 4. "
                           f"Make sure this model was trained for 4-channel conditioning.")
    else:
        logger.info("Initializing ControlNet from UNet with 4-channel conditioning...")
        try:
            controlnet = ControlNetModel.from_unet(unet, conditioning_channels=4)
        except TypeError:
            # 兼容旧版 diffusers（不支持直接传 conditioning_channels 的情况）
            # 这种情况下建议升级 diffusers 版本。这里仍然构建一个默认3通道的模型并提示错误。
            raise RuntimeError(
                "Your diffusers version does not support ControlNetModel.from_unet(..., conditioning_channels=4). "
                "Please upgrade diffusers to a newer version."
            )

    # 可选：xFormers
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            controlnet.enable_xformers_memory_efficient_attention()
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Please install it correctly.")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # 冻结除 ControlNet 外的模块
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # dtype 检查
    if unwrap_model(accelerator, controlnet).dtype != torch.float32:
        raise ValueError(
            f"ControlNet dtype is {unwrap_model(accelerator, controlnet).dtype}. "
            f"Please keep trainable weights in float32 at start."
        )

    # TF32
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 优化器
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Install bitsandbytes to use 8-bit Adam: pip install bitsandbytes")
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 数据集与 DataLoader
    train_dataset = FusionControlNetDataset(
        dir_A=dir_A, dir_B=dir_B, dir_C=dir_C,
        height=args.height, width=args.width,
        tokenizer=tokenizer, is_train=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 学习率调度
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_after_shard = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_after_shard / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # hooks：让 accelerator.save_state() 以更清晰的方式仅保存 controlnet
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # accelerator.prepare
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # 混合精度
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 将非训练模块移动到设备并cast
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # 训练步数/轮数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if accelerator.is_main_process and num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                "train_dataloader长度在 prepare 前后不一致，学习率调度可能不完全匹配（通常无伤大雅）。"
            )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 追踪器
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers("train_controlnet_fusion", config=tracker_config)

    # 可能从断点恢复
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is not None:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    # 进度条
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Dataset: {args.dataset_name}")
    logger.info(f"  Train samples = {len(train_dataset)}")
    logger.info(f"  Image size = {args.height}x{args.width}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total steps = {args.max_train_steps}")

    # 训练循环
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Encode target images to latents
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(dtype=weight_dtype)

                # Empty text embeddings
                if "input_ids" in batch:
                    input_ids = batch["input_ids"].to(accelerator.device)
                    encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]
                else:
                    # 理论上不会走到这里
                    raise RuntimeError("Missing input_ids in batch.")

                # 4通道条件图
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype, device=accelerator.device)

                # ControlNet forward
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # UNet 预测噪声
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[s.to(dtype=weight_dtype) for s in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # 目标
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # 步进与日志
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    # 限制checkpoint数量
                    if args.checkpoints_total_limit is not None:
                        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            for rm in checkpoints[:num_to_remove]:
                                shutil.rmtree(os.path.join(args.output_dir, rm))
                    # 保存
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # 保存最终 ControlNet
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        cn = unwrap_model(accelerator, controlnet)
        cn.save_pretrained(args.output_dir)
        logger.info(f"ControlNet saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()