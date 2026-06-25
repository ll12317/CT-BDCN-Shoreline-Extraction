# -*- coding: utf-8 -*-
"""
Improved CNN + Transformer / CT-BDCN training script
Unified comparison experiment version:
- Uses data_rgb.py
- Uses three-channel RGB input: [B, 3, 256, 256]
- Keeps the improved architecture:
    1) conv0 high-resolution branch
    2) CNN + Transformer backbone
    3) U-Net-BDCN multi-scale boundary branch
    4) side-output deep supervision
    5) gated boundary enhancement
    6) boundary residual refinement
- Loss terms:
    1) boundary-weighted BCE segmentation loss
    2) main edge BCE loss
    3) multi-scale side-output edge loss
    4) boundary soft IoU loss
- Saves last / best_loss / best_iou checkpoints
- Exports prediction maps, edge maps, triplet comparison images, and training curves
- Computes PA / Recall / Precision / F1 / IoU
- Computes PA_edge / Recall_edge / Precision_edge / F1_edge / IoU_edge
- Computes Boundary_IoU / HD95_m
- Profiles model parameters, FLOPs, and inference speed
"""

import os
import sys
import time
import json
import random
import platform
import subprocess
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
import tqdm


# ================== Required dependency: scipy for Boundary IoU and HD95 ==================
try:
    from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
    SCIPY_AVAILABLE = True
except Exception as e:
    SCIPY_AVAILABLE = False
    raise ImportError(
        "\n[Error] scipy is required for Boundary IoU and HD95 calculation.\n"
        "Please install scipy before running this script:\n"
        "    pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
        "or:\n"
        "    conda install scipy\n"
    ) from e


# ================== Paths and imports ==================
current_dir = Path(__file__).resolve().parent
print("current_dir:", current_dir)

sys.path.append(str(current_dir))

try:
    from data_rgb import MyDataset
    print("Successfully imported MyDataset from data_rgb.py")
except Exception as e:
    raise ImportError(
        "Failed to import MyDataset from data_rgb.py. "
        "Please make sure data_rgb.py is in the same folder as this script "
        "and contains a class or function named MyDataset."
    ) from e


# ================== Global configuration ==================
IN_CHANNELS = 3
NUM_CLASSES = 1

PATCH_SIZE = (256, 256)
BATCH_SIZE = 4
NUM_WORKERS = 0
NUM_EPOCHS = 150
SEED = 42
INITIAL_LR = 1e-4

# Loss weights
ALPHA_BOUNDARY_WEIGHT = 2.0      # BCE weighting coefficient for boundary pixels
LAMBDA_EDGE = 0.2                # Weight for the main edge-branch loss
LAMBDA_SIDE = 0.3                # Weight for side-output deep supervision
GAMMA_EDGE_IOU = 1.0             # Weight for boundary soft-IoU loss

# Sentinel-2 spatial resolution: 10 m
PIXEL_SIZE_M = 10.0

# Boundary narrow-band width used for edge loss and IoU_edge
EDGE_BAND_K = 5

# Boundary dilation radius for Boundary IoU; 2 pixels are approximately 20 m
BOUNDARY_DILATION_ITER = 2

# Inference speed benchmark
PROFILE_INPUT_BATCH = 1
BENCHMARK_WARMUP = 30
BENCHMARK_REPEATS = 200

MODEL_NAME = "ct_bdcn"
DATASET_SOURCE = "data_rgb.py / MyDataset"


# ================== Random seed ==================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


set_seed(SEED)


# ================== Device configuration ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("using device:", device)
print("SCIPY_AVAILABLE:", SCIPY_AVAILABLE)


# ================== Output paths ==================
save_path = current_dir / "result"
save_path.mkdir(parents=True, exist_ok=True)

run_name = f"{MODEL_NAME}_RGB_data_rgb_in{IN_CHANNELS}_seed{SEED}"

weight_path = save_path / f"{run_name}_last.pth"
best_loss_weight_path = save_path / f"{run_name}_best_loss.pth"
best_iou_weight_path = save_path / f"{run_name}_best_iou.pth"

best_pred_dir = save_path / f"{run_name}_best_iou_preds"
compare_dir = save_path / f"{run_name}_best_iou_compare"
edge_pred_dir = save_path / f"{run_name}_best_iou_edge_preds"
metrics_dir = save_path / f"{run_name}_metrics"

for d in (best_pred_dir, compare_dir, edge_pred_dir, metrics_dir):
    d.mkdir(parents=True, exist_ok=True)


# =========================================================
# Network architecture: Improved CNN + Transformer / CT-BDCN
# =========================================================
class ConvBNReLU(nn.Module):
    """Conv + BN + ReLU."""
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                bias=False
            ),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetBDCN_EdgeBranch(nn.Module):
    """
    U-Net-BDCN style multi-scale edge branch.

    Input:
        f1: [B, 64, H/2, W/2]

    Output:
        edge_feat: [B, out_ch, H/2, W/2]
        side1:     [B, 1, H/2, W/2]
        side2_up:  [B, 1, H/2, W/2]
        side3_up:  [B, 1, H/2, W/2]
    """
    def __init__(self, in_ch=64, c1=32, c2=64, out_ch=16):
        super().__init__()

        self.enc1_1 = ConvBNReLU(in_ch, c1)
        self.enc1_2 = ConvBNReLU(c1, c1)
        self.side1 = nn.Conv2d(c1, 1, kernel_size=1)

        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = ConvBNReLU(c1, c2)
        self.enc2_2 = ConvBNReLU(c2, c2)
        self.side2 = nn.Conv2d(c2, 1, kernel_size=1)

        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = ConvBNReLU(c2, c2)
        self.enc3_2 = ConvBNReLU(c2, c2)
        self.side3 = nn.Conv2d(c2, 1, kernel_size=1)

        self.up2 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)
        self.dec2_1 = ConvBNReLU(c2 + c2, c2)
        self.dec2_2 = ConvBNReLU(c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1_1 = ConvBNReLU(c1 + c1, c1)
        self.dec1_2 = ConvBNReLU(c1, c1)

        self.edge_fuse = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.out_feat = nn.Conv2d(c1 + 16, out_ch, kernel_size=1)

    def forward(self, f1):
        B, _, H2, W2 = f1.shape

        e1 = self.enc1_1(f1)
        e1 = self.enc1_2(e1)
        side1 = self.side1(e1)

        x = self.pool1(e1)
        e2 = self.enc2_1(x)
        e2 = self.enc2_2(e2)
        side2 = self.side2(e2)

        x = self.pool2(e2)
        e3 = self.enc3_1(x)
        e3 = self.enc3_2(e3)
        side3 = self.side3(e3)

        d2 = self.up2(e3)

        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(
                d2,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2_1(d2)
        d2 = self.dec2_2(d2)

        d1 = self.up1(d2)

        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(
                d1,
                size=e1.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1_1(d1)
        d1 = self.dec1_2(d1)

        side2_up = F.interpolate(
            side2,
            size=(H2, W2),
            mode="bilinear",
            align_corners=False
        )

        side3_up = F.interpolate(
            side3,
            size=(H2, W2),
            mode="bilinear",
            align_corners=False
        )

        side_cat = torch.cat([side1, side2_up, side3_up], dim=1)
        edge_multi = self.edge_fuse(side_cat)

        feat_cat = torch.cat([d1, edge_multi], dim=1)
        edge_feat = self.out_feat(feat_cat)

        return edge_feat, side1, side2_up, side3_up


class EdgeEnhancedSegNet_UNetBDCN(nn.Module):
    """
    Improved CNN + Transformer / CT-BDCN.

    Input:
        [B, 3, H, W]

    Output:
        seg_logits:  [B, 1, H, W]
        edge_logits: [B, 1, H, W]
    """
    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        embed_dim=128,
        num_heads=4,
        transformer_layers=2
    ):
        super().__init__()

        self._last_side_maps = None

        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                64,
                embed_dim,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=transformer_layers
        )

        self.edge_branch = UNetBDCN_EdgeBranch(
            in_ch=64,
            c1=32,
            c2=64,
            out_ch=16
        )

        self.edge_gate = nn.Conv2d(16, 1, kernel_size=1, bias=True)

        self.edge_out = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        self.up1 = nn.ConvTranspose2d(
            embed_dim,
            64,
            kernel_size=2,
            stride=2
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(
                64 + 64 + 16,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.high_fuse = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        self.edge_refine = nn.Sequential(
            nn.Conv2d(num_classes + 1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        f0 = self.conv0(x)
        f1 = self.conv1(f0)
        f2 = self.conv2(f1)

        B, C, h, w = f2.shape

        t = f2.flatten(2).transpose(1, 2)
        t = self.transformer(t)
        f2_trans = t.transpose(1, 2).reshape(B, C, h, w)

        edge_feat, s1, s2_up, s3_up = self.edge_branch(f1)
        self._last_side_maps = (s1, s2_up, s3_up)

        gate = torch.sigmoid(self.edge_gate(edge_feat))
        edge_feat_gated = edge_feat * (0.5 + gate)

        edge_map = self.edge_out(edge_feat_gated)

        if edge_map.shape[-2:] != input_size:
            edge_map = F.interpolate(
                edge_map,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )

        up1 = self.up1(f2_trans)

        if up1.shape[-2:] != f1.shape[-2:]:
            up1 = F.interpolate(
                up1,
                size=f1.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        fusion = torch.cat([up1, f1, edge_feat_gated], dim=1)
        x = self.fuse_conv(fusion)
        x = self.up2(x)

        if x.shape[-2:] != f0.shape[-2:]:
            x = F.interpolate(
                x,
                size=f0.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = self.high_fuse(torch.cat([x, f0], dim=1))

        if x.shape[-2:] != input_size:
            x = F.interpolate(
                x,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )

        seg_logits = self.final_conv(x)

        if edge_map.shape[-2:] != seg_logits.shape[-2:]:
            edge_map = F.interpolate(
                edge_map,
                size=seg_logits.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        refine_in = torch.cat([seg_logits, edge_map], dim=1)
        edge_res = self.edge_refine(refine_in)
        seg_logits = seg_logits + edge_res

        return seg_logits, edge_map


# ================== Data loading ==================
train_dataset = MyDataset(
    mode="train",
    size=PATCH_SIZE,
    augment=True,
)

data_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print(f"Loaded train dataset: {len(train_dataset)} samples")


# ================== Model ==================
model = EdgeEnhancedSegNet_UNetBDCN(
    in_channels=IN_CHANNELS,
    num_classes=NUM_CLASSES,
    embed_dim=128,
    num_heads=4,
    transformer_layers=2
).to(device)


# ================== Checkpoint loading ==================
if weight_path.exists():
    try:
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        print("Successfully loaded the last checkpoint.")
    except RuntimeError as e:
        print("Existing checkpoint does not match the current model.")
        print("Training from scratch.")
        print("Detail:", e)
else:
    print("No existing checkpoint found. Training from scratch.")


# ================== Loss functions and optimizer ==================
criterion_seg = nn.BCEWithLogitsLoss()
criterion_edge = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)


# =========================================================
# Utility functions
# =========================================================
def unpack_batch(batch):
    """
    Supports Dataset outputs:
        (image, mask)
        (image, mask, name)
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return batch[0], batch[1]

    raise RuntimeError(
        "Batch format error. Expected (images, masks) or (images, masks, names)."
    )


def ensure_mask_bchw(masks: torch.Tensor) -> torch.Tensor:
    """
    Ensure masks are [B, 1, H, W] and binary float.
    """
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)

    masks = masks.float()

    if masks.max() > 1.0:
        masks = masks / 255.0

    masks = (masks > 0.5).float()

    return masks


def ensure_output_tuple(outputs):
    """
    Ensure model output is returned as (seg_logits, edge_logits).
    """
    if isinstance(outputs, (tuple, list)):
        if len(outputs) >= 2:
            return outputs[0], outputs[1]
        if len(outputs) == 1:
            return outputs[0], None

    return outputs, None


def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def check_dataset_channel(dataset, dataset_name="train"):
    """
    Check whether images returned by data_rgb.py are three-channel RGB tensors.
    """
    sample = dataset[0]

    if isinstance(sample, (tuple, list)):
        img = sample[0]
        mask = sample[1]
    else:
        raise RuntimeError(
            f"{dataset_name} dataset should return at least (image, mask)."
        )

    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)

    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)

    print(f"{dataset_name} sample image shape:", img.shape)
    print(f"{dataset_name} sample mask shape:", mask.shape)

    if img.dim() != 3:
        raise ValueError(
            f"{dataset_name} image should be [C, H, W], but got {img.shape}. "
            f"Please check data_rgb.py."
        )

    actual_channels = img.shape[0]

    if actual_channels != IN_CHANNELS:
        raise ValueError(
            f"\nInput channel mismatch in {dataset_name} dataset:\n"
            f"  IN_CHANNELS in this script = {IN_CHANNELS}\n"
            f"  image channels returned by data_rgb.py = {actual_channels}\n\n"
            f"This RGB experiment requires 3-channel images."
        )

    img_float = img.float()

    channel_min = img_float.reshape(actual_channels, -1).min(dim=1).values
    channel_max = img_float.reshape(actual_channels, -1).max(dim=1).values

    print(
        f"{dataset_name} channel min:",
        [round(float(v), 4) for v in channel_min]
    )

    print(
        f"{dataset_name} channel max:",
        [round(float(v), 4) for v in channel_max]
    )


check_dataset_channel(train_dataset, "train")


# =========================================================
# Environment information, model complexity, and inference speed
# =========================================================
def get_environment_info():
    info = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "seed": SEED,
        "model_name": MODEL_NAME,
        "dataset_source": DATASET_SOURCE,
        "input_channels": IN_CHANNELS,
        "num_classes": NUM_CLASSES,
        "patch_size": PATCH_SIZE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "initial_lr": INITIAL_LR,
        "alpha_boundary_weight": ALPHA_BOUNDARY_WEIGHT,
        "lambda_edge": LAMBDA_EDGE,
        "lambda_side": LAMBDA_SIDE,
        "gamma_edge_iou": GAMMA_EDGE_IOU,
        "pixel_size_m": PIXEL_SIZE_M,
        "edge_band_k": EDGE_BAND_K,
        "boundary_dilation_iter": BOUNDARY_DILATION_ITER,
        "scipy_available": SCIPY_AVAILABLE,
        "current_dir": str(current_dir),
    }

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(device_id)

        info.update({
            "gpu_name": torch.cuda.get_device_name(device_id),
            "gpu_total_memory_GB": round(prop.total_memory / 1024 ** 3, 3),
            "gpu_compute_capability": f"{prop.major}.{prop.minor}",
        })

        try:
            nvcc_info = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
            info["nvcc_version"] = nvcc_info
        except Exception:
            info["nvcc_version"] = "Not available"

    return info


def save_environment_info(save_dir):
    env_info = get_environment_info()

    print("\n========== Environment Info ==========")
    for k, v in env_info.items():
        print(f"{k}: {v}")
    print("======================================\n")

    with open(Path(save_dir) / "environment_info.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=4, ensure_ascii=False)

    return env_info


def profile_model_complexity(
    model: nn.Module,
    input_size=(1, 3, 256, 256),
    device=device,
    save_dir=None
):
    """
    Profile model parameters and FLOPs.
    """
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    total_params, trainable_params = count_parameters(model)

    flops = None
    gflops = None

    try:
        activities = [torch.profiler.ProfilerActivity.CPU]

        if torch.cuda.is_available() and device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.no_grad():
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_flops=True
            ) as prof:
                outputs = model(dummy_input)
                seg_logits, edge_logits = ensure_output_tuple(outputs)

        flops = sum(
            evt.flops
            for evt in prof.key_averages()
            if hasattr(evt, "flops") and evt.flops is not None
        )

        gflops = flops / 1e9

    except Exception as e:
        print("[Warning] FLOPs profiling failed:", e)

    result = {
        "input_size": list(input_size),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "total_params_M": round(total_params / 1e6, 4),
        "trainable_params_M": round(trainable_params / 1e6, 4),
        "FLOPs": int(flops) if flops is not None else None,
        "GFLOPs": round(gflops, 4) if gflops is not None else None,
    }

    print("\n========== Model Complexity ==========")
    print(f"Input size: {input_size}")
    print(f"Total params: {result['total_params_M']} M")
    print(f"Trainable params: {result['trainable_params_M']} M")
    print(f"GFLOPs: {result['GFLOPs']}")
    print("======================================\n")

    if save_dir is not None:
        with open(Path(save_dir) / "model_complexity.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    return result


def benchmark_inference_speed(
    model: nn.Module,
    input_size=(1, 3, 256, 256),
    device=device,
    warmup=30,
    repeats=200,
    save_dir=None
):
    """
    Benchmark inference speed, FPS, and peak GPU memory.
    """
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(warmup):
            outputs = model(dummy_input)
            seg_logits, edge_logits = ensure_output_tuple(outputs)

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        for _ in range(repeats):
            outputs = model(dummy_input)
            seg_logits, edge_logits = ensure_output_tuple(outputs)

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start_time

    avg_time_per_batch = elapsed / repeats
    avg_time_per_image = avg_time_per_batch / input_size[0]
    fps = 1.0 / avg_time_per_image

    peak_memory = None

    if torch.cuda.is_available() and device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2

    result = {
        "input_size": list(input_size),
        "warmup": warmup,
        "repeats": repeats,
        "avg_inference_time_ms_per_batch": round(avg_time_per_batch * 1000, 4),
        "avg_inference_time_ms_per_image": round(avg_time_per_image * 1000, 4),
        "FPS_images_per_second": round(fps, 4),
        "peak_gpu_memory_MB": round(peak_memory, 4) if peak_memory is not None else None,
    }

    print("\n========== Inference Speed ==========")
    print(f"Avg inference time: {result['avg_inference_time_ms_per_image']} ms/image")
    print(f"FPS: {result['FPS_images_per_second']}")
    print(f"Peak GPU memory: {result['peak_gpu_memory_MB']} MB")
    print("=====================================\n")

    if save_dir is not None:
        with open(Path(save_dir) / "inference_speed.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    return result


# =========================================================
# IoU_edge, Boundary IoU, and HD95
# =========================================================
def get_edge_band(gt_mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Construct a narrow boundary band from the ground-truth mask for edge loss and IoU_edge.
    """
    gt_mask = ensure_mask_bchw(gt_mask)

    pad = k // 2

    dilate = F.max_pool2d(
        gt_mask,
        kernel_size=k,
        stride=1,
        padding=pad
    )

    erode = 1.0 - F.max_pool2d(
        1.0 - gt_mask,
        kernel_size=k,
        stride=1,
        padding=pad
    )

    band = (dilate - erode) > 0.5

    return band.float()


def edge_iou_loss(
    seg_logits: torch.Tensor,
    masks: torch.Tensor,
    k: int = 5
) -> torch.Tensor:
    """
    Compute soft IoU loss within the boundary band.

    loss = 1 - IoU_edge
    """
    masks = ensure_mask_bchw(masks)
    probs = torch.sigmoid(seg_logits)

    with torch.no_grad():
        band = get_edge_band(masks, k=k)

    p_edge = probs * band
    g_edge = masks * band

    inter = (p_edge * g_edge).sum()
    union = (p_edge + g_edge - p_edge * g_edge).sum() + 1e-6

    iou_edge = inter / union

    return 1.0 - iou_edge


def mask_to_boundary_np(mask: np.ndarray) -> np.ndarray:
    """
    Convert a binary mask to a boundary map.
    """
    mask = (mask > 0).astype(bool)

    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(
        mask,
        structure=np.ones((3, 3)),
        border_value=0
    )

    boundary = mask ^ eroded

    return boundary


def boundary_iou_np(
    pred: np.ndarray,
    gt: np.ndarray,
    dilation_iter: int = 2
):
    """
    Compute Boundary IoU.
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    pred_b = mask_to_boundary_np(pred)
    gt_b = mask_to_boundary_np(gt)

    if gt_b.sum() == 0:
        return None

    if pred_b.sum() == 0:
        return 0.0

    pred_band = binary_dilation(
        pred_b,
        structure=np.ones((3, 3)),
        iterations=dilation_iter
    )

    gt_band = binary_dilation(
        gt_b,
        structure=np.ones((3, 3)),
        iterations=dilation_iter
    )

    inter = np.logical_and(pred_band, gt_band).sum()
    union = np.logical_or(pred_band, gt_band).sum()

    return float(inter / (union + 1e-6))


def hd95_np(
    pred: np.ndarray,
    gt: np.ndarray,
    pixel_size: float = 10.0
):
    """
    Compute HD95, the 95th percentile Hausdorff distance.

    Unit: metres.
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    pred_b = mask_to_boundary_np(pred)
    gt_b = mask_to_boundary_np(gt)

    if gt_b.sum() == 0:
        return None

    H, W = gt.shape

    if pred_b.sum() == 0:
        max_dist_m = np.sqrt(H ** 2 + W ** 2) * pixel_size
        return float(max_dist_m)

    dt_gt = distance_transform_edt(~gt_b)
    dt_pred = distance_transform_edt(~pred_b)

    d_pred_to_gt = dt_gt[pred_b]
    d_gt_to_pred = dt_pred[gt_b]

    distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])

    if distances.size == 0:
        max_dist_m = np.sqrt(H ** 2 + W ** 2) * pixel_size
        return float(max_dist_m)

    hd95_pixel = np.percentile(distances, 95)

    return float(hd95_pixel * pixel_size)


# =========================================================
# Training function: save best_loss and best_iou checkpoints
# =========================================================
def train(
    model,
    data_loader,
    criterion_seg,
    criterion_edge,
    optimizer,
    num_epochs=100
):
    history = []

    best_loss = float("inf")
    best_loss_epoch = -1

    best_iou = -1.0
    best_iou_epoch = -1

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()

        running_loss = 0.0
        running_seg_loss = 0.0
        running_edge_main_loss = 0.0
        running_side_loss = 0.0
        running_edge_iou_loss = 0.0

        total_pixels = 0
        total_correct = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for i, batch in enumerate(
            tqdm.tqdm(data_loader, desc=f"Epoch {epoch + 1}")
        ):
            images, masks = unpack_batch(batch)

            images = images.to(device).float()
            masks = ensure_mask_bchw(masks.to(device))

            outputs = model(images)
            seg_logits, edge_logits = ensure_output_tuple(outputs)

            if seg_logits.shape[-2:] != masks.shape[-2:]:
                seg_logits = F.interpolate(
                    seg_logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            edge_gt = get_edge_band(masks, k=EDGE_BAND_K)

            if edge_logits is not None:
                if edge_logits.shape[-2:] != edge_gt.shape[-2:]:
                    edge_logits = F.interpolate(
                        edge_logits,
                        size=edge_gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

            weight_map = 1.0 + ALPHA_BOUNDARY_WEIGHT * edge_gt

            loss_seg = F.binary_cross_entropy_with_logits(
                seg_logits,
                masks,
                weight=weight_map
            )

            if edge_logits is None:
                loss_edge_main = torch.tensor(0.0, device=device)
            else:
                loss_edge_main = criterion_edge(edge_logits, edge_gt)

            loss_side = torch.tensor(0.0, device=device)

            side_maps = getattr(model, "_last_side_maps", None)

            if side_maps is not None:
                valid_side_count = 0

                for side in side_maps:
                    edge_gt_side = F.interpolate(
                        edge_gt,
                        size=side.shape[-2:],
                        mode="nearest"
                    )

                    loss_side = loss_side + criterion_edge(side, edge_gt_side)
                    valid_side_count += 1

                if valid_side_count > 0:
                    loss_side = loss_side / valid_side_count

            loss_edge_iou = edge_iou_loss(seg_logits, masks, k=EDGE_BAND_K)

            loss_edge_total = loss_edge_main + LAMBDA_SIDE * loss_side

            loss = (
                loss_seg
                + LAMBDA_EDGE * loss_edge_total
                + GAMMA_EDGE_IOU * loss_edge_iou
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                print(
                    f"{epoch + 1}-{i}-loss={loss.item():.6f} "
                    f"(seg={loss_seg.item():.6f}, "
                    f"edge_main={loss_edge_main.item():.6f}, "
                    f"side={loss_side.item():.6f}, "
                    f"edgeIoU={loss_edge_iou.item():.6f})"
                )

            running_loss += loss.item() * images.size(0)
            running_seg_loss += loss_seg.item() * images.size(0)
            running_edge_main_loss += loss_edge_main.item() * images.size(0)
            running_side_loss += loss_side.item() * images.size(0)
            running_edge_iou_loss += loss_edge_iou.item() * images.size(0)

            with torch.no_grad():
                probs = torch.sigmoid(seg_logits)
                preds = (probs > 0.5).float()

                preds_flat = preds.reshape(-1)
                masks_flat = masks.reshape(-1)

                total_correct += (preds_flat == masks_flat).sum().item()
                total_pixels += masks_flat.numel()

                tp = ((preds_flat == 1) & (masks_flat == 1)).sum().item()
                fp = ((preds_flat == 1) & (masks_flat == 0)).sum().item()
                fn = ((preds_flat == 0) & (masks_flat == 1)).sum().item()

                total_tp += tp
                total_fp += fp
                total_fn += fn

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_seg_loss = running_seg_loss / len(data_loader.dataset)
        epoch_edge_main_loss = running_edge_main_loss / len(data_loader.dataset)
        epoch_side_loss = running_side_loss / len(data_loader.dataset)
        epoch_edge_iou_loss = running_edge_iou_loss / len(data_loader.dataset)

        pa = total_correct / (total_pixels + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)

        precision = (
            total_tp / (total_tp + total_fp + 1e-6)
            if (total_tp + total_fp) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall + 1e-6)
            if (precision + recall) > 0
            else 0.0
        )

        iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"loss={epoch_loss:.6f} seg_loss={epoch_seg_loss:.6f} "
            f"edge_main_loss={epoch_edge_main_loss:.6f} "
            f"side_loss={epoch_side_loss:.6f} "
            f"edge_iou_loss={epoch_edge_iou_loss:.6f} "
            f"PA={pa:.4f} Recall={recall:.4f} "
            f"Precision={precision:.4f} F1={f1:.4f} IoU={iou:.4f} "
            f"time={epoch_time:.2f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "seg_loss": epoch_seg_loss,
            "edge_main_loss": epoch_edge_main_loss,
            "side_loss": epoch_side_loss,
            "edge_iou_loss": epoch_edge_iou_loss,
            "PA": pa,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "IoU": iou,
            "epoch_time_sec": epoch_time,
        })

        torch.save(model.state_dict(), weight_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_loss_epoch = epoch + 1

            torch.save(model.state_dict(), best_loss_weight_path)

            print(
                f"*** Update best-loss model at epoch {best_loss_epoch} "
                f"(loss={best_loss:.6f})"
            )

        if iou > best_iou:
            best_iou = iou
            best_iou_epoch = epoch + 1

            torch.save(model.state_dict(), best_iou_weight_path)

            print(
                f"*** Update best-IoU model at epoch {best_iou_epoch} "
                f"(IoU={best_iou:.6f})"
            )

    save_training_history(
        history=history,
        best_loss_epoch=best_loss_epoch,
        best_loss=best_loss,
        best_iou_epoch=best_iou_epoch,
        best_iou=best_iou,
    )

    print(
        f"Training done. "
        f"Best loss epoch = {best_loss_epoch}, best loss = {best_loss:.6f}; "
        f"Best IoU epoch = {best_iou_epoch}, best IoU = {best_iou:.6f}"
    )

    return best_iou_epoch, history


def save_training_history(
    history,
    best_loss_epoch,
    best_loss,
    best_iou_epoch,
    best_iou
):
    log_path = metrics_dir / "metrics_history.csv"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,loss,seg_loss,edge_main_loss,side_loss,edge_iou_loss,"
            "PA,Recall,Precision,F1,IoU,epoch_time_sec\n"
        )

        for h in history:
            f.write(
                f"{h['epoch']},{h['loss']:.6f},"
                f"{h['seg_loss']:.6f},{h['edge_main_loss']:.6f},"
                f"{h['side_loss']:.6f},{h['edge_iou_loss']:.6f},"
                f"{h['PA']:.6f},{h['Recall']:.6f},{h['Precision']:.6f},"
                f"{h['F1']:.6f},{h['IoU']:.6f},{h['epoch_time_sec']:.4f}\n"
            )

    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    seg_losses = [h["seg_loss"] for h in history]
    edge_main_losses = [h["edge_main_loss"] for h in history]
    side_losses = [h["side_loss"] for h in history]
    edge_iou_losses = [h["edge_iou_loss"] for h in history]
    pas = [h["PA"] for h in history]
    recalls = [h["Recall"] for h in history]
    precisions = [h["Precision"] for h in history]
    f1s = [h["F1"] for h in history]
    ious = [h["IoU"] for h in history]

    plt.figure()
    plt.plot(epochs, losses, label="Loss")
    plt.plot(epochs, seg_losses, label="Seg loss")
    plt.plot(epochs, edge_main_losses, label="Edge main loss")
    plt.plot(epochs, side_losses, label="Side loss")
    plt.plot(epochs, edge_iou_losses, label="Edge IoU loss")
    plt.plot(epochs, pas, label="PA")
    plt.plot(epochs, recalls, label="Recall")
    plt.plot(epochs, precisions, label="Precision")
    plt.plot(epochs, f1s, label="F1")
    plt.plot(epochs, ious, label="IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Metric / loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(metrics_dir / "metrics_curve.png", dpi=300)
    plt.close()

    avg_epoch_time = float(np.mean([h["epoch_time_sec"] for h in history]))
    best_info_path = metrics_dir / "best_epoch.txt"

    with open(best_info_path, "w", encoding="utf-8") as f:
        f.write(f"best_loss_epoch={best_loss_epoch}\n")
        f.write(f"best_loss={best_loss:.6f}\n")
        f.write(f"best_iou_epoch={best_iou_epoch}\n")
        f.write(f"best_iou={best_iou:.6f}\n")
        f.write(f"average_epoch_time_sec={avg_epoch_time:.4f}\n")
        f.write(f"seed={SEED}\n")
        f.write(f"model_name={MODEL_NAME}\n")
        f.write(f"dataset_source={DATASET_SOURCE}\n")
        f.write(f"input_channels={IN_CHANNELS}\n")
        f.write(f"num_classes={NUM_CLASSES}\n")
        f.write(f"patch_size={PATCH_SIZE}\n")
        f.write(f"batch_size={BATCH_SIZE}\n")
        f.write(f"alpha_boundary_weight={ALPHA_BOUNDARY_WEIGHT}\n")
        f.write(f"lambda_edge={LAMBDA_EDGE}\n")
        f.write(f"lambda_side={LAMBDA_SIDE}\n")
        f.write(f"gamma_edge_iou={GAMMA_EDGE_IOU}\n")

        for h in history:
            if h["epoch"] == best_iou_epoch:
                f.write("\n[Best IoU epoch metrics]\n")
                f.write(f"epoch={h['epoch']}\n")
                f.write(f"loss={h['loss']:.6f}\n")
                f.write(f"seg_loss={h['seg_loss']:.6f}\n")
                f.write(f"edge_main_loss={h['edge_main_loss']:.6f}\n")
                f.write(f"side_loss={h['side_loss']:.6f}\n")
                f.write(f"edge_iou_loss={h['edge_iou_loss']:.6f}\n")
                f.write(f"PA={h['PA']:.6f}\n")
                f.write(f"Recall={h['Recall']:.6f}\n")
                f.write(f"Precision={h['Precision']:.6f}\n")
                f.write(f"F1={h['F1']:.6f}\n")
                f.write(f"IoU={h['IoU']:.6f}\n")
                f.write(f"epoch_time_sec={h['epoch_time_sec']:.4f}\n")
                break


# =========================================================
# Export predictions from the best-IoU checkpoint
# =========================================================
def export_best_predictions(
    model,
    dataset,
    best_weight_path: Path
):
    if not best_weight_path.exists():
        print("Best-IoU weight file not found. Skip export.")
        return

    print("Loading best-IoU model for export...")

    state = torch.load(best_weight_path, map_location=device)

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    if hasattr(dataset, "augment"):
        dataset.augment = False

    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(dataset)), desc="Export predictions"):
            sample = dataset[idx]

            if isinstance(sample, (tuple, list)):
                img = sample[0]
                mask = sample[1]
            else:
                raise RuntimeError("Dataset should return at least (img, mask).")

            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img).float()

            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask).float()

            if hasattr(dataset, "samples"):
                try:
                    img_path, _ = dataset.samples[idx]
                    name = Path(img_path).stem + ".png"
                except Exception:
                    name = f"{idx:05d}.png"
            elif hasattr(dataset, "names"):
                try:
                    name = f"{dataset.names[idx]}.png"
                except Exception:
                    name = f"{idx:05d}.png"
            else:
                name = f"{idx:05d}.png"

            inp = img.unsqueeze(0).to(device).float()

            outputs = model(inp)
            seg_logits, edge_logits = ensure_output_tuple(outputs)

            if seg_logits.shape[-2:] != img.shape[-2:]:
                seg_logits = F.interpolate(
                    seg_logits,
                    size=img.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            prob = torch.sigmoid(seg_logits)[0, 0]
            pred = (prob > 0.5).float().cpu()

            save_image(pred, best_pred_dir / name)

            if edge_logits is not None:
                if edge_logits.shape[-2:] != img.shape[-2:]:
                    edge_logits = F.interpolate(
                        edge_logits,
                        size=img.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                edge_prob = torch.sigmoid(edge_logits)[0, 0].detach().cpu().clamp(0, 1)
                save_image(edge_prob.unsqueeze(0), edge_pred_dir / name)

            vis_img = img.detach().cpu().clone()

            if vis_img.shape[0] >= 3:
                vis_img = vis_img[:3, :, :]
            elif vis_img.shape[0] == 1:
                vis_img = vis_img.repeat(3, 1, 1)

            vis_mask = mask.detach().cpu().clone()

            if vis_mask.dim() == 3 and vis_mask.shape[0] == 1:
                vis_mask = vis_mask.repeat(3, 1, 1)
            elif vis_mask.dim() == 2:
                vis_mask = vis_mask.unsqueeze(0).repeat(3, 1, 1)
            elif vis_mask.dim() == 3 and vis_mask.shape[0] >= 3:
                vis_mask = vis_mask[:3, :, :]

            vis_pred = pred.unsqueeze(0).repeat(3, 1, 1)

            triplet = torch.cat(
                [vis_img, vis_mask, vis_pred],
                dim=2
            )

            save_image(triplet, compare_dir / name)

    print("Best-IoU predictions and comparison images saved.")


# =========================================================
# Evaluation: global metrics + IoU_edge + Boundary IoU + HD95 + edge-head metrics
# =========================================================
def evaluate_metrics(
    model,
    dataset,
    best_weight_path: Path,
    k: int = 5,
    save_dir=None
):
    if not best_weight_path.exists():
        raise FileNotFoundError(
            f"Best-IoU weight file not found: {best_weight_path}"
        )

    print("Loading best-IoU model for final evaluation...")

    state = torch.load(best_weight_path, map_location=device)

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    if hasattr(dataset, "augment"):
        dataset.augment = False

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    total_pixels = 0
    total_correct = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_pixels_edge = 0
    total_correct_edge = 0
    total_tp_edge = 0
    total_fp_edge = 0
    total_fn_edge = 0

    head_total_pixels = 0
    head_total_correct = 0
    head_tp = 0
    head_fp = 0
    head_fn = 0

    boundary_iou_list = []
    hd95_list = []

    skipped_no_gt_boundary = 0
    pred_empty_penalty_count = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Evaluate best-IoU model"):
            images, masks = unpack_batch(batch)

            images = images.to(device).float()
            masks = ensure_mask_bchw(masks.to(device))

            outputs = model(images)
            seg_logits, edge_logits = ensure_output_tuple(outputs)

            if seg_logits.shape[-2:] != masks.shape[-2:]:
                seg_logits = F.interpolate(
                    seg_logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            probs = torch.sigmoid(seg_logits)
            preds = (probs > 0.5).float()

            preds_flat = preds.reshape(-1)
            masks_flat = masks.reshape(-1)

            total_correct += (preds_flat == masks_flat).sum().item()
            total_pixels += masks_flat.numel()

            tp = ((preds_flat == 1) & (masks_flat == 1)).sum().item()
            fp = ((preds_flat == 1) & (masks_flat == 0)).sum().item()
            fn = ((preds_flat == 0) & (masks_flat == 1)).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

            band = get_edge_band(masks, k=k)
            band_flat = band.reshape(-1)

            preds_edge = preds_flat[band_flat == 1]
            masks_edge = masks_flat[band_flat == 1]

            if preds_edge.numel() > 0:
                total_correct_edge += (preds_edge == masks_edge).sum().item()
                total_pixels_edge += masks_edge.numel()

                tp_e = ((preds_edge == 1) & (masks_edge == 1)).sum().item()
                fp_e = ((preds_edge == 1) & (masks_edge == 0)).sum().item()
                fn_e = ((preds_edge == 0) & (masks_edge == 1)).sum().item()

                total_tp_edge += tp_e
                total_fp_edge += fp_e
                total_fn_edge += fn_e

            if edge_logits is not None:
                if edge_logits.shape[-2:] != band.shape[-2:]:
                    edge_logits = F.interpolate(
                        edge_logits,
                        size=band.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                edge_probs = torch.sigmoid(edge_logits)
                edge_preds = (edge_probs > 0.5).float()

                edge_preds_flat = edge_preds.reshape(-1)
                band_flat_all = band.reshape(-1)

                head_total_correct += (edge_preds_flat == band_flat_all).sum().item()
                head_total_pixels += band_flat_all.numel()

                tp_h = ((edge_preds_flat == 1) & (band_flat_all == 1)).sum().item()
                fp_h = ((edge_preds_flat == 1) & (band_flat_all == 0)).sum().item()
                fn_h = ((edge_preds_flat == 0) & (band_flat_all == 1)).sum().item()

                head_tp += tp_h
                head_fp += fp_h
                head_fn += fn_h

            pred_np = (
                preds[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            mask_np = (
                masks[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            gt_boundary = mask_to_boundary_np(mask_np)

            if gt_boundary.sum() == 0:
                skipped_no_gt_boundary += 1
                continue

            pred_boundary = mask_to_boundary_np(pred_np)

            if pred_boundary.sum() == 0:
                pred_empty_penalty_count += 1

            b_iou = boundary_iou_np(
                pred_np,
                mask_np,
                dilation_iter=BOUNDARY_DILATION_ITER
            )

            hd95_value = hd95_np(
                pred_np,
                mask_np,
                pixel_size=PIXEL_SIZE_M
            )

            if b_iou is not None and np.isfinite(b_iou):
                boundary_iou_list.append(float(b_iou))

            if hd95_value is not None and np.isfinite(hd95_value):
                hd95_list.append(float(hd95_value))

    PA = total_correct / (total_pixels + 1e-6)
    Recall = total_tp / (total_tp + total_fn + 1e-6)

    Precision = (
        total_tp / (total_tp + total_fp + 1e-6)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    F1 = (
        2 * Precision * Recall / (Precision + Recall + 1e-6)
        if (Precision + Recall) > 0
        else 0.0
    )

    IoU = total_tp / (total_tp + total_fp + total_fn + 1e-6)

    PA_edge = total_correct_edge / (total_pixels_edge + 1e-6)

    Recall_edge = (
        total_tp_edge / (total_tp_edge + total_fn_edge + 1e-6)
        if (total_tp_edge + total_fn_edge) > 0
        else 0.0
    )

    Precision_edge = (
        total_tp_edge / (total_tp_edge + total_fp_edge + 1e-6)
        if (total_tp_edge + total_fp_edge) > 0
        else 0.0
    )

    F1_edge = (
        2 * Precision_edge * Recall_edge / (Precision_edge + Recall_edge + 1e-6)
        if (Precision_edge + Recall_edge) > 0
        else 0.0
    )

    IoU_edge = total_tp_edge / (
        total_tp_edge + total_fp_edge + total_fn_edge + 1e-6
    )

    PA_head = head_total_correct / (head_total_pixels + 1e-6)

    Recall_head = (
        head_tp / (head_tp + head_fn + 1e-6)
        if (head_tp + head_fn) > 0
        else 0.0
    )

    Precision_head = (
        head_tp / (head_tp + head_fp + 1e-6)
        if (head_tp + head_fp) > 0
        else 0.0
    )

    F1_head = (
        2 * Precision_head * Recall_head / (Precision_head + Recall_head + 1e-6)
        if (Precision_head + Recall_head) > 0
        else 0.0
    )

    IoU_head = (
        head_tp / (head_tp + head_fp + head_fn + 1e-6)
        if (head_tp + head_fp + head_fn) > 0
        else 0.0
    )

    Boundary_IoU = (
        float(np.mean(boundary_iou_list))
        if len(boundary_iou_list) > 0
        else None
    )

    HD95_m = (
        float(np.mean(hd95_list))
        if len(hd95_list) > 0
        else None
    )

    results = {
        "weight": best_weight_path.name,
        "PA": PA,
        "Recall": Recall,
        "Precision": Precision,
        "F1": F1,
        "IoU": IoU,
        "PA_edge": PA_edge,
        "Recall_edge": Recall_edge,
        "Precision_edge": Precision_edge,
        "F1_edge": F1_edge,
        "IoU_edge": IoU_edge,
        "Boundary_IoU": Boundary_IoU,
        "HD95_m": HD95_m,
        "valid_boundary_iou_samples": len(boundary_iou_list),
        "valid_hd95_samples": len(hd95_list),
        "skipped_no_gt_boundary": skipped_no_gt_boundary,
        "pred_empty_penalty_count": pred_empty_penalty_count,
        "PA_head": PA_head,
        "Recall_head": Recall_head,
        "Precision_head": Precision_head,
        "F1_head": F1_head,
        "IoU_head": IoU_head,
        "pixel_size_m": PIXEL_SIZE_M,
        "boundary_dilation_iter": BOUNDARY_DILATION_ITER,
        "edge_band_k": k,
        "scipy_available": SCIPY_AVAILABLE,
    }

    print("\n========== Final Evaluation Metrics: Best-IoU Epoch ==========")

    for k_, v in results.items():
        if isinstance(v, float):
            print(f"{k_}={v:.6f}")
        else:
            print(f"{k_}={v}")

    print("==============================================================\n")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(
            save_dir / "evaluation_metrics_best_iou.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        with open(
            save_dir / "evaluation_metrics_best_iou.txt",
            "w",
            encoding="utf-8"
        ) as f:
            for k_, v in results.items():
                f.write(f"{k_}={v}\n")

    return results


# ================== Main workflow ==================
if __name__ == "__main__":
    input_size = (
        PROFILE_INPUT_BATCH,
        IN_CHANNELS,
        PATCH_SIZE[0],
        PATCH_SIZE[1]
    )

    save_environment_info(metrics_dir)

    profile_model_complexity(
        model,
        input_size=input_size,
        device=device,
        save_dir=metrics_dir
    )

    benchmark_inference_speed(
        model,
        input_size=input_size,
        device=device,
        warmup=BENCHMARK_WARMUP,
        repeats=BENCHMARK_REPEATS,
        save_dir=metrics_dir
    )

    best_iou_epoch, history = train(
        model,
        data_loader,
        criterion_seg,
        criterion_edge,
        optimizer,
        num_epochs=NUM_EPOCHS
    )

    export_best_predictions(
        model,
        train_dataset,
        best_iou_weight_path
    )

    evaluate_metrics(
        model,
        train_dataset,
        best_iou_weight_path,
        k=EDGE_BAND_K,
        save_dir=metrics_dir
    )
