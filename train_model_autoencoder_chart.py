# -*- coding: utf-8 -*-
import os
import heapq
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

# ====== 导入你上条模型文件（需包含 TeacherAE / StudentEncoder / StudentWithTeacher）======
from model_two_stage import TeacherAE, StudentEncoder, StudentWithTeacher

# ======================
# 超参数 & 路径
# ======================
lr            = 1e-4
batch_size    = 4
epochs        = 101
image_size    = 256
alpha_latent  = 1.0     # 潜在对齐损系数
beta_recon    = 1.0     # 重建 L1 损系数
TOP_K         = 5
z_channels    = 256

# 数据路径（修改为你的）
x_data_dir        = "/root/autodl-tmp/project/dataset/charts/train/input"
y_data_dir        = "/root/autodl-tmp/project/dataset/charts/train/target"
test_x_data_dir   = "/root/autodl-tmp/project/dataset/charts/test/input"
test_y_data_dir   = "/root/autodl-tmp/project/dataset/charts/test/target"

# 权重与输出目录（修改为你的）
teacher_weights   = "/root/autodl-tmp/project/Ours_autoencoder/AutoEncode/checkpoint_chart/autoencoder_epoch_150.pth"   # <<< 必填：教师AE权重
save_root         = "/root/autodl-tmp/project/Ours_autoencoder/model_two_stage_chart_l1_wrecon"
model_dir         = os.path.join(save_root, "model")
optim_dir         = os.path.join(save_root, "optimizer")
sched_dir         = os.path.join(save_root, "scheduler")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(optim_dir, exist_ok=True)
os.makedirs(sched_dir, exist_ok=True)

# 断点继续（可选）
resume_model_ckpt = ""   # e.g. f"{model_dir}/student_100_0.900000000000.pth"
resume_optim_ckpt = ""   # e.g. f"{optim_dir}/optim_100_0.900000000000.pth"
resume_sched_ckpt = ""   # e.g. f"{sched_dir}/sched_100_0.900000000000.pth"
resume_epoch      = 0    # 从第几个 epoch 继续

# ======================
# Top-K Saver
# ======================
def _safe_remove(p):
    try:
        os.remove(p)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[TopKSaver] remove fail {p}: {e}")

class TopKSaver:
    """按指标保留前K个checkpoint，并删除其余文件。"""
    def __init__(self, k, model_dir, optim_dir, scheduler_dir=None,
                 model_prefix="student", optim_prefix="optim", scheduler_prefix="sched"):
        self.k = k
        self.heap = []  # 最小堆：(score, epoch, mpath, opath, spath or None)
        self.model_dir = model_dir
        self.optim_dir = optim_dir
        self.scheduler_dir = scheduler_dir
        self.model_prefix = model_prefix
        self.optim_prefix = optim_prefix
        self.scheduler_prefix = scheduler_prefix
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(optim_dir, exist_ok=True)
        if scheduler_dir:
            os.makedirs(scheduler_dir, exist_ok=True)

    def push_if_topk(self, score, epoch, model_state, optim_state, scheduler_state=None):
        # 若堆未满或更优于当前最差，才写盘
        if len(self.heap) < self.k or score > self.heap[0][0]:
            # 先保存
            mpath = os.path.join(self.model_dir, f"{self.model_prefix}_{epoch}_{score:.12f}.pth")
            opath = os.path.join(self.optim_dir, f"{self.optim_prefix}_{epoch}_{score:.12f}.pth")
            torch.save(model_state, mpath)
            torch.save(optim_state, opath)

            spath = None
            if self.scheduler_dir and scheduler_state is not None:
                spath = os.path.join(self.scheduler_dir, f"{self.scheduler_prefix}_{epoch}_{score:.12f}.pth")
                torch.save(scheduler_state, spath)

            heapq.heappush(self.heap, (score, epoch, mpath, opath, spath))

            # 超过K则弹出并删除最差
            while len(self.heap) > self.k:
                _, _, old_m, old_o, old_s = heapq.heappop(self.heap)
                _safe_remove(old_m); _safe_remove(old_o)
                if old_s: _safe_remove(old_s)
            return True
        return False

# ======================
# 数据
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((image_size, image_size)),
])

class XYDataset(Dataset):
    def __init__(self, x_dir, y_dir, transform=None):
        self.x_paths = sorted([os.path.join(x_dir, f) for f in os.listdir(x_dir)
                               if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        self.y_paths = sorted([os.path.join(y_dir, f) for f in os.listdir(y_dir)
                               if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        assert len(self.x_paths) == len(self.y_paths), "x 与 y 文件数不一致"
        self.transform = transform

    def __len__(self): return len(self.x_paths)

    def __getitem__(self, idx):
        x = Image.open(self.x_paths[idx]).convert("RGB")
        y = Image.open(self.y_paths[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

def compute_ssim_batch(img1, img2):
    """
    img1, img2: (N,C,H,W) in [0,1]
    """
    a = img1.permute(0,2,3,1).cpu().numpy()
    b = img2.permute(0,2,3,1).cpu().numpy()
    vals = []
    for i in range(a.shape[0]):
        vals.append(ssim(a[i], b[i], data_range=1.0, channel_axis=-1))
    return float(np.mean(vals))

# ======================
# 模型构建
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 与TeacherAE训练一致的通道/尺度配置
Cs = (64,128,256,512,1024)

# 1) TeacherAE（只用 encode/decode，冻结）
teacher = TeacherAE(
        c_y=3, Cs=Cs,
        up_mode="interp", z_channels=z_channels, groups=8
    ).to(device)
assert os.path.exists(teacher_weights), f"Teacher weights not found: {teacher_weights}"
t_sd = torch.load(teacher_weights, map_location="cpu")
# 允许移除 'module.' 前缀
def _strip_module(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }
try:
    teacher.load_state_dict(_strip_module(t_sd), strict=True)
except Exception:
    teacher.load_state_dict(t_sd, strict=False)
teacher.eval()
for p in teacher.parameters(): p.requires_grad = False

# 2) StudentEncoder
student = StudentEncoder(
        c_in=3, Cs=Cs,
        z_channels=z_channels, groups=8
    ).to(device)
# 多卡（可选）
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    student = nn.DataParallel(student)
student = student.to(device)

# 3) 组合封装（前向里会用 teacher.decode(ẑ)）
model = StudentWithTeacher(student_encoder=student, teacher_ae=teacher, freeze_teacher=True).to(device)

def get_student_state_dict(model_wrapped: nn.Module):
    """只保存学生编码器参数（避免把冻结的teacher也存进checkpoint）"""
    m = model_wrapped
    if isinstance(m, nn.DataParallel): m = m.module
    stu = m.student
    if isinstance(stu, nn.DataParallel): stu = stu.module
    return stu.state_dict()

# ======================
# 优化器 & 调度器（按 batch 退火）
# ======================
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

# 构建数据集后再确定 steps_per_epoch
train_set = XYDataset(x_data_dir, y_data_dir, transform=transform)
test_set  = XYDataset(test_x_data_dir, test_y_data_dir, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4, drop_last=True, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)

steps_per_epoch = len(train_loader)
total_steps = max(1, epochs * steps_per_epoch)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps, eta_min=1e-5
)

# 断点恢复（可选）
def load_checkpoint(model, optimizer, scheduler,
                    model_ckpt, optim_ckpt, sched_ckpt):
    if model_ckpt and os.path.exists(model_ckpt):
        print(f"[Resume] Loading model from {model_ckpt}")
        sd = torch.load(model_ckpt, map_location="cpu")
        # 仅恢复学生
        student.load_state_dict(sd, strict=False)  # 注意：直接写入 student
    if optim_ckpt and os.path.exists(optim_ckpt):
        print(f"[Resume] Loading optimizer from {optim_ckpt}")
        optimizer.load_state_dict(torch.load(optim_ckpt, map_location="cpu"))
    if sched_ckpt and os.path.exists(sched_ckpt):
        print(f"[Resume] Loading scheduler from {sched_ckpt}")
        scheduler.load_state_dict(torch.load(sched_ckpt, map_location="cpu"))
    return model, optimizer, scheduler

if resume_model_ckpt or resume_optim_ckpt or resume_sched_ckpt:
    model, optimizer, scheduler = load_checkpoint(
        model, optimizer, scheduler,
        resume_model_ckpt, resume_optim_ckpt, resume_sched_ckpt
    )

# ======================
# 损失
# ======================
criterion_L1 = nn.L1Loss().to(device)

def training_step(x, y):
    out = model(x, y, need_teacher_z=True)  # 训练需要 teacher z*
    y_hat, z_hat, z_star = out["y_hat"], out["z_hat"], out["z_star"]
    #loss_latent =  1.0 - F.cosine_similarity(z_hat, z_star, dim=1).mean()
    loss_latent = criterion_L1(z_hat, z_star)
    loss_recon  = criterion_L1(y_hat, y)
    loss = alpha_latent * loss_latent + beta_recon * loss_recon
    logs = {"latent": float(loss_latent.detach()),
            "recon":  float(loss_recon.detach()),
            "total":  float(loss.detach())}
    return loss, logs

# ======================
# Top-K 保存器
# ======================
topk_saver = TopKSaver(
    k=TOP_K,
    model_dir=model_dir,
    optim_dir=optim_dir,
    scheduler_dir=sched_dir,            # 若不想保存调度器，改为 None
    model_prefix="student",
    optim_prefix="optim",
    scheduler_prefix="sched"
)

# ======================
# 训练 & 验证 & Top-K 保存
# ======================
best_ssim = 0.0
global_step = 0

for epoch in range(resume_epoch, epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Training {epoch+1}/{epochs}", leave=False)
    for i, (x, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss, logs = training_step(x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 按 batch 调度
        scheduler.step()
        global_step += 1

        lr_now = scheduler.get_last_lr()[0]
        pbar.set_postfix({**logs, "lr": f"{lr_now:.2e}"})

    # 每 5 个 epoch 进行一次验证 + TopK 保存（与参考代码一致）
    if epoch % 5 == 0:
        model.eval()
        total_ssim = 0.0
        with torch.no_grad():
            vbar = tqdm(test_loader, desc="Testing", leave=False)
            for x_val, y_val in vbar:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                # 验证时不需要 teacher z*，只做前向生成
                out = model(x_val, need_teacher_z=False)
                y_hat = out["y_hat"].clamp(0, 1)
                total_ssim += compute_ssim_batch(y_hat, y_val)

        avg_ssim = total_ssim / len(test_loader)

        # 只把“学生参数”送进 TopK（避免把冻结的 teacher 一起存）
        saved = topk_saver.push_if_topk(
            score=avg_ssim,
            epoch=epoch,
            model_state=get_student_state_dict(model),    # 仅学生
            optim_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict()
        )
        if saved:
            best_ssim = max(best_ssim, avg_ssim)
            print(f"[Top-{TOP_K}] saved at epoch {epoch}: SSIM={avg_ssim:.6f}")
        print(f"Epoch [{epoch+1}/{epochs}]  Val SSIM: {avg_ssim:.4f} | Best: {best_ssim:.4f}")

print("Training done.")
