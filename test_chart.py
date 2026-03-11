# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error as mse
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as ms_ssim_func

# ====== 新模型入口（与训练代码保持一致）======
from model_two_stage import TeacherAE, StudentEncoder, StudentWithTeacher

# ========== 配置区 ==========
save_images = True     # True：保存图像+输出指标；False：仅输出指标
batch_size  = 1
image_size  = 256

# ====== 路径（按需修改）======
test_x_data_dir   = "/root/autodl-tmp/project/dataset/charts/test/input"
test_y_data_dir   = "/root/autodl-tmp/project/dataset/charts/test/target"
teacher_weights  = "/root/autodl-tmp/project/Ours_autoencoder/AutoEncode/checkpoint_chart/autoencoder_epoch_150.pth"
student_ckpt_path = "/root/autodl-tmp/project/Ours_autoencoder/model_two_stage_chart_wolat/student_30_0.683103660107.pth"  # 选一个Top-K里的学生权重
fake_save_path   = "/root/autodl-tmp/project/Ours_autoencoder/model_two_stage_chart_wolat/chart_test"

# ========== 预处理 ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((image_size, image_size)),
])

def convert_to_uint8(imgs):
    return (imgs * 255).clamp(0, 255).to(torch.uint8)

# ========== 数据集：x/y 成对 ==========
class XYPairDataset(Dataset):
    def __init__(self, x_dir, y_dir, transform=None):
        self.x_paths = sorted([os.path.join(x_dir, f) for f in os.listdir(x_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.y_paths = sorted([os.path.join(y_dir, f) for f in os.listdir(y_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        assert len(self.x_paths) == len(self.y_paths), "x 与 y 文件数不一致"
        self.transform = transform

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        x = Image.open(self.x_paths[idx]).convert("RGB")
        y = Image.open(self.y_paths[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        img_name = os.path.basename(self.x_paths[idx])
        return x, y, img_name

# ========== 加载“学生编码器”权重的辅助函数 ==========
def load_student_into_model(model_wrapped: nn.Module, student_ckpt_path: str):
    """
    训练阶段TopK只保存了“学生编码器”的state_dict。本函数将其载入到 model.student。
    兼容 DataParallel 与非 DataParallel 两种情况，以及 state_dict 的 'module.' 前缀差异。
    """
    assert os.path.exists(student_ckpt_path), f"No checkpoint found at {student_ckpt_path}"
    sd = torch.load(student_ckpt_path, map_location="cpu")

    # 取出未被DataParallel包裹的“真实模型”
    m = model_wrapped
    if isinstance(m, nn.DataParallel):
        m = m.module

    # 得到学生编码器对象（可能被DataParallel包着）
    stu = m.student
    target = stu.module if isinstance(stu, nn.DataParallel) else stu

    # 处理可能的 'module.' 前缀
    if len(sd) > 0 and list(sd.keys())[0].startswith("module."):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

    missing, unexpected = target.load_state_dict(sd, strict=False)
    if missing:
        print(f"[Warn] Missing keys in student: {missing}")
    if unexpected:
        print(f"[Warn] Unexpected keys in student: {unexpected}")
    print(f"[OK] Loaded student weights from {student_ckpt_path}")

# ========== 评估指标 ==========
def compute_metrics(fake, real):
    # 转为 NumPy 的 0~255 浮点
    fake_np = (fake.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.float32)
    real_np = (real.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.float32)

    psnr_vals, mse_vals, rmse_vals, ms_ssim_vals, ssim_vals = [], [], [], [], []
    for i in range(fake_np.shape[0]):
        f_img = fake_np[i]
        r_img = real_np[i]
        psnr_vals.append(psnr(r_img, f_img, data_range=255.0))
        mse_val = mse(r_img, f_img)
        mse_vals.append(mse_val)
        rmse_vals.append(np.sqrt(mse_val))
        ssim_vals.append(ssim(r_img, f_img, channel_axis=-1, data_range=255.0))
        # MS-SSIM 用 [0,1] 的Tensor
        ms_ssim_vals.append(ms_ssim_func(fake[i].unsqueeze(0), real[i].unsqueeze(0), data_range=1.0).item())

    return {
        "psnr": np.mean(psnr_vals),
        "mse": np.mean(mse_vals),
        "rmse": np.mean(rmse_vals),
        "ssim": np.mean(ssim_vals),
        "ms_ssim": np.mean(ms_ssim_vals),
        "lpips": lpips_loss(fake, real).mean().item()
    }

# ========== 初始化模型 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 与训练一致的结构超参
Cs = (64, 128, 256, 512, 1024)
z_channels = 256

# 1) TeacherAE：加载权重并冻结
teacher = TeacherAE(
    c_y=3, Cs=Cs, up_mode="interp", z_channels=z_channels, groups=8
).to(device)
assert os.path.exists(teacher_weights), f"Teacher weights not found: {teacher_weights}"
t_sd = torch.load(teacher_weights, map_location="cpu")
if len(t_sd) > 0 and list(t_sd.keys())[0].startswith("module."):
    t_sd = {k.replace("module.", ""): v for k, v in t_sd.items()}
# 放宽strict以适配可能的非关键buffer
teacher.load_state_dict(t_sd, strict=False)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

# 2) StudentEncoder：空模型先建图
student = StudentEncoder(
    c_in=3, Cs=Cs, z_channels=z_channels, groups=8
).to(device)

# 3) 组合：Student + 冻结 Teacher（推理不需要teacher encoder）
combined = StudentWithTeacher(student_encoder=student, teacher_ae=teacher, freeze_teacher=True).to(device)

# 多卡（可选）
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    combined = nn.DataParallel(combined)

# 加载学生权重（TopK保存的就是学生state_dict）
load_student_into_model(combined, student_ckpt_path)
combined.eval()

# 评估器
lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
fid_metric = FrechetInceptionDistance(feature=64).to(device)

# 数据与加载器
test_dataset   = XYPairDataset(test_x_data_dir, test_y_data_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

metric_records = { "psnr": [], "mse": [], "rmse": [], "ssim": [], "ms_ssim": [], "lpips": [] }
if save_images:
    os.makedirs(fake_save_path, exist_ok=True)

# ========== 推理 ==========
with torch.no_grad():
    test_loader = tqdm(test_dataloader, desc="Inferencing", leave=False)
    for x_val, y_val, img_names in test_loader:
        x_val = x_val.to(device, non_blocking=True)
        y_val = y_val.to(device, non_blocking=True)

        # 新模型前向：不需要 teacher z*
        out = combined(x_val, need_teacher_z=False)
        fake = out["y_hat"].clamp(0, 1)
        real = y_val.clamp(0, 1)

        # 指标
        batch_metrics = compute_metrics(fake, real)
        for k in metric_records:
            metric_records[k].append(batch_metrics[k])

        # FID 需要 uint8，[N,C,H,W]
        fake_uint8 = convert_to_uint8(fake)
        real_uint8 = convert_to_uint8(real)
        fid_metric.update(real_uint8, real=True)
        fid_metric.update(fake_uint8, real=False)

        # 保存图像
        if save_images:
            epoch_fake_path = os.path.join(fake_save_path, "infer_epoch")
            os.makedirs(epoch_fake_path, exist_ok=True)
            for i in range(fake.size(0)):
                save_path = os.path.join(epoch_fake_path, img_names[i])
                save_image(fake[i].cpu(), save_path)

# ========== 汇总输出 ==========
print("\n评估结果（平均 ± 标准差）：")
for k in metric_records:
    arr = np.array(metric_records[k], dtype=np.float64)
    print(f"{k.upper():<10}: {arr.mean():.4f} ± {arr.std():.4f}")

print(f"FID       : {fid_metric.compute().item():.4f}")
