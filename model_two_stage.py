# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 小工具
# =========================================================
def _pad_to(x, ref):
    dh, dw = ref.shape[-2] - x.shape[-2], ref.shape[-1] - x.shape[-1]
    if dh or dw:
        x = F.pad(x, (0, max(dw, 0), 0, max(dh, 0)))
        if dh < 0 or dw < 0:
            x = x[..., :ref.shape[-2], :ref.shape[-1]]
    return x

def count_params(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def pretty_params(n: int) -> str:
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.2f}K"
    return str(n)

# =========================================================
# 基础层
# =========================================================
class ConvGNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, groups=8, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, padding=k//2, bias=False)
        self.gn   = nn.GroupNorm(min(groups, c_out), c_out)
        self.act  = nn.SiLU(inplace=False) if act else nn.Identity()
    def forward(self, x): return self.act(self.gn(self.conv(x)))

# =========================================================
# ConvNeXt 风格 FFN（1x1 → DWConv → 1x1）+ LayerScale + DropPath
# =========================================================
class ConvNeXtFFN(nn.Module):
    def __init__(self, c, expand=2.0, dw_kernel=3, groups=8,
                 drop_path=0.0, layerscale=1e-4):
        super().__init__()
        hid = int(round(c * expand))
        self.pw1 = nn.Conv2d(c,  hid, 1, bias=False)
        self.dw  = nn.Conv2d(hid, hid, dw_kernel, padding=dw_kernel//2, groups=hid, bias=False)
        self.act = nn.SiLU(True)
        self.pw2 = nn.Conv2d(hid, c, 1, bias=False)
        self.gn  = nn.GroupNorm(min(groups, c), c)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * layerscale) if layerscale is not None else None
        self.drop_path = float(drop_path)

    def forward(self, x):
        id = x
        y = self.pw1(x); y = self.dw(y); y = self.act(y); y = self.pw2(y)
        if self.gamma is not None: y = self.gamma * y
        if self.training and self.drop_path > 0:
            keep = 1.0 - self.drop_path
            mask = torch.empty(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep) / keep
            y = y * mask
        y = id + self.gn(y)
        return y

# =========================================================
# 新命名的局部/全局模块（固定核大小，且不对外暴露可调参数）
# =========================================================
class LocalBlock(nn.Module):
    """PW → DW(3x3) → PW + GN + 残差"""
    def __init__(self, c, groups=8):
        super().__init__()
        # 固定 expand=2.0, drop_path=0.0, layerscale=1e-4；仅固定 3x3 DW
        self.ffn = ConvNeXtFFN(c, expand=2.0, dw_kernel=3, groups=groups,
                               drop_path=0.0, layerscale=1e-4)
    def forward(self, x): return self.ffn(x)

class GlobalBlock(nn.Module):
    """PW → DW(11x11) → PW + GN + 残差"""
    def __init__(self, c, groups=8):
        super().__init__()
        # 固定 11x11 DW
        self.ffn = ConvNeXtFFN(c, expand=2.0, dw_kernel=11, groups=groups,
                               drop_path=0.0, layerscale=1e-4)
    def forward(self, x): return self.ffn(x)

# =========================================================
# 组合块：Global(11x11) → Local(3x3)
# =========================================================
class GlobalLocalBlock(nn.Module):
    def __init__(self, c, groups=8):
        super().__init__()
        self.global_blk = GlobalBlock(c, groups=groups)
        self.local_blk  = LocalBlock(c, groups=groups)
    def forward(self, x):
        y = self.global_blk(x)   # 大核
        y = self.local_blk(y)    # 小核
        return y

# =========================================================
# 教师自编码器：y -> z -> ŷ 
# =========================================================
class TeacherAE(nn.Module):
    """
    - Encoder: 逐级下采样得到瓶颈特征，并用 1x1 Conv 压到 z_channels
    - Decoder: 逐级上采样解码，仅依赖 z（**不接任何来自 encoder 的跳连**）
    """
    def __init__(self, c_y=3, Cs=(64,128,256,512,1024),
                 up_mode="interp", z_channels=256, groups=8):
        super().__init__()
        self.levels = len(Cs) - 1
        self.up_mode = up_mode
        self.Cs = Cs

        # --- Encoder on y ---
        self.in_proj = ConvGNAct(c_y, Cs[0], k=3, groups=groups)
        self.cells_enc = nn.ModuleList([
            GlobalLocalBlock(Cs[i], groups=groups) for i in range(len(Cs))
        ])
        self.downs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(Cs[i], Cs[i+1], 3, stride=2, padding=1, bias=False),
                          nn.GroupNorm(min(groups, Cs[i+1]), Cs[i+1]), nn.SiLU(True))
            for i in range(self.levels)
        ])
        self.to_z = nn.Conv2d(Cs[-1], z_channels, 1, bias=False)
        self.z_norm = nn.GroupNorm(min(groups, z_channels), z_channels)  # 稳定 z 统计

        # --- Decoder from z (NO skip from encoder) ---
        up_in_cs = [z_channels] + [Cs[i+1] for i in reversed(range(self.levels-1))]
        up_out_cs = [Cs[i] for i in reversed(range(self.levels))]

        self.up_projs = nn.ModuleList([
            (nn.ConvTranspose2d(up_in_cs[idx], up_out_cs[idx], 2, stride=2)
             if up_mode=="deconv" else nn.Conv2d(up_in_cs[idx], up_out_cs[idx], 1))
            for idx in range(self.levels)
        ])
        self.dec_cells = nn.ModuleList([
            GlobalLocalBlock(up_out_cs[idx], groups=groups)
            for idx in range(self.levels)
        ])
        self.head = nn.Conv2d(Cs[0], c_y, 1)

    def _encode_feat(self, y):
        t = self.in_proj(y)
        t = self.cells_enc[0](t)
        for i in range(self.levels):
            t = self.downs[i](t)
            t = self.cells_enc[i+1](t)
        return t

    def encode(self, y):
        """y -> z（训练学生时通常不回传梯度）"""
        t = self._encode_feat(y)
        z = self.z_norm(self.to_z(t))
        return z

    def decode(self, z):
        """z -> ŷ（仅依赖 z，无跨瓶颈跳连）"""
        y = z
        for idx in range(self.levels):
            if self.up_mode == "deconv":
                y = self.up_projs[idx](y)
            else:
                y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
                y = self.up_projs[idx](y)
            y = self.dec_cells[idx](y)
        return self.head(y)

    def forward(self, y):
        """训练 AE 用：重建 ŷ 与潜在 z"""
        t = self._encode_feat(y)
        z = self.z_norm(self.to_z(t))
        y_hat = self.decode(z)
        return y_hat, z

# =========================================================
# 学生主干：x -> ẑ（仅编码端，投影到 z_channels）
# =========================================================
class StudentEncoder(nn.Module):
    def __init__(self, c_in=3, Cs=(64,128,256,512,1024),
                 z_channels=256, groups=8):
        super().__init__()
        self.levels = len(Cs) - 1

        self.in_proj = ConvGNAct(c_in, Cs[0], k=3, groups=groups)
        self.cells_enc = nn.ModuleList([
            GlobalLocalBlock(Cs[i], groups=groups) for i in range(len(Cs))
        ])
        self.downs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(Cs[i], Cs[i+1], 3, stride=2, padding=1, bias=False),
                          nn.GroupNorm(min(groups, Cs[i+1]), Cs[i+1]), nn.SiLU(True))
            for i in range(self.levels)
        ])
        self.to_z = nn.Conv2d(Cs[-1], z_channels, 1, bias=False)
        self.z_norm = nn.GroupNorm(min(groups, z_channels), z_channels)

    def forward(self, x):
        h = self.in_proj(x)
        h = self.cells_enc[0](h)
        for i in range(self.levels):
            h = self.downs[i](h)
            h = self.cells_enc[i+1](h)
        z_hat = self.z_norm(self.to_z(h))
        return z_hat

# =========================================================
# 训练/推理拼装：Student + 冻结 Teacher decoder
# =========================================================
class StudentWithTeacher(nn.Module):
    """
    - 训练：x -> ẑ；y -> z*（teacher.encode, 常用 no_grad）；ŷ = teacher.decode(ẑ)
      损失示例：α·||ẑ - z*|| + β·ℓ(ŷ, y)
    - 推理：只需 ŷ = teacher.decode(student(x))（teacher encoder 不参与）
    """
    def __init__(self, student_encoder: StudentEncoder, teacher_ae: TeacherAE, freeze_teacher=True):
        super().__init__()
        self.student = student_encoder
        self.teacher = teacher_ae
        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False

    def forward(self, x, y=None, need_teacher_z=True):
        z_hat = self.student(x)               # 学生预测潜在
        y_hat = self.teacher.decode(z_hat)    # 冻结的教师解码器
        out = {"y_hat": y_hat, "z_hat": z_hat}
        if (y is not None) and need_teacher_z:
            with torch.no_grad():
                z_star = self.teacher.encode(y)
            out["z_star"] = z_star
        return out

# =========================================================
# 示例损失（供参考）
# =========================================================
def student_loss(batch, model, alpha=1.0, beta=1.0, use_cos=False):
    """
    batch: {"x": ..., "y": ...}
    α: 潜在对齐权重；β: 重建权重
    """
    x, y = batch["x"], batch["y"]
    out = model(x, y, need_teacher_z=True)
    y_hat, z_hat, z_star = out["y_hat"], out["z_hat"], out["z_star"]

    if use_cos:
        latent = 1.0 - F.cosine_similarity(z_hat, z_star, dim=1).mean()
    else:
        latent = F.mse_loss(z_hat, z_star)

    # 可替换为 Charbonnier / 感知损
    recon = F.l1_loss(y_hat, y)

    loss = alpha * latent + beta * recon
    logs = {"latent": float(latent.detach()), "recon": float(recon.detach())}
    return loss, logs

# =========================================================
# 快速自检
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Cs = (64,128,256,512,1024)
    z_channels = 256

    # 1) 教师 AE：先单独训练（这里仅做建图与统计）
    teacher = TeacherAE(
        c_y=3, Cs=Cs,
        up_mode="interp", z_channels=z_channels, groups=8
    ).to(device)

    # 2) 学生编码器
    student = StudentEncoder(
        c_in=3, Cs=Cs,
        z_channels=z_channels, groups=8
    ).to(device)

    # 3) 组合（训练学生时通常冻结教师）
    model = StudentWithTeacher(student, teacher, freeze_teacher=True).to(device)

    # 形状自检
    x = torch.randn(2, 3, 256, 256, device=device)
    y = torch.randn(2, 3, 256, 256, device=device)  # 监督可为真值图像/标签的连续形式
    with torch.no_grad():
        # 教师自编码器重建
        y_rec, z = teacher(y)
        print("[TeacherAE] y_rec:", tuple(y_rec.shape), " z:", tuple(z.shape))
        # 学生+教师解码
        out = model(x, y)
        print("[Student] y_hat:", tuple(out["y_hat"].shape), " z_hat:", tuple(out["z_hat"].shape))

    # 参数量
    print("Params(TeacherAE):", pretty_params(count_params(teacher)))
    print("Params(Student)  :", pretty_params(count_params(student)))
    print("Params(Combined) :", pretty_params(count_params(model)))
