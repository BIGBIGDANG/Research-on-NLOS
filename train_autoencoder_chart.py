import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from model_two_stage import *
from tqdm import tqdm  # 导入 tqdm 用于显示进度条
from pathlib import Path


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 定义一个自定义数据集类，用于从目录中加载图像
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img  # 输入和目标相同（用于自编码器）

# 训练函数
def train_autoencoder(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="训练进度")  # 添加进度条
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"损失值": loss.item()})  # 更新进度条显示的损失值

    return total_loss / len(dataloader)

if __name__ == "__main__":
    # 超参数设置
    batch_size = 4
    learning_rate = 0.0001
    num_epochs = 31
    img_size = 256  # 如果需要，可以调整图片尺寸
    data_dir = "/root/autodl-tmp/project/dataset/charts/train/target"  # 指定图像目录

    # 加载数据集
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 如果需要调整大小可以启用
        transforms.ToTensor(),
    ])
    dataset = CustomDataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) TeacherAE（只用 encode/decode，冻结）
    Cs = (64,128,256,512,1024)
    z_channels = 256
    
    model = TeacherAE(
        c_y=3, Cs=Cs,
        up_mode="interp", z_channels=z_channels, groups=8
    ).to(device)

    checkpoint_path = ""  # 你希望加载的 .pth 文件路径
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"成功加载模型权重：{checkpoint_path}")
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    for epoch in range(num_epochs):
        print(f"开始第 {epoch + 1}/{num_epochs} 轮训练...")
        avg_loss = train_autoencoder(model, dataloader, optimizer, criterion, device)
        print(f"第 {epoch + 1} 轮完成，平均损失值: {avg_loss:.10f}")

        # 每 30 轮保存一次模型权重
        if (epoch + 1) % 5 == 0:
            checkpoint_dir = Path("/root/autodl-tmp/project/Ours_autoencoder/AutoEncode/checkpoint_chart")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)  # 只创建目录
            save_path = checkpoint_dir / f"autoencoder_epoch_{epoch + 1}.pth"  # 再拼文件名
            torch.save(model.state_dict(), str(save_path))
            print(f"权重已保存至 '{save_path}'")

            # 保存一张模型输出的图像用于可视化
            model.eval()
            with torch.no_grad():
                sample_input, _ = next(iter(dataloader))  # 获取一个 batch
                sample_input = sample_input.to(device)
                output, _ = model(sample_input)
                output = output.cpu()

                # 将第一张图像转换为 PIL 格式保存
                output_image = transforms.ToPILImage()(output[0].clamp(0, 1))  # 注意 clamp 避免越界
                output_dir = "/root/autodl-tmp/project/Ours_autoencoder/AutoEncode/output/chart"
                os.makedirs(output_dir, exist_ok=True)
                output_image.save(os.path.join(output_dir, f"epoch_{epoch + 1}.png"))
                print(f"样例输出图像已保存：epoch_{epoch + 1}.png")
