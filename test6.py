import os
import glob
import itertools
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ==========================
# 1. 定义 DnCNN 模型
# ==========================
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_layers=17, features=64):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(channels, features, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise  # 输出去噪图像


# ==========================
# 2. BSD300 数据集类（自动调整为128x128）
# ==========================
class BSD300Dataset(Dataset):
    def __init__(self, image_dir, ground_dir, grayscale=True, target_size=(128, 128)):
        self.grayscale = grayscale
        self.target_size = target_size

        # 检查路径
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"❌ 训练图像路径不存在: {image_dir}")
        if not os.path.exists(ground_dir):
            raise FileNotFoundError(f"❌ Ground Truth 路径不存在: {ground_dir}")

        # 支持多种格式
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        self.image_files = sorted(itertools.chain.from_iterable(
            glob.glob(os.path.join(image_dir, ext)) for ext in exts))
        self.ground_files = sorted(itertools.chain.from_iterable(
            glob.glob(os.path.join(ground_dir, ext)) for ext in exts))

        # 检查是否有文件
        if len(self.image_files) == 0:
            raise RuntimeError(f"⚠️ 未在 {image_dir} 中找到任何图像文件，请检查路径或扩展名")
        if len(self.ground_files) == 0:
            raise RuntimeError(f"⚠️ 未在 {ground_dir} 中找到任何图像文件，请检查路径或扩展名")

        # 检查数量是否匹配
        if len(self.image_files) != len(self.ground_files):
            raise ValueError(f"⚠️ train_image 与 train_ground 数量不匹配：{len(self.image_files)} vs {len(self.ground_files)}")

        print(f"✅ 加载数据集成功: {len(self.image_files)} 张图像")
        print(f"   来自: {image_dir}")
        print(f"   Ground Truth: {ground_dir}")
        print(f"   所有图像将被调整为: {target_size}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_noisy = Image.open(self.image_files[idx])
        img_clean = Image.open(self.ground_files[idx])

        # 转灰度或RGB
        if self.grayscale:
            img_noisy = img_noisy.convert('L')
            img_clean = img_clean.convert('L')
        else:
            img_noisy = img_noisy.convert('RGB')
            img_clean = img_clean.convert('RGB')

        # ✅ 统一到 128x128 尺寸
        img_noisy = img_noisy.resize(self.target_size, Image.BICUBIC)
        img_clean = img_clean.resize(self.target_size, Image.BICUBIC)

        # 转为张量（0~1）
        noisy = np.array(img_noisy).astype(np.float32) / 255.0
        clean = np.array(img_clean).astype(np.float32) / 255.0

        if self.grayscale:
            noisy = np.expand_dims(noisy, axis=0)
            clean = np.expand_dims(clean, axis=0)
        else:
            noisy = noisy.transpose((2, 0, 1))
            clean = clean.transpose((2, 0, 1))

        return torch.from_numpy(noisy), torch.from_numpy(clean)


# ==========================
# 3. 训练流程
# ==========================
def train_dncnn(train_image_dir, train_ground_dir, save_dir='./checkpoints',
                epochs=30, batch_size=8, lr=1e-3, grayscale=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 使用设备: {device}")

    # 加载数据
    train_dataset = BSD300Dataset(train_image_dir, train_ground_dir, grayscale=grayscale, target_size=(128, 128))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 定义模型、损失函数和优化器
    channels = 1 if grayscale else 3
    model = DnCNN(channels=channels).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)

    print("\n开始训练 DnCNN 模型...\n")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] - 平均Loss: {avg_loss:.6f}")

        # 保存模型
        model_path = os.path.join(save_dir, f'dncnn_128_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"✅ 模型已保存: {model_path}")

    print("\n🎉 训练结束！模型已全部保存。")


# ==========================
# 4. 主程序入口
# ==========================
if __name__ == '__main__':
    train_dncnn(
        train_image_dir=r'E:\Projects\python\BSD300\train_image',   # 修改为你的路径
        train_ground_dir=r'E:\Projects\python\BSD300\train_ground', # 修改为你的路径
        save_dir=r'E:\Projects\python\checkpoints',
        epochs=10,           # 可调
        batch_size=4,        # 可调
        lr=1e-3,
        grayscale=True        # 若要训练彩色图片，改成 False
    )
