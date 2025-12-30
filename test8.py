import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import random


# -----------------------------
# 1. 创建一个小型合成数据集（避免下载大型数据集）
# -----------------------------
class SyntheticSegmentationDataset(Dataset):
    def __init__(self, num_samples=100, img_size=(224, 224), num_classes=3):
        """
        创建一个合成语义分割数据集
        num_classes: 3类（0: 背景, 1: 物体A, 2: 物体B）
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 创建一个随机的RGB图像
        img = np.random.randint(0, 255, (self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

        # 创建一个简单的分割掩码
        mask = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint8)

        # 在图像中随机添加一些几何形状作为分割目标
        h, w = self.img_size

        # 添加一个圆形（类别1）
        center_x, center_y = random.randint(50, w - 50), random.randint(50, h - 50)
        radius = random.randint(20, 40)
        y, x = np.ogrid[:h, :w]
        mask_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        mask[mask_circle] = 1

        # 添加一个矩形（类别2）
        x1, y1 = random.randint(20, w - 100), random.randint(20, h - 100)
        x2, y2 = x1 + random.randint(30, 60), y1 + random.randint(30, 60)
        mask[y1:y2, x1:x2] = 2

        # 转换图像和掩码为张量
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(Image.fromarray(img))
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor


# -----------------------------
# 2. 构建简化版的FCN模型
# -----------------------------
class SimpleFCN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleFCN, self).__init__()

        # 编码器部分（简化版）
        self.encoder = nn.Sequential(
            # 输入: 3x224x224
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 输出: 64x112x112

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 输出: 128x56x56

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 输出: 256x28x28
        )

        # 中间层
        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

        # 解码器部分（上采样）
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 上采样到56x56
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 上采样到112x112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 上采样到224x224
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x


# 可视化函数
def visualize_sample(img, mask, idx=0):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 反标准化图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img[idx].numpy().transpose(1, 2, 0)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    axes[0].imshow(img_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(mask[idx].numpy(), cmap='jet')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('sample_data.png', dpi=100, bbox_inches='tight')
    plt.show()


# 主函数
def main():
    # 创建小型数据集
    print("创建合成数据集...")
    dataset = SyntheticSegmentationDataset(num_samples=200, img_size=(224, 224), num_classes=3)

    # 设置num_workers=0以避免多进程问题
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    # 获取一个批次的数据并可视化
    print("可视化样本数据...")
    for img_batch, mask_batch in train_loader:
        visualize_sample(img_batch, mask_batch)
        break

    # 创建模型
    print("构建FCN模型...")
    model = SimpleFCN(num_classes=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)

    # 打印模型结构
    print("模型结构:")
    print(model)

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # 训练模型（少量epoch）
    num_epochs = 5
    train_losses = []

    print("\n开始训练...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (inputs, masks) in enumerate(train_loader):
            inputs = inputs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # 确保输出尺寸与标签尺寸匹配
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=True)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if (batch_idx + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}] 完成, 平均Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

    # 绘制训练损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=100, bbox_inches='tight')
    plt.show()

    # 测试模型
    print("\n测试模型...")
    model.eval()
    with torch.no_grad():
        test_batch, test_masks = next(iter(train_loader))
        test_batch = test_batch.to(device)
        predictions = model(test_batch)

        # 反标准化并可视化结果
        for i in range(min(2, test_batch.size(0))):  # 显示前2个样本
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # 输入图像
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            img = test_batch[i] * std + mean
            img = img.cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)

            axes[0].imshow(img)
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            # 真实掩码
            axes[1].imshow(test_masks[i].cpu().numpy(), cmap='jet')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            # 预测结果
            pred_mask = torch.argmax(predictions[i], dim=0).cpu().numpy()
            axes[2].imshow(pred_mask, cmap='jet')
            axes[2].set_title('Prediction')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(f'result_sample_{i}.png', dpi=100, bbox_inches='tight')
            plt.show()

    # 保存模型为.pth文件
    print("\n保存模型...")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_losses[-1],
        'num_classes': 3,
    }, 'fcn_synthetic.pth')

    print("模型已保存为 fcn_synthetic.pth")

    # 保存为单独权重文件（便于加载）
    torch.save(model.state_dict(), 'fcn_synthetic_weights.pth')
    print("权重文件已保存为 fcn_synthetic_weights.pth")

    # 打印模型大小
    model_size = os.path.getsize('fcn_synthetic.pth') / (1024 * 1024)  # 转换为MB
    print(f"模型文件大小: {model_size:.2f} MB")

    # 测试加载模型
    print("\n测试加载保存的模型...")
    checkpoint = torch.load('fcn_synthetic.pth')
    loaded_model = SimpleFCN(num_classes=3)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.to(device)
    loaded_model.eval()

    # 用加载的模型进行预测
    with torch.no_grad():
        test_output = loaded_model(test_batch[:1])
        print(f"加载的模型预测成功! 输出形状: {test_output.shape}")

    print("\n训练完成!")


if __name__ == '__main__':
    # 在Windows上运行需要添加这个
    import multiprocessing

    multiprocessing.freeze_support()

    main()