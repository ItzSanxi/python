# ======================
# ✅ MNIST 完整实验脚本
# ======================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 OMP 冲突

import matplotlib
matplotlib.use('TkAgg')  # 使用系统弹窗绘图

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================
# 1️⃣ 数据加载与预处理
# ==========================
def get_dataloader(train=True, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader


# ==========================
# 2️⃣ 模型定义
# ==========================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


mnist_net = Net()
optimizer = optim.Adam(mnist_net.parameters(), lr=0.001)
train_loss_list, train_count_list, test_accuracy_list = [], [], []


# ==========================
# 3️⃣ 训练函数
# ==========================
def train(epoch):
    mnist_net.train()
    train_loader = get_dataloader(train=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = mnist_net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]"
                  f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            train_loss_list.append(loss.item())
            train_count_list.append(batch_idx + (epoch - 1) * len(train_loader))


# ==========================
# 4️⃣ 测试函数
# ==========================
def test():
    mnist_net.eval()
    test_loader = get_dataloader(train=False)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            output = mnist_net(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracy_list.append(accuracy)
    print(f"\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")


# ==========================
# 5️⃣ 绘图函数（独立弹窗 3 张）
# ==========================
def plot_results():
    # ---- (1) 训练损失曲线 ----
    plt.figure(figsize=(8, 5))
    plt.plot(train_count_list, train_loss_list, color='blue')
    plt.title('Training Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show(block=True)

    # ---- (2) 测试准确率曲线 ----
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(test_accuracy_list) + 1), test_accuracy_list, marker='o', color='green')
    plt.title('Test Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show(block=True)

    # ---- (3) 错误识别图片 ----
    mnist_net.eval()
    test_loader = get_dataloader(train=False)
    misclassified = []
    with torch.no_grad():
        for data, target in test_loader:
            output = mnist_net(data)
            pred = output.argmax(dim=1, keepdim=True)
            wrong_idx = pred.ne(target.view_as(pred)).squeeze()
            for img, p, t in zip(data[wrong_idx], pred[wrong_idx], target[wrong_idx]):
                misclassified.append((img, p.item(), t.item()))
            if len(misclassified) >= 10:
                break

    plt.figure(figsize=(10, 5))
    for i, (img, pred, true) in enumerate(misclassified[:10]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f'P:{pred}/T:{true}', fontsize=9)
        plt.axis('off')
    plt.suptitle("Misclassified Images")
    plt.tight_layout()
    plt.show(block=True)


# ==========================
# 6️⃣ 主程序
# ==========================
if __name__ == "__main__":
    for epoch in range(1, 6):
        train(epoch)
        test()

    # ✅ 保存模型参数
    os.makedirs("models", exist_ok=True)
    save_path = "models/mnist_model.pth"
    torch.save(mnist_net.state_dict(), save_path)
    print(f"✅ 模型已保存到: {os.path.abspath(save_path)}")

    # ✅ 绘制结果
    plot_results()

    # ✅ 测试加载模型
    loaded_model = Net()
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.eval()
    print("✅ 模型加载成功，可以用于推理。")
