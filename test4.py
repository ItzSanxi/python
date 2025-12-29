# ===============================
# 实验题目：多类别图像分类（CIFAR-10）
# 实验目标：
# 1. 理解图像分类任务
# 2. 学会使用PyTorch搭建深度学习模型
# 3. 掌握迁移学习原理与应用
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
import os


def main():
    # ===============================
    # 1. 环境配置
    # ===============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备：", device)

    # ===============================
    # 2. 数据加载与预处理
    # ===============================
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=0)  # Windows系统设为0

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=0)  # Windows系统设为0

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # ===============================
    # 3. 构建模型（迁移学习：ResNet18）
    # ===============================
    from torchvision import models

    # 更新：使用新的API加载预训练模型
    try:
        # PyTorch 1.13+ 新API
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        print("使用新API加载预训练模型")
    except:
        # 兼容旧版本
        model = models.resnet18(pretrained=True)
        print("使用旧API加载预训练模型")

    # 冻结前面的卷积层，只训练全连接层
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # 修改最后一层输出为10类
    model = model.to(device)

    # ===============================
    # 4. 定义损失函数与优化器
    # ===============================
    criterion = nn.CrossEntropyLoss()
    # 只优化最后一层参数
    optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)

    # ===============================
    # 5. 训练模型
    # ===============================
    writer = SummaryWriter('runs/cifar10_experiment')

    def train_model(model, criterion, optimizer, num_epochs=3):
        since = time.time()

        train_losses, test_accs = [], []

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # 计算训练准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(trainloader)
            train_acc = 100 * correct / total
            train_losses.append(epoch_loss)

            # 验证
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            test_accs.append(acc)

            print(
                f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.3f} Train Acc: {train_acc:.2f}% Test Acc: {acc:.2f}%')
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', acc, epoch)

        time_elapsed = time.time() - since
        print(f'训练完成，耗时 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
        return train_losses, test_accs

    train_losses, test_accs = train_model(model, criterion, optimizer, num_epochs=10)

    # ===============================
    # 6. 模型评估
    # ===============================
    model.eval()
    correct, total = 0, 0
    class_correct = [0.] * 10
    class_total = [0.] * 10

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("\n=== 模型评估结果 ===")
    print(f"总体准确率: {100 * correct / total:.2f}%")
    print("\n分类准确率：")
    for i in range(10):
        if class_total[i] > 0:
            print(f'{classes[i]:5s} 准确率: {100 * class_correct[i] / class_total[i]:.2f}%')

    # ===============================
    # 7. 可视化结果
    # ===============================
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(test_accs, label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ===============================
    # 8. 模型保存与加载
    # ===============================
    torch.save(model.state_dict(), 'cifar10_resnet18.pth')
    print("模型已保存为 cifar10_resnet18.pth")

    # 显示模型结构
    print("\n=== 模型结构 ===")
    print(model)

    writer.close()


if __name__ == '__main__':
    main()