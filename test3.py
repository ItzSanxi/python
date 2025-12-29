import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 避免 OpenMP 冲突错误

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import ResNet18_Weights
from matplotlib import font_manager

# ==============================
# 字体设置（解决中文警告）
# ==============================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# Step 1: 数据准备
# ==============================
data_dir = r"E:\Projects\python\dc_2000"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==============================
# Step 2: 模型定义
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False  # 冻结特征提取层
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# ==============================
# Step 3: 损失函数与优化器
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# ==============================
# Step 4: 模型训练
# ==============================
num_epochs = 5
train_losses, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 验证集准确率
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = correct / total

    train_losses.append(running_loss / len(train_loader))
    val_accuracies.append(acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}")

# ==============================
# Step 5: 学习曲线可视化
# ==============================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="训练损失")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("训练损失曲线")
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accuracies, label="验证准确率", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("验证准确率曲线")
plt.legend()
plt.show()

# ==============================
# Step 6: 模型评估与错误可视化
# ==============================
model.eval()
y_true, y_pred, wrong_imgs = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        for img, pred, label in zip(images, preds, labels):
            if pred != label:
                wrong_imgs.append((img.cpu(), pred.cpu().item(), label.cpu().item()))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=train_dataset.classes)
disp.plot(cmap="Blues")
plt.title("混淆矩阵")
plt.show()

# 显示部分错误分类示例
if wrong_imgs:
    plt.figure(figsize=(10,6))
    for i, (img, pred, label) in enumerate(wrong_imgs[:6]):
        img = img.permute(1,2,0).numpy()
        img = np.clip(img * 0.229 + 0.485, 0, 1)  # 去标准化
        plt.subplot(2,3,i+1)
        plt.imshow(img)
        plt.title(f"预测: {train_dataset.classes[pred]}\n实际: {train_dataset.classes[label]}")
        plt.axis('off')
    plt.suptitle("错误分类示例")
    plt.show()

# ==============================
# Step 7: 模型保存与加载
# ==============================
torch.save(model.state_dict(), "cat_dog_model.pth")
print("✅ 模型已保存为 cat_dog_model.pth")

# 加载模型示例
# model.load_state_dict(torch.load("cat_dog_model.pth"))
# model.eval()
