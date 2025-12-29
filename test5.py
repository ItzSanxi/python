import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm  # 显示训练进度条

# ===============================
# 1. 设备设置
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备：", device)

# ===============================
# 2. 数据加载与预处理（64x64）
# ===============================
data_dir = './data/face_data'

transform_train = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'training_data'), transform=transform_train)
test_dataset  = datasets.ImageFolder(os.path.join(data_dir, 'testing_data'), transform=transform_test)

# ===============================
# 仅使用部分数据快速训练（可修改）
# ===============================
train_subset, _ = random_split(train_dataset, [5000, len(train_dataset)-5000])
test_subset, _  = random_split(test_dataset, [1000, len(test_dataset)-1000])

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_subset, batch_size=32, shuffle=False)

print(f"训练样本数量（子集）：{len(train_subset)}")
print(f"测试样本数量（子集）：{len(test_subset)}")
print(f"类别：{train_dataset.classes}")

# ===============================
# 3. 定义CNN模型
# ===============================
class FaceCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(FaceCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 64x64 输入 -> 特征图尺寸为 8x8
        self.classifier = nn.Sequential(
            nn.Linear(128*8*8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = FaceCNN(num_classes=len(train_dataset.classes)).to(device)
print(model)

# ===============================
# 4. 损失函数和优化器
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===============================
# 5. 模型训练
# ===============================
num_epochs = 2
train_losses, test_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', ncols=100)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # 测试集准确率
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    test_accuracies.append(accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] 完成, Loss: {epoch_loss:.4f}, Test Acc: {accuracy:.4f}")

print("✅ 训练完成！")

# ===============================
# 6. 可视化训练曲线
# ===============================
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ===============================
# 7. 保存模型
# ===============================
torch.save(model.state_dict(), 'face_cnn_fast.pth')
print("✅ 模型已保存为 face_cnn_fast.pth")

# ===============================
# 8. 推理函数
# ===============================
def predict_image(image_path, model, transform, class_names):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    print(f"图片 {os.path.basename(image_path)} 预测结果：{class_names[predicted.item()]}")
    return class_names[predicted.item()]

# ===============================
# 9. 示例预测
# ===============================
sample_path = './data/example.jpg'  # 替换为你图片
if os.path.exists(sample_path):
    predict_image(sample_path, model, transform_test, train_dataset.classes)
else:
    print("未找到示例图片，请检查路径。")
