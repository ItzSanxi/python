import torch
from ultralytics import YOLO
import os
import cv2  # 用于生成检测图片
import numpy as np
import matplotlib.pyplot as plt  # 用于可视化

# 步骤 1: 加载 YOLOv3 模型（预训练权重）
model = YOLO('yolov3.pt')  # 或 'yolov3-tiny.pt' 以加速

# 步骤 2: 可选 - 优化 anchors 为小目标（使用 K-means）
# 假设你有 labels 文件（YOLO 格式: class x_center y_center width height）
def compute_custom_anchors(label_dir, num_anchors=9, num_clusters=3):
    from sklearn.cluster import KMeans
    wh = []  # 收集宽高
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    _, _, _, w, h = map(float, line.strip().split())
                    wh.append([w, h])
    wh = np.array(wh)
    kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(wh)
    anchors = kmeans.cluster_centers_ * [640, 640]  # 假设图像大小 640x640
    print("Custom anchors:", anchors)
    return anchors

# 示例调用（替换为你的 labels 路径）
# custom_anchors = compute_custom_anchors('/path/to/train/labels')
# 在 model.cfg 中手动更新 anchors（或通过 hyperparameters）

# 步骤 3: 训练模型
# 对于红外小目标，建议参数：小 batch_size、更多 epochs、imgsz=640+、数据增强
results = model.train(
    data='/path/to/data.yaml',  # 你的数据集配置文件
    epochs=100,  # 调整为更多以收敛
    imgsz=640,  # 适合小目标
    batch=16,  # 根据 GPU 内存调整
    workers=4,
    device=0 if torch.cuda.is_available() else 'cpu',  # GPU 或 CPU
    name='yolov3_ir_small_target',  # 实验名称
    optimizer='Adam',  # 或 SGD
    lr0=0.001,  # 初始学习率
    augment=True,  # 启用 Mosaic/Mixup 等增强
    iou=0.6,  # IoU 阈值，提高以优化小目标
    conf=0.001  # 低置信度阈值以捕捉小目标
)

# 训练完成后，模型权重保存在 runs/detect/yolov3_ir_small_target/weights/best.pt

# 步骤 4: 推理并生成检测图片
# 加载训练好的模型
trained_model = YOLO('runs/detect/yolov3_ir_small_target/weights/best.pt')

# 测试图像路径（替换为你的红外测试图像）
test_image_path = '/path/to/test/ir_image.jpg'

# 进行检测
results = trained_model(test_image_path)

# 可视化并保存检测结果
for result in results:
    img = cv2.imread(test_image_path)
    boxes = result.boxes.xyxy.cpu().numpy()  # bounding boxes
    confs = result.boxes.conf.cpu().numpy()  # 置信度
    classes = result.boxes.cls.cpu().numpy()  # 类别

    for box, conf, cls in zip(boxes, confs, classes):
        if conf > 0.5:  # 阈值过滤
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿框
            cv2.putText(img, f'Target {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存生成的检测图片
    output_path = 'detected_ir_image.jpg'
    cv2.imwrite(output_path, img)
    print(f"检测图片已保存至: {output_path}")

    # 可选：显示图片
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# 额外提示：对于红外小目标，考虑添加注意力模块或使用 YOLOv8（model = YOLO('yolov8n.pt')）以获得更好性能。
