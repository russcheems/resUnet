import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
# 检查是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.set_per_process_memory_fraction(0.9, 0)
# 数据预处理和增强
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(20),     # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 调整亮度和对比度
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])

# 数据加载
train_dir = './data/train'  # 训练集路径
test_dir = './data/test'    # 测试集路径

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 加载 VGG-19 模型
def build_vgg19_model(num_classes):
    # 加载预训练模型
    model = models.vgg19(pretrained=True)
    
    # 冻结预训练层的参数
    for param in model.features.parameters():
        param.requires_grad = False
    
    # 替换分类器部分
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes)  # 输出类别数
    )
    return model

# 初始化模型
num_classes = len(train_dataset.classes)  # 自动获取类别数量
model = build_vgg19_model(num_classes=num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用 tqdm 包装数据加载器，显示进度条
        with tqdm(train_loader, unit="batch") as tepoch:
            torch.cuda.empty_cache()
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 更新进度条上的信息
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 自定义显示内容
                tepoch.set_postfix({
                    "loss": f"{running_loss / total:.4f}",
                    "accuracy": f"{correct / total:.4f}"
                })

        # 每个 epoch 结束后打印结果
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# 测试模型函数（不需要修改）
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f"Test Accuracy: {correct / total:.4f}")

# 运行训练和评估
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
evaluate_model(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'vgg19_emotion_model.pth')
print("Model saved successfully!")
