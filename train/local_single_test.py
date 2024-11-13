import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PPGC import PPGC  # 导入 PPGC 模块

# 参数设置
EPOCHS = 10              # 本地训练总轮数
BATCH_SIZE = 32          # 批大小
LEARNING_RATE = 0.001    # 学习率
epsilon = 1.0            # PPGC 使用的 epsilon 值

# 检测是否有可用的 GPU，如果没有则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CIFAR-10数据集的预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10原始尺寸为32x32，不调整尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 使用参考代码中的均值和标准差
])

# 处理命令行参数
parser = argparse.ArgumentParser(description='Local Training with mechanism selection')
parser.add_argument('--mechanism', type=str, default='baseline', choices=['baseline', 'PPGC', 'QSGD'],
                    help='Choose the quantization mechanism: "baseline", "PPGC" or "QSGD"')
parser.add_argument('--out_bits', type=int, default=2, help='Number of bits for QSGD or PPGC quantization')
args = parser.parse_args()

# 创建日志文件夹和日志文件名，并重定向输出
log_dir = "Codes/train/logs"
os.makedirs(log_dir, exist_ok=True)  # 如果文件夹不存在则创建
log_filename = os.path.join(log_dir, f"{args.mechanism}_outbits{args.out_bits}.log")
sys.stdout = open(log_filename, "w")
print(f"Logging to {log_filename}")

# 打印带时间戳的日志信息
def log_with_time(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# QSGD量化函数
def quantize(x, d):
    """Quantize the tensor x to d levels based on absolute value coefficient-wise."""
    norm = np.sqrt(np.sum(np.square(x)))
    level_float = d * np.abs(x) / norm
    previous_level = np.floor(level_float)
    is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
    new_level = previous_level + is_next_level
    return np.sign(x) * norm * new_level / d

# 检查数据集是否存在并加载数据
def load_data():
    data_path = './data'
    if not os.path.exists(os.path.join(data_path, 'cifar-10-batches-py')):
        log_with_time("Downloading CIFAR-10 dataset...")
        train_set = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    else:
        log_with_time("CIFAR-10 dataset already exists.")
        train_set = datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform)
    
    test_set = datasets.CIFAR10(root=data_path, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    return train_loader, test_loader

# 定义ResNet-18模型
def create_model():
    model = models.resnet18(weights=None)  # 使用 weights=None 替代 pretrained=False
    model.fc = nn.Linear(model.fc.in_features, 10)  # 修改输出层以适应 CIFAR-10 的 10 类
    return model.to(device)  # 将模型加载到设备（CPU或GPU）

# 本地训练
def train(model, train_loader, test_loader, epochs=10, mechanism='baseline', out_bits=2):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    print("create ppgc:")
    # 创建 PPGC 实例
    ppgc_instance = PPGC(epsilon)
    print("finished creating ppgc")

    for epoch in range(epochs):
        # 训练过程
        step = 1
        for data, target in train_loader:
            log_with_time(f"Start training for the {step}th step")
            step = step + 1
            data, target = data.to(device), target.to(device)  # 将数据和目标加载到设备上
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # 如果使用 QSGD 或 PPGC，在梯度更新前对梯度量化
            if mechanism == 'QSGD':
                for param in model.parameters():
                    with torch.no_grad():
                        param_np = param.grad.cpu().numpy()
                        quantized_gradient = quantize(param_np, 2 ** out_bits)
                        param.grad = torch.tensor(quantized_gradient, dtype=param.dtype).to(device)

            elif mechanism == 'PPGC':
                for param in model.parameters():
                    with torch.no_grad():
                        param_np = param.grad.cpu().numpy()
                        quantized_gradient = ppgc_instance.map_gradient(param_np, out_bits)
                        param.grad = torch.tensor(quantized_gradient, dtype=param.dtype).to(device)

            optimizer.step()

        # 每个 epoch 结束后测试模型的准确性
        accuracy = test_model(model, test_loader)
        log_with_time(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

# 测试模型准确性
def test_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 将数据加载到设备
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy

# 运行本地训练并测试模型
train_loader, test_loader = load_data()
model = create_model()
train(model, train_loader, test_loader, epochs=EPOCHS, mechanism=args.mechanism, out_bits=args.out_bits)

# 关闭日志文件
sys.stdout.close()
