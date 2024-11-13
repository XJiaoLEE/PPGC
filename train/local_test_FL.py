import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PPGC import PPGC  # 导入 PPGC 模块

# 参数设置
NUM_CLIENTS = 10         # 客户端数量
NUM_ROUNDS = 10          # 联邦学习轮数
EPOCHS_PER_CLIENT = 1    # 每轮客户端本地训练次数
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
parser = argparse.ArgumentParser(description='Federated Learning with mechanism selection')
parser.add_argument('--mechanism', type=str, default='baseline', choices=['baseline', 'PPGC', 'QSGD'],
                    help='Choose the aggregation mechanism: "baseline", "PPGC" or "QSGD"')
parser.add_argument('--out_bits', type=int, default=2, help='Number of bits for QSGD or PPGC quantization')
args = parser.parse_args()

# 创建日志文件夹和日志文件名，并重定向输出
log_dir = "Codes/train/FLlogs"
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
        dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    else:
        log_with_time("CIFAR-10 dataset already exists.")
        dataset = datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform)
    
    client_datasets = random_split(dataset, [len(dataset) // NUM_CLIENTS] * NUM_CLIENTS)
    return client_datasets

# 定义客户端的ResNet-18模型
def create_model():
    model = models.resnet18(weights=None)  # 使用 weights=None 替代 pretrained=False
    model.fc = nn.Linear(model.fc.in_features, 10)  # 修改输出层以适应CIFAR-10的10类
    return model.to(device)  # 将模型加载到设备（CPU或GPU）

# 在每个客户端上训练模型，并在上传前进行量化
def train_client(model, dataloader, epochs=1, mechanism='baseline', out_bits=2):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # 创建 PPGC 实例
    ppgc_instance = PPGC(epsilon)

    for epoch in range(epochs):
        for step, (data, target) in enumerate(dataloader):
            log_with_time(f"Training step {step + 1}")
            data, target = data.to(device), target.to(device)  # 将数据和目标加载到设备上
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # 如果使用 QSGD 或 PPGC，在梯度上传之前对梯度量化
            if mechanism == 'QSGD':
                for idx, param in enumerate(model.parameters(), 1):
                    with torch.no_grad():
                        log_with_time(f"QSGD layer quantization for layer {idx}")
                        param_np = param.grad.cpu().numpy()
                        quantized_gradient = quantize(param_np, 2 ** out_bits)
                        param.grad = torch.tensor(quantized_gradient, dtype=param.dtype).to(device)

            elif mechanism == 'PPGC':
                for idx, param in enumerate(model.parameters(), 1):
                    with torch.no_grad():
                        log_with_time(f"PPGC layer quantization for layer {idx}")
                        param_np = param.grad.cpu().numpy()
                        quantized_gradient = ppgc_instance.map_gradient(param_np, out_bits)
                        param.grad = torch.tensor(quantized_gradient, dtype=param.dtype).to(device)

            optimizer.step()

# 聚合客户端模型参数
def aggregate_global_model(global_model, client_models):
    log_with_time("Aggregating global model from client models")
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client_model.state_dict()[key].float().to(device) for client_model in client_models], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)

# 测试模型准确性
def test_model(model, test_loader):
    log_with_time("Testing global model accuracy")
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    log_with_time(f"Model accuracy: {accuracy:.4f}")
    return accuracy

# 联邦学习主循环
def federated_learning(mechanism):
    client_datasets = load_data()
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    global_model = create_model()

    for round in range(NUM_ROUNDS):
        log_with_time(f"Round {round + 1}/{NUM_ROUNDS} started")

        # 创建每个客户端的模型
        client_models = [create_model() for _ in range(NUM_CLIENTS)]
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())  # 初始化为全局模型权重

        # 每个客户端本地训练
        for i, client_dataset in enumerate(client_datasets):
            log_with_time(f"Training client {i + 1}")
            client_loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)
            train_client(client_models[i], client_loader, EPOCHS_PER_CLIENT, mechanism=mechanism, out_bits=args.out_bits)

        # 聚合客户端的模型到全局模型
        aggregate_global_model(global_model, client_models)

        # 测试全局模型
        accuracy = test_model(global_model, test_loader)
        log_with_time(f"End of round {round + 1}, global model accuracy: {accuracy:.4f}")

# 运行联邦学习，传入机制参数
federated_learning(args.mechanism)

# 关闭日志文件
sys.stdout.close()
