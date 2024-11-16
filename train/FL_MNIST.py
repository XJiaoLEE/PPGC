import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PPGC import PPGC  # 导入 PPGC 模块
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import random

print(torch.cuda.is_available())  # 检查 CUDA 是否可用
print(torch.cuda.device_count())  # 检查系统中 GPU 的数量
print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# 参数设置
NUM_ROUNDS = 10          # 联邦学习轮数
EPOCHS_PER_CLIENT = 1    # 每轮客户端本地训练次数
BATCH_SIZE = 32          # 批大小
LEARNING_RATE = 0.001    # 学习率
epsilon = 3.0            # PPGC 使用的 epsilon 值

# 检测是否有可用的 GPU，如果没有则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# MNIST 数据集的预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

# 处理命令行参数
parser = argparse.ArgumentParser(description='Federated Learning with mechanism selection')
parser.add_argument('--mechanism', type=str, default='baseline', choices=['baseline', 'PPGC', 'QSGD'],
                    help='Choose the aggregation mechanism: "baseline", "PPGC" or "QSGD"')
parser.add_argument('--out_bits', type=int, default=2, help='Number of bits for QSGD or PPGC quantization')
parser.add_argument('--world_size', type=int, default=2, help='Number of processes participating in the job')
parser.add_argument('--rank', type=int, required=True, help='Rank of the current process')
parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend')
parser.add_argument('--dist_url', type=str, default='tcp://<master_ip>:<port>', help='URL used to set up distributed training')
args = parser.parse_args()

# 初始化进程组
dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

# 创建日志文件夹和日志文件名，并重定向输出
log_dir = "MNISTFLlogs"
os.makedirs(log_dir, exist_ok=True)  # 如果文件夹不存在则创建
log_filename = os.path.join(log_dir, f"MNIST_{args.mechanism}_outbits{args.out_bits}_epsilon{epsilon}_rank{args.rank}_1114.log")
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

# MNIST 数据加载
def load_data():
    data_path = './data'
    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    client_datasets = random_split(train_dataset, [len(train_dataset) // args.world_size] * args.world_size)
    return client_datasets, DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 定义 ConvNet 模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, padding=0)
        self.fc1 = nn.Linear(32 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = nn.functional.avg_pool2d(x, 1)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = nn.functional.avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

# 创建模型
def create_model():
    model = ConvNet().to(device)
    model = DDP(model, device_ids=[args.rank % torch.cuda.device_count()])
    return model

# 每个客户端上训练模型，并在上传前进行量化
def train_client(rank, world_size, mechanism='baseline', out_bits=1):
    client_datasets, test_loader = load_data()
    model = create_model()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    ppgc_instance = PPGC(epsilon, out_bits)  # 创建 PPGC 实例

    client_loader = DataLoader(client_datasets[rank], batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCHS_PER_CLIENT):
        for step, (data, target) in enumerate(client_loader):
            log_with_time(f"Client {rank}, Training step {step + 1}")
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # 根据机制对梯度进行量化
            if mechanism == 'QSGD':
                for param in model.module.parameters():
                    if param.grad is not None:
                        param_np = param.grad.cpu().numpy()
                        quantized_gradient = quantize(param_np, 2 ** out_bits)
                        param.grad = torch.tensor(quantized_gradient, dtype=param.dtype).to(device)

            elif mechanism == 'PPGC':
                for param in model.module.parameters():
                    if param.grad is not None:
                        param_np = param.grad.cpu().numpy()
                        quantized_gradient = ppgc_instance.map_gradient(param_np, out_bits)
                        param.grad = torch.tensor(quantized_gradient, dtype=param.dtype).to(device)

            optimizer.step()

    # 测试模型准确性
    return model

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
    client_datasets, test_loader = load_data()
    global_model = create_model()

    for round in range(NUM_ROUNDS):
        log_with_time(f"Round {round + 1}/{NUM_ROUNDS} started")

        client_model = train_client(args.rank, args.world_size, mechanism=mechanism, out_bits=args.out_bits)

        # 聚合客户端模型参数
        aggregate_global_model(global_model.module, [client_model.module])
        accuracy = test_model(global_model, test_loader)
        log_with_time(f"End of round {round + 1}, global model accuracy: {accuracy:.4f}")

# 聚合客户端模型参数
def aggregate_global_model(global_model, client_models):
    log_with_time("Aggregating global model from client models")
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client_model.state_dict()[key].float().to(device) for client_model in client_models], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)

# 运行联邦学习
if __name__ == "__main__":
    federated_learning(args.mechanism)

# 关闭日志文件
sys.stdout.close()
