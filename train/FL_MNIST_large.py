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
from mechanisms import RAPPORMechanism
from Compressors import PPGC, QSGD, TernGrad, OneBit
# from QSGD import QSGD
# from PPGC import PPGC  # 导入 PPGC 模块
# from ONEBIT import OneBit
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
NUM_ROUNDS = 100          # 联邦学习轮数
EPOCHS_PER_CLIENT = 1    # 每轮客户端本地训练次数
BATCH_SIZE = 128          # 批大小
LEARNING_RATE = 0.01    # 学习率
epsilon = 0.0            # DP 使用的 epsilon 值
NUM_CLIENTS_PER_NODE = 10  # 每个主机上的客户端数量

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
parser.add_argument('--mechanism', type=str, default='BASELINE', choices=['BASELINE', 'PPGC', 'QSGD', 'ONEBIT', 'RAPPOR', 'TERNGRAD'],
                    help='Choose the aggregation mechanism: "BASELINE", "PPGC", "QSGD", "ONEBIT", "RAPPOR" or "TERNGRAD"')
parser.add_argument('--out_bits', type=int, default=2, help='Number of bits for QSGD or PPGC quantization')
parser.add_argument('--world_size', type=int, default=2, help='Number of processes participating in the job')
parser.add_argument('--rank', type=int, required=True, help='Rank of the current process')
parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend')
parser.add_argument('--dist_url', type=str, default='tcp://<master_ip>:<port>', help='URL used to set up distributed training')
parser.add_argument('--epsilon', type=float, default=0, help='Privacy budget for Differential Privacy')
args = parser.parse_args()
epsilon = args.epsilon

# 初始化进程组
dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

# 创建日志文件夹和日志文件名，并重定向输出
log_dir = "MNISTFLlogs"
os.makedirs(log_dir, exist_ok=True)  # 如果文件夹不存在则创建
log_filename = os.path.join(log_dir, f"MNIST_{args.mechanism}_outbits{args.out_bits}_epsilon{epsilon}_rank{args.rank}_large.log")
sys.stdout = open(log_filename, "w")
print(f"Logging to {log_filename}")

# 打印带时间戳的日志信息
def log_with_time(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# MNIST 数据加载
def load_data():
    data_path = './data'
    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    client_datasets = random_split(train_dataset, [len(train_dataset) // (args.world_size * NUM_CLIENTS_PER_NODE)] * (args.world_size * NUM_CLIENTS_PER_NODE))
    client_datasets = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]

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
def train_client(global_model, rank, world_size, mechanism='BASELINE', out_bits=1):
    client_datasets, test_loader = load_data()

    local_models = []
    for client_idx in range(NUM_CLIENTS_PER_NODE):
        model = create_model()
        model.load_state_dict(global_model.state_dict())  # 使用全局模型的参数作为初始参数
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        if mechanism == 'QSGD':
            qsgd_instance = QSGD(epsilon)
        elif mechanism == 'PPGC':
            ppgc_instance = PPGC(epsilon, out_bits, model)  # 创建 PPGC 实例
        elif mechanism == 'ONEBIT':
            onebit_instance = OneBit(epsilon)
            onebit_instance.initialize_error_feedback(model)
        elif mechanism == 'RAPPOR':
            rappor_instance = RAPPORMechanism(out_bits, epsilon, out_bits)  # 创建 RAPPOR 实例
        elif mechanism == 'TERNGRAD':
            terngrad_instance = TernGrad(epsilon)
        # client_loader = DataLoader(client_datasets[args.rank * NUM_CLIENTS_PER_NODE:(args.rank + 1) * NUM_CLIENTS_PER_NODE], batch_size=BATCH_SIZE, shuffle=True)
        client_loader = client_datasets[args.rank * NUM_CLIENTS_PER_NODE + client_idx]
        
        for epoch in range(EPOCHS_PER_CLIENT):
            for step, (data, target) in enumerate(client_loader):
                log_with_time(f"Client {args.rank * NUM_CLIENTS_PER_NODE + client_idx}, Training step {step + 1}")
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                # 根据机制对梯度进行量化
                if mechanism == 'QSGD':
                    for param in model.module.parameters():
                        if param.grad is not None:
                            quantized_gradient = qsgd_instance.quantize(param, out_bits)
                            param.grad = torch.tensor(quantized_gradient, dtype=param.dtype).to(device)

                elif mechanism == 'PPGC':
                    for name, param in model.module.named_parameters() if hasattr(model, 'module') else model.named_parameters():
                        if param.grad is not None:
                            quantized_gradient = ppgc_instance.map_gradient(param)
                            param.grad = torch.tensor(quantized_gradient, dtype=param.dtype).to(device)

                elif mechanism == 'ONEBIT':
                    for name, param in model.module.named_parameters() if hasattr(model, 'module') else model.named_parameters():
                        if param.grad is not None:
                            quantized_gradient = onebit_instance.apply_1bit_sgd_quantization(name, param)
                            param.grad = torch.tensor(quantized_gradient, dtype=param.dtype).to(device)

                elif mechanism == 'RAPPOR':
                    for param in model.module.parameters():
                        if param.grad is not None:
                            # 将检测过的模型参数进行根据化到 [0, 1] 范围
                            min_grad = param.grad.min().item()
                            max_grad = param.grad.max().item()
                            normalized_grad = (param.grad - min_grad) / (max_grad - min_grad)

                            # 使用 RAPPOR 机制进行批量化
                            perturbed_grad = rappor_instance.privatize(normalized_grad.cpu().numpy())
                            param.grad = torch.tensor(perturbed_grad, dtype=param.grad.dtype).to(device)

                            # 返回到原始范围
                            # perturbed_grad_rescaled = torch.tensor(perturbed_grad, dtype=param.grad.dtype).to(device)
                            # param.grad = perturbed_grad_rescaled * (max_grad - min_grad) + min_grad
                elif mechanism == 'TERNGRAD':
                    for param in model.module.parameters():
                        if param.grad is not None:
                            quantized_gradient = terngrad_instance.compress(param)
                            param.grad = torch.tensor(quantized_gradient, dtype=param.dtype).to(device)

                optimizer.step()
                # 聚合前测试本地模型

        local_accuracy = test_model(model, test_loader)
        log_with_time(f"Local model accuracy of client {args.rank * NUM_CLIENTS_PER_NODE + client_idx} before aggregation: {local_accuracy:.4f}")
        local_models.append(model)

    # 使用一个新的临时模型汇总所有客户端的模型参数（本地聚合时取平均）
    temp_global_model = create_model()
    with torch.no_grad():
        for param_temp in temp_global_model.parameters():
            param_temp.data.zero_()

        for model in local_models:
            for param_temp, param_local in zip(temp_global_model.parameters(), model.parameters()):
                param_temp.data += param_local.data / NUM_CLIENTS_PER_NODE

    return temp_global_model

# 测试模型准确性
def test_model(model, test_loader):
    log_with_time("Testing model accuracy")
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

        client_model = train_client(global_model, args.rank, args.world_size, mechanism=mechanism, out_bits=args.out_bits)

        # 聚合客户端模型参数
        dist.barrier()  # 确保所有节点都完成训练再进行聚合
        aggregate_global_model(global_model.module, client_model.module, mechanism)

        # 聚合后测试模型
        aggregated_accuracy = test_model(global_model, test_loader)
        log_with_time(f"Global model accuracy after aggregation: {aggregated_accuracy:.4f}")

# 聚合客户端模型参数
def aggregate_global_model(global_model, client_model, mechanism):
    log_with_time("Aggregating global model from client models")

    # 遍历每个参数，通过 all_reduce 汇总每个客户端的梯度
    for param_global, param_client in zip(global_model.parameters(), client_model.parameters()):

        param_global.data = param_client.data.clone()
        dist.all_reduce(param_global.data, op=dist.ReduceOp.SUM)
        param_global.data /= args.world_size  # 对参数取平均，形成全局模型

# 运行联邦学习
if __name__ == "__main__":
    federated_learning(args.mechanism)

# 关闭日志文件
sys.stdout.close()
