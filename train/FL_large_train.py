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
from mechanisms import RAPPORMechanism, Laplace
from Compressors import PPGC, QSGD, TernGrad, OneBit,TopK
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
NUM_ROUNDS = 120          # 联邦学习轮数
EPOCHS_PER_CLIENT = 1    # 每轮客户端本地训练次数
BATCH_SIZE = 32          # 批大小
LEARNING_RATE = 0.01    # 学习率
epsilon = 0.0            # DP 使用的 epsilon 值
NUM_CLIENTS_PER_NODE = 125  # 每个主机上的客户端数量

# 检测是否有可用的 GPU，如果没有则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
parser.add_argument('--sparsification', type=float, default=0, help='Sparsification ratio for TopK')
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'CIFAR100'],
                    help='Choose the dataset: "MNIST", "CIFAR10", or "CIFAR100"')
args = parser.parse_args()
epsilon = args.epsilon
sparsification_ratio = args.sparsification
mechanism = args.mechanism
if sparsification_ratio > 0:
    if mechanism == 'BASELINE' :
        mechanism = 'TopK'
# 初始化进程组
dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

# Create log directory and log filename, and redirect output
log_dir = "FLlogs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"{args.dataset}_{mechanism}_outbits{args.out_bits}_epsilon{epsilon}_sparsification{args.sparsification}_large.log")
sys.stdout = open(log_filename, "w")
print(f"Logging to {log_filename}")

# 打印带时间戳的日志信息
def log_with_time(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# non-iid loaddata
def load_data():
    data_path = './data'
    if args.dataset == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 mean and std
        ])
        train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
    elif args.dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean and std
        ])
        train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    else:  # MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    # Number of clients
    num_clients = args.world_size * NUM_CLIENTS_PER_NODE
    num_classes = len(train_dataset.classes)

    # Use Dirichlet distribution to assign data to clients
    alpha = 0.5  # Controls the degree of non-IID
    client_loaders = []
    idxs_per_class = {i: np.where(np.array(train_dataset.targets) == i)[0] for i in range(num_classes)}
    client_idxs = [[] for _ in range(num_clients)]
    
    for c, idxs in idxs_per_class.items():
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(idxs)).astype(int)
        
        for i in range(num_clients):
            client_idxs[i].extend(idxs[sum(proportions[:i]):sum(proportions[:i+1])])
    
    for client_data in client_idxs:
        client_subset = torch.utils.data.Subset(train_dataset, client_data)
        client_loaders.append(DataLoader(client_subset, batch_size=BATCH_SIZE, shuffle=True))

    return client_loaders, DataLoader(test_dataset, batch_size=BATCH_SIZE)


# Data loading based on dataset selection
# def load_data():
#     data_path = './data'
#     if args.dataset == 'CIFAR100':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 mean and std
#         ])
#         train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
#         test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
#     elif args.dataset == 'CIFAR10':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean and std
#         ])
#         train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
#         test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
#     else:  # MNIST
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
#         ])
#         train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
#         test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

#     client_datasets = random_split(train_dataset, [len(train_dataset) // (args.world_size * NUM_CLIENTS_PER_NODE)] * (args.world_size * NUM_CLIENTS_PER_NODE))
#     client_datasets = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]

#     return client_datasets, DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Define ConvNet model for MNIST
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

# Create model based on dataset selection
def create_model():
    if args.dataset == 'CIFAR100':
        model = models.resnet50(num_classes=100).to(device)
    elif args.dataset == 'CIFAR10':
        model = models.resnet18(num_classes=10).to(device)
    else:  # MNIST
        model = ConvNet().to(device)
    model = DDP(model, device_ids=[args.rank % torch.cuda.device_count()])
    return model

# 客户端模型训练
class GradientCompressor:
    def __init__(self, mechanism, sparsification_ratio, epsilon, out_bits):
        self.mechanism = mechanism
        self.epsilon = epsilon
        self.out_bits = out_bits
        self.topk_instance = TopK(sparsification_ratio) if sparsification_ratio > 0 else None
        self.compressor_instance = None
        if self.mechanism == 'BASELINE':
            if epsilon > 0:
                self.compressor_instance = Laplace(epsilon)
        elif mechanism == 'QSGD':
            self.compressor_instance = QSGD(epsilon, out_bits)
        elif mechanism == 'PPGC':
            self.compressor_instance = PPGC(epsilon, out_bits)
        elif mechanism == 'ONEBIT':
            self.compressor_instance = OneBit(epsilon)
        elif mechanism == 'RAPPOR':
            self.compressor_instance = RAPPORMechanism(out_bits, epsilon, out_bits)
        elif mechanism == 'TERNGRAD':
            self.compressor_instance = TernGrad(epsilon)

    def gradient_hook(self, grad):
        grad_np1 = grad.cpu().numpy()
        
        shape = grad_np1.shape
        grad_np = grad_np1.flatten()
        # Step 1: Apply sparsification if TopK is enabled
        if self.topk_instance is not None:
            values, indices = self.topk_instance.sparsify(grad_np)
        else:
            values = grad_np
            indices = None

        # Step 2: Compress non-zero values only
        if self.compressor_instance is not None:
            values = self.compressor_instance.compress(values)
        
        # Step 3: Reconstruct the sparse gradient (with compression applied)
        if indices is not None:
            grad_np = np.zeros_like(grad_np)
            grad_np[indices] = values
        else:
            grad_np = values

        grad_np = grad_np.reshape(shape)

        return torch.tensor(grad_np, dtype=grad.dtype, device=grad.device)


# 每个客户端上训练模型，并在上传前进行量化
def train_client(global_model, rank, world_size, mechanism='BASELINE', out_bits=1):
    client_datasets, test_loader = load_data()

    local_models = []
    gradient_compressor = GradientCompressor(mechanism, sparsification_ratio, epsilon, out_bits)

    # 随机选择 50% 的本地客户端
    total_local_clients = NUM_CLIENTS_PER_NODE  
    selected_clients = random.sample(range(total_local_clients), total_local_clients // 2)  # 随机选取一半客户端

    for client_idx in range(NUM_CLIENTS_PER_NODE):
        if client_idx not in selected_clients:
            continue  # 如果当前客户端未被选中，跳过训练
        model = create_model()
        model.load_state_dict(global_model.state_dict())  # 使用全局模型的参数作为初始参数
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        client_loader = client_datasets[args.rank * NUM_CLIENTS_PER_NODE + client_idx]
        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(gradient_compressor.gradient_hook)
        for epoch in range(EPOCHS_PER_CLIENT):
            for step, (data, target) in enumerate(client_loader):
                # log_with_time(f"Client {args.rank * NUM_CLIENTS_PER_NODE + client_idx}, Training step {step + 1}")
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # local_accuracy = test_model(model, test_loader)
        # log_with_time(f"Local model accuracy of client {args.rank * NUM_CLIENTS_PER_NODE + client_idx} before aggregation: {local_accuracy:.4f}")
        local_models.append(model)

    # 使用一个新的临时模型汇总所有客户端的模型参数（本地聚合时取平均）
    temp_global_model = create_model()
    with torch.no_grad():
        for param_temp in temp_global_model.parameters():
            param_temp.data.zero_()

        for model in local_models:
            for param_temp, param_local in zip(temp_global_model.parameters(), model.parameters()):
                param_temp.data += param_local.data / len(selected_clients)

    return temp_global_model

# 测试模型准确性
def test_model(model, test_loader):
    # log_with_time("Testing model accuracy")
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    # log_with_time(f"Model accuracy: {accuracy:.4f}")
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
