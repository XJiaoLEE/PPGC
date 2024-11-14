import os
import argparse
import logging
from datetime import datetime
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PPGC import PPGC
import numpy as np
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# 参数设置
NUM_ROUNDS = 10
EPOCHS_PER_CLIENT = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
epsilon = 1.0

# 检测是否有可用的 GPU，如果没有则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据集预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 日志配置
log_dir = "Codes/train/DFLlogs"
os.makedirs(log_dir, exist_ok=True)

def setup_logging(mechanism, out_bits):
    log_filename = os.path.join(log_dir, f"{mechanism}_outbits{out_bits}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="w"),
            logging.StreamHandler()  # 额外添加控制台输出
        ]
    )
    logging.info(f"Logging to {log_filename}")
    return log_filename

# 定义日志记录函数
def log_with_time(rank, message):
    logging.info(f"[Rank {rank}] {message}")



# QSGD量化函数
def quantize(x, d):
    """Quantize the tensor x to d levels based on absolute value coefficient-wise."""
    norm = np.sqrt(np.sum(np.square(x)))
    level_float = d * np.abs(x) / norm
    previous_level = np.floor(level_float)
    is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
    new_level = previous_level + is_next_level
    return np.sign(x) * norm * new_level / d


# 创建模型
def create_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# 训练客户端模型
def train_client(rank, model, dataloader, mechanism, out_bits):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    ppgc_instance = PPGC(epsilon)

    for epoch in range(EPOCHS_PER_CLIENT):
        for step, (data, target) in enumerate(dataloader):
            log_with_time(rank, f"Training step {step + 1}")
            data, target = data.cuda(rank), target.cuda(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            if mechanism == "QSGD":
                for param in model.parameters():
                    param_np = param.grad.cpu().numpy()
                    quantized_gradient = quantize(param_np, 2 ** out_bits)
                    param.grad = torch.tensor(quantized_gradient, dtype=param.dtype).cuda(rank)
            elif mechanism == "PPGC":
                for param in model.parameters():
                    param_np = param.grad.cpu().numpy()
                    quantized_gradient = ppgc_instance.map_gradient(param_np, out_bits)
                    param.grad = torch.tensor(quantized_gradient, dtype=param.dtype).cuda(rank)

            optimizer.step()

# 聚合全局模型
def aggregate_global_model(global_model, local_model, world_size):
    for param_global, param_local in zip(global_model.parameters(), local_model.parameters()):
        dist.all_reduce(param_local.data, op=dist.ReduceOp.SUM)
        param_global.data = param_local.data / world_size

# 测试模型
def test_model(rank, model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(rank), target.cuda(rank)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    log_with_time(rank, f"Model accuracy: {accuracy:.4f}")
    return accuracy

# 主任务
def main_task(rank, world_size, master_addr, master_port, mechanism, out_bits):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
#     dist.init_process_group(
#     backend="gloo",
#     init_method=f"tcp://{master_addr}:{master_port}",
#     rank=rank,
#     world_size=world_size
# )

    torch.cuda.set_device(rank)

    # 数据分配
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    data_per_client = len(dataset) // world_size
    client_dataset, _ = random_split(dataset, [data_per_client, len(dataset) - data_per_client])
    dataloader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 模型初始化
    local_model = create_model().cuda(rank)
    global_model = create_model().cuda(rank)

    for round in range(NUM_ROUNDS):
        log_with_time(rank, f"Round {round + 1}/{NUM_ROUNDS} started")
        train_client(rank, local_model, dataloader, mechanism, out_bits)
        aggregate_global_model(global_model, local_model, world_size)

        if rank == 0:  # 仅主节点测试并记录准确性
            accuracy = test_model(rank, global_model, test_loader)
            log_with_time(rank, f"End of round {round + 1}, global model accuracy: {accuracy:.4f}")

    dist.destroy_process_group()

# 主函数
def main():
    parser = argparse.ArgumentParser(description="Federated Learning with Distributed Training")
    parser.add_argument("--mechanism", type=str, default="baseline", choices=["baseline", "PPGC", "QSGD"],
                        help="Choose the aggregation mechanism")
    parser.add_argument("--out_bits", type=int, default=2, help="Number of bits for quantization")
    parser.add_argument("--world_size", type=int, required=True, help="Total number of clients (processes)")
    parser.add_argument("--rank", type=int, required=True, help="Rank of this client")
    parser.add_argument("--master_addr", type=str, required=True, help="Master node IP address")
    parser.add_argument("--master_port", type=str, required=True, help="Master node port")
    args = parser.parse_args()

    log_filename = setup_logging(args.mechanism, args.out_bits)
    logging.info(f"Starting distributed training with world size {args.world_size}, mechanism {args.mechanism}")
    main_task(args.rank, args.world_size, args.master_addr, args.master_port, args.mechanism, args.out_bits)

if __name__ == "__main__":
    main()
