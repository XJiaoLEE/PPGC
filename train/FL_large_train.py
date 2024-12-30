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
from load_model import Cifar10FLNet, ConvNet
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
NUM_ROUNDS = 150          # 联邦学习轮数
EPOCHS_PER_CLIENT = 5    # 每轮客户端本地训练次数 4 150
BATCH_SIZE = 150 #150          # 批大小32 300 FOR MNIST 200 FOR CIFAR100 125 FOR CIFAR10
LEARNING_RATE = 0.01    # 学习率
epsilon = 0.0            # DP 使用的 epsilon 值
NUM_CLIENTS_PER_NODE = 25  # 每个主机上的客户端数量125 
accumulated_gradients=None

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
if epsilon > 0 :
    if mechanism == 'BASELINE' :
        mechanism = 'LDP-FL'
if args.dataset != 'MNIST':
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 125  #125
    EPOCHS_PER_CLIENT = 10#500 //2
    NUM_CLIENTS_PER_NODE = 5
    NUM_ROUNDS = 3000
# if args.dataset != 'MNIST':
#     LEARNING_RATE = 0.001
#     BATCH_SIZE = 125  #125
#     EPOCHS_PER_CLIENT = 100#500 //2
#     NUM_CLIENTS_PER_NODE = 1
#     NUM_ROUNDS = 3000

# import wandb
# wandb.init(project="federated-learning", config={
#     "learning_rate": LEARNING_RATE,
#     "num_rounds": NUM_ROUNDS,
#     "num_clients_per_node": NUM_CLIENTS_PER_NODE,
#     "epochs_per_client": EPOCHS_PER_CLIENT,
#     "batch_size": BATCH_SIZE,
#     "mechanism": mechanism,
#     "epsilon": epsilon,
#     "sparsification": sparsification_ratio,
#     "dataset": args.dataset
# })

# 初始化进程组
dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

# Create log directory and log filename, and redirect output
log_dir = "FLlogs_afsub"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"{args.dataset}_{mechanism}_outbits{args.out_bits}_epsilon{epsilon}_sparsification{args.sparsification}_large.log")
sys.stdout = open(log_filename, "w")
print(f"Logging to {log_filename}")
pruning_mask = {}
# 打印带时间戳的日志信息
def log_with_time(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# non-iid loaddata
# 将数据加载放在全局，只调用一次
train_dataset, test_dataset = None, None

def load_data():
    global train_dataset, test_dataset
    if train_dataset is None or test_dataset is None:
        data_path = './data'
        if args.dataset == 'CIFAR100':
            # LEARNING_RATE = 0.001
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并调整到64x64大小
                transforms.RandomHorizontalFlip(),                    # 随机水平翻转
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差
            ])
            transform_test = transforms.Compose([
                transforms.Resize((224, 224)),  # 将测试集图像调整为64x64大小
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
        elif args.dataset == 'CIFAR10':
            # IMAGENET_MEAN = [0.485, 0.456, 0.406]
            # IMAGENET_STD = [0.229, 0.224, 0.225]

            # # 数据预处理
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),  # 随机裁剪和填充，以增强数据
            #     transforms.RandomHorizontalFlip(),     # 随机水平翻转
            #     transforms.Resize(224),                # 调整图片大小为224x224，以适应ResNet输入要求
            #     transforms.ToTensor(),                 # 转换为tensor
            #     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # 使用ImageNet的均值和标准差
            # ])
            
            # transform_test = transforms.Compose([
            #     transforms.Resize(224),                # 调整图片大小为224x224
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # 使用ImageNet的均值和标准差
            # ])
            # CIFAR10的均值和标准差
            CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
            CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

            # 训练集数据预处理
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  # 随机裁剪并填充至32x32
                transforms.RandomHorizontalFlip(),     # 随机水平翻转
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)  # 使用CIFAR10的均值和标准差
            ])

            # 测试集数据预处理
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)  # 使用CIFAR10的均值和标准差
            ])

            # 加载训练集和测试集
            train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        else:  # MNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
            ])
            train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

        client_datasets = random_split(train_dataset, [len(train_dataset) // (args.world_size * NUM_CLIENTS_PER_NODE)] * (args.world_size * NUM_CLIENTS_PER_NODE))
        client_datasets = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) for ds in client_datasets]
        # Number of clients
        # num_clients = args.world_size * NUM_CLIENTS_PER_NODE
        # num_classes = len(train_dataset.classes)

        # # Use Dirichlet distribution to assign data to clients
        # alpha = 5  # Controls the degree of non-IID 0.5
        # client_loaders = []
        # idxs_per_class = {i: np.where(np.array(train_dataset.targets) == i)[0] for i in range(num_classes)}
        # client_idxs = [[] for _ in range(num_clients)]
    
        # for c, idxs in idxs_per_class.items():
        #     np.random.shuffle(idxs)
        #     proportions = np.random.dirichlet([alpha] * num_clients)
        #     proportions = (proportions * len(idxs)).astype(int)
            
        #     for i in range(num_clients):
        #         client_idxs[i].extend(idxs[sum(proportions[:i]):sum(proportions[:i+1])])
        
        # for client_data in client_idxs:
        #     client_subset = torch.utils.data.Subset(train_dataset, client_data)
        #     client_loaders.append(DataLoader(client_subset, batch_size=BATCH_SIZE, shuffle=True))

        # return client_loaders, DataLoader(test_dataset, batch_size=BATCH_SIZE)
        return client_datasets, DataLoader(test_dataset, batch_size=BATCH_SIZE)


# Create model based on dataset selection
def create_model():
    if args.dataset == 'CIFAR100':
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
        model.fc = nn.Linear(model.fc.in_features, 100).to(device)
    elif args.dataset == 'CIFAR10':
        model = models.resnet18(num_classes=10).to(device)
        # from torchvision.models import ResNet18_Weights
        # model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        # model.fc = nn.Linear(model.fc.in_features, 10).to(device)
    else:  # MNIST
        model = ConvNet().to(device)
    model = DDP(model, device_ids=[args.rank % torch.cuda.device_count()])
    # generate_global_mask(model, pruning_ratio=0.0)
    return model

# 客户端模型训练
class GradientCompressor:
    def __init__(self, mechanism, sparsification_ratio, epsilon, out_bits):
        self.mechanism = mechanism
        self.epsilon = epsilon
        self.out_bits = out_bits
        self.compression_ratio = sparsification_ratio
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
    
    def _sparsify(self, tensor):
        # Flatten the tensor
        tensor = tensor.view(-1)
        numel = tensor.numel()
        num_selects = max(1, int(numel * self.compression_ratio))

        # Get the importance (absolute values)
        importance = tensor.abs()

        # Find the top k% elements directly using topk
        threshold, indices = torch.topk(importance, num_selects, sorted=False)

        # Extract the values at these indices
        values = tensor[indices]
        if self.compressor_instance is not None:
            device_info = values.device
            values = values.cpu().numpy()
            values = self.compressor_instance.compress(values)
            values = torch.from_numpy(values).to(device_info)
        return values, indices, numel

    def compress(self, tensor):
        # Perform sparsification
        values, indices, numel = self._sparsify(tensor)
        
        # Return the compressed representation and the original tensor size
        return (values, indices, numel)

    def decompress(self, compressed_tensor, original_size):
        values, indices, numel = compressed_tensor
        # Create an empty tensor of the original size
        decompressed_tensor = torch.zeros(original_size, device=values.device)
        # Put the values back to their original positions
        decompressed_tensor.index_put_([indices], values)
        # Reshape back to the original shape
        return decompressed_tensor.view(original_size)

    
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
    
def sparsify_comm_hook(state, bucket):
    global accumulated_gradients
    # global global_model
    # print("sparsify_comm_hook")
    tensor = bucket.buffer()

    # Create a compressor and compress the tensor
    compressor =  state['gradient_compressor']
    values, indices, numel = compressor.compress(tensor)


    def gather():
        # Gather values and indices from all processes
        gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
        gathered_indices = [torch.zeros_like(indices) for _ in range(dist.get_world_size())]

        # All gather values and indices
        dist.all_gather(gathered_values, values)
        dist.all_gather(gathered_indices, indices)

        # Combine gathered results
        combined_values = torch.cat(gathered_values)
        combined_indices = torch.cat(gathered_indices)
        return combined_values, combined_indices

    combined_values, combined_indices = gather()

    # Decompress the tensor with combined values and indices
    decompressed_tensor = compressor.decompress((combined_values, combined_indices, numel), tensor.size())
    print("decompressed_tensor",decompressed_tensor.shape) 

    # Return the decompressed tensor divided by world size
    fut = torch.futures.Future() 
    # fut.set_result(decompressed_tensor / dist.get_world_size())
    decompressed_tensor = decompressed_tensor / dist.get_world_size()
    # 如果 accumulated_gradients 为空，则初始化为与 tensor 相同形状的零张量
    if accumulated_gradients is None:
        accumulated_gradients = torch.zeros_like(tensor)
        print("accumulated_gradients",accumulated_gradients.shape)
    print("tensor",decompressed_tensor)
    for namename in decompressed_tensor[0]:
        print("namename",namename)
        accumulated_gradients[0][namename] += decompressed_tensor[0][namename] 
    # 将 decompressed_tensor 的每个部分累加到 accumulated_gradients 的正确位置
    # offset = 0
    # for param in tensor:
    #     num_elements = param.numel()  # 当前部分的梯度元素数量
    #     reshaped_grad = decompressed_tensor[offset:offset + num_elements].view(param.size())
    #     accumulated_gradients[offset:offset + num_elements].add_(reshaped_grad)
    #     offset += num_elements

                    
    # if accumulated_gradients is None:
    #     accumulated_gradients = decompressed_tensor.clone()

    # else:
    #     accumulated_gradients.copy_(decompressed_tensor)
    fut.set_result(tensor)
    return fut

import torch.nn.utils.prune as prune

def generate_global_mask(model, pruning_ratio=0.1):
    """
    Generate a global pruning mask based on the global model's weights.
    Args:
        model (nn.Module): The global model.
        pruning_ratio (float): The fraction of weights to prune (0.3 means 30% pruning).
    Returns:
        dict: A dictionary where each layer's name has the corresponding mask.
    """
    pruning_mask = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Generate a pruning mask for each layer
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            pruning_mask[name] = module.weight_mask.clone()  # Store the mask
            prune.remove(module, 'weight')  # Remove the pruning method from the module
    return pruning_mask

def apply_global_mask(model, pruning_mask):
    """
    Apply the global pruning mask to a model.
    Args:
        model (nn.Module): The model to apply the mask.
        pruning_mask (dict): The global pruning mask.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if name in pruning_mask:
                # Apply the global pruning mask to the model's weights
                module.weight.data *= pruning_mask[name]
                module.weight.requires_grad = False  # Optional: Freeze the pruned weights


from torch.optim.lr_scheduler import StepLR
# Create client models only once
client_models = [create_model() for _ in range(NUM_CLIENTS_PER_NODE)]
optimizers = [optim.Adam(model.parameters(), lr=LEARNING_RATE) for model in client_models]
schedulers = [StepLR(optimizer, step_size=70, gamma=0.8) for optimizer in optimizers]
# optimizers = [torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) for model in client_models]
gradient_compressor = GradientCompressor(mechanism, sparsification_ratio, epsilon, args.out_bits)
state = {'gradient_compressor': gradient_compressor}
for model in client_models:
    model.train()
    # model.re
    # model.register_comm_hook(state, sparsify_comm_hook)
global_model = create_model()
# 使用 Adam 作为优化器，设置合适的学习率
global_optimizer = optim.Adam(global_model.parameters(), lr=LEARNING_RATE*5)
global_scheduler = StepLR(global_optimizer, step_size=7, gamma=0.8)
# global_optimizer = optim.Adam(global_model.parameters(), lr=LEARNING_RATE)  # 你可以根据需要调整lr
# global_optimizer = optim.SGD(global_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
gradient_compressor = GradientCompressor(mechanism, sparsification_ratio, epsilon, args.out_bits)
state = {'gradient_compressor': gradient_compressor}
# global_model.register_comm_hook(state, sparsify_comm_hook)
global_model.train()
client_datasets, test_loader = load_data()

# Train client function
def train_epoch(global_model, global_optimizer, client_datasets, test_loader, mechanism='BASELINE', out_bits=1):
    log_with_time(f"Start training ....")
    # global_model.train()
    # Randomly select 50% of local clients
    total_local_clients = NUM_CLIENTS_PER_NODE 
    global accumulated_gradients   
    # Train the model for one epoch
    for round in range(NUM_ROUNDS):
        
        log_with_time(f"Round {round + 1}/{NUM_ROUNDS} started")
        selected_clients = random.sample(range(total_local_clients), total_local_clients // 1)  # Randomly select half of the clients
        print("selected_clients",selected_clients)
        accumulated_gradients=None
        global_model.train()
        for client_idx in selected_clients:
            log_with_time(f"Training client {client_idx}")
            model = client_models[client_idx]
            optimizer = optimizers[client_idx]
            scheduler = schedulers[client_idx]
            criterion = nn.CrossEntropyLoss()
            client_loader = client_datasets[args.rank * NUM_CLIENTS_PER_NODE + client_idx]
    
            model.load_state_dict(global_model.state_dict())
            for epoch in range(EPOCHS_PER_CLIENT):
                log_with_time(f"Epoch {epoch + 1}")
            
                for step, (data, target) in enumerate(client_loader):
                    model.train()
                    # log_with_time(f"Client {args.rank * NUM_CLIENTS_PER_NODE + client_idx}, Training step {step + 1}")
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    if accumulated_gradients is None:
                        accumulated_gradients = {name: torch.zeros_like(param.grad) for name, param in model.named_parameters() if param.requires_grad}
                    
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            accumulated_gradients[name] += param.grad 
                # for name, grad in accumulated_gradients.items():
                #     print(f"Shape of gradient for {name}: {grad.shape}")
                # aggregated_accuracy = test_model(model, test_loader)
                # log_with_time(f"Model accuracy at client {client_idx} : {aggregated_accuracy:.4f}")
                scheduler.step()
            
            aggregated_accuracy = test_model(model, test_loader)
            log_with_time(f"Model accuracy at client {client_idx} : {aggregated_accuracy:.4f}")
            print("optimizer.__getattribute__('param_groups')[0]['lr']",optimizer.__getattribute__('param_groups')[0]['lr'])
        global_optimizer.zero_grad()
        dist.barrier()
        for name, param in model.named_parameters():
            if param.requires_grad:    
                dist.all_reduce(accumulated_gradients[name], op=dist.ReduceOp.SUM) 
                # if name == "module.layer1.0.conv2.weight":
                #     print("name before aggregation",name,accumulated_gradients[name][0][0]) 
        for name, param in global_model.named_parameters():
            if param.requires_grad: 
                param.grad=accumulated_gradients[name].to(param.device) / (len(selected_clients)*len(client_loader)*EPOCHS_PER_CLIENT*args.world_size)
                # param.data -= global_optimizer.__getattribute__('param_groups')[0]['lr'] * param.grad
                # if name == "module.layer1.0.conv2.weight":    
                #     print("global gradients after aggregation",param.grad[0][0])   

        
        global_optimizer.step()
        global_scheduler.step()

        aggregated_accuracy = test_model(global_model, test_loader)
        log_with_time(f"Global model accuracy after aggregation: {aggregated_accuracy:.4f}")
        print("global_optimizer.__getattribute__('param_groups')[0]['lr']",global_optimizer.__getattribute__('param_groups')[0]['lr'])

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

train_epoch(global_model, global_optimizer, client_datasets, test_loader, mechanism, args.out_bits)

# # 运行联邦学习
# if __name__ == "__main__":
#     train_client(args.mechanism)

# 关闭日志文件
sys.stdout.close()


