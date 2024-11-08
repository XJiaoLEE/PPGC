import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt

# 数据加载和预处理（增加数据增强）
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 定义 VGG-16 模型并冻结部分层
model = models.vgg16(pretrained=True)  # 使用预训练的参数
for param in model.features.parameters():
    param.requires_grad = False  # 冻结卷积部分参数

model.classifier[6] = nn.Linear(4096, 100)  # 修改最后一层输出为 100 类
model = model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)  # 只对全连接层使用较大的学习率

# 使用学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 记录训练和测试准确率
train_accuracies = []
test_accuracies = []
# 记录每个 step 中的最大和最小 L2 Norm
max_l2_norms = []
min_l2_norms = []
max_abs_grads = []
max_abs_grads_after_sparse = []

# 打印每个层的梯度信息，并进行稀疏化
def analyze_and_sparse_gradients(model, threshold=0.01):
    num_layers = 0
    max_grad_norm = 0
    min_grad_norm = float('inf')
    max_abs_grad = 0
    max_abs_grad_after_sparse = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 记录每个层的梯度 L2 范数
            num_layers += 1
            grad_tensor = param.grad
            grad_norm = grad_tensor.norm(2).item()
            max_grad_norm = max(max_grad_norm, grad_norm)
            min_grad_norm = min(min_grad_norm, grad_norm)
            max_abs_grad = max(max_abs_grad, torch.max(torch.abs(grad_tensor)).item())
            #print(f"Layer {num_layers}: {name}, Gradient L2 Norm: {grad_norm:.6f}")

            # 进行稀疏化
            grad_tensor[torch.abs(grad_tensor) < threshold] = 0
            max_abs_grad_after_sparse = max(max_abs_grad_after_sparse, torch.max(torch.abs(grad_tensor)).item())
            
    max_l2_norms.append(max_grad_norm)
    min_l2_norms.append(min_grad_norm)
    max_abs_grads.append(max_abs_grad)
    max_abs_grads_after_sparse.append(max_abs_grad_after_sparse)
    # print(f"Total Number of Layers with Gradients: {num_layers}")
    # print(f"Max L2 Norm in this Step: {max_grad_norm:.6f}")
    # print(f"Min L2 Norm in this Step: {min_grad_norm:.6f}")
    # print(f"Max Absolute Gradient in this Step: {max_abs_grad:.6f}")
    # print(f"Max Absolute Gradient After Sparsification in this Step: {max_abs_grad_after_sparse:.6f}\n")

# 计算准确率
def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 训练过程
num_epochs = 30
for epoch in range(num_epochs):
    model.train()  # 确保模型处于训练模式
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.cuda(), labels.cuda()

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 打印和稀疏化每个层的梯度 L2 范数
        analyze_and_sparse_gradients(model, threshold=0.01)

        # 更新权重
        optimizer.step()

        running_loss += loss.item()

    # 计算训练和测试准确率
    train_accuracy = calculate_accuracy(model, trainloader)
    test_accuracy = calculate_accuracy(model, testloader)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # 更新学习率
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%\n")

# 绘制最大和最小 L2 Norm 的变化曲线
plt.figure(figsize=(10, 5))
plt.plot(max_l2_norms, label='Max L2 Norm (Sparse)')
plt.plot(min_l2_norms, label='Min L2 Norm (Sparse)')
plt.xlabel('Training Step')
plt.ylabel('L2 Norm')
plt.title('Max and Min L2 Norm per Training Step (Sparse)')
plt.legend()
plt.savefig('sparse_vgg16_l2_norms_plot.png')

# 绘制最大梯度绝对值的变化曲线
plt.figure(figsize=(10, 5))
plt.plot(max_abs_grads, label='Max Absolute Gradient (Before Sparse)')
plt.plot(max_abs_grads_after_sparse, label='Max Absolute Gradient (After Sparse)')
plt.xlabel('Training Step')
plt.ylabel('Max Absolute Gradient')
plt.title('Max Absolute Gradient per Training Step (Sparse)')
plt.legend()
plt.savefig('sparse_vgg16_max_abs_grads_plot.png')

# 绘制训练和测试准确率的变化曲线
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy (Sparse)')
plt.plot(test_accuracies, label='Test Accuracy (Sparse)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy per Epoch (Sparse)')
plt.legend()
plt.savefig('sparse_vgg16_accuracy_plot.png')

print("Finished Training")
