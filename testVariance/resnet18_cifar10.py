import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt

# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# 定义 ResNet-18 模型
model = models.resnet18(pretrained=False, num_classes=10)
model = model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录每个 step 中的最大和最小 L2 Norm
max_l2_norms = []
min_l2_norms = []
max_abs_grads = []
# 记录训练和测试准确率
train_accuracies = []
test_accuracies = []

# 打印每个层的校验向量信息和种类数据的几个有指标的函数
def analyze_and_sparse_gradients(model, threshold=0.01):
    num_layers = 0
    max_grad_norm = 0
    min_grad_norm = float('inf')
    max_abs_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 记录每个校验向量的层数、校验的数量、及于 L2 模值
            num_layers += 1
            grad_tensor = param.grad
            num_elements = grad_tensor.numel()
            grad_norm = grad_tensor.norm(2).item()
            max_grad_norm = max(max_grad_norm, grad_norm)
            min_grad_norm = min(min_grad_norm, grad_norm)
            max_abs_grad = max(max_abs_grad, torch.max(torch.abs(grad_tensor)).item())
            #print(f"Layer {num_layers}: {name}, Gradient Elements: {num_elements}, L2 Norm: {grad_norm:.6f}")

            # 进行森疏化 - 如果应用阈值清理模值
            #grad_tensor[torch.abs(grad_tensor) < threshold] = 0
            
    max_l2_norms.append(max_grad_norm)
    min_l2_norms.append(min_grad_norm)
    max_abs_grads.append(max_abs_grad)
    # print(f"Total Number of Layers with Gradients: {num_layers}")
    # print(f"Max L2 Norm in this Step: {max_grad_norm:.6f}")
    # print(f"Min L2 Norm in this Step: {min_grad_norm:.6f}")
    # print(f"Max Absolute Gradient in this Step: {max_abs_grad:.6f}\n")

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
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.cuda(), labels.cuda()

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 打印和森疏化每个层的校验向量的 L2 模值
        analyze_and_sparse_gradients(model, threshold=0.01)

        # 更新权重
        optimizer.step()

        running_loss += loss.item()

    # 计算训练和测试准确率
    train_accuracy = calculate_accuracy(model, trainloader)
    test_accuracy = calculate_accuracy(model, testloader)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%\n")

# 绘制最大和最小 L2 Norm 的变化曲线
plt.figure(figsize=(10, 5))
plt.plot(max_l2_norms, label='Max L2 Norm')
plt.plot(min_l2_norms, label='Min L2 Norm')
plt.xlabel('Training Step')
plt.ylabel('L2 Norm')
plt.title('Max and Min L2 Norm per Training Step')
plt.legend()
plt.savefig('l2_norms_plot.png')

# 绘制最大梯度绝对值的变化曲线
plt.figure(figsize=(10, 5))
plt.plot(max_abs_grads, label='Max Absolute Gradient')
plt.xlabel('Training Step')
plt.ylabel('Max Absolute Gradient')
plt.title('Max Absolute Gradient per Training Step')
plt.legend()
plt.savefig('max_abs_grads_plot.png')

# 绘制训练和测试准确率的变化曲线
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy per Epoch')
plt.legend()
plt.savefig('accuracy_plot.png')

print("Finished Training")
