import torch
import torch.nn as nn
import torch.nn.functional as F

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
    


class Cifar10FLNet(nn.Module):
    def __init__(self):
        super(Cifar10FLNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(3, 2, 1)
        self.norm1 = nn.LocalResponseNorm(size=4, alpha=0.001/9.0, beta=0.75, k=1.0)
        self.norm2 = nn.LocalResponseNorm(size=4, alpha=0.001/9.0, beta=0.75, k=1.0)
        self.fc1 = nn.Linear(4096, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)
        self.name = 'cifar10flnet'

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool2(x)
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x)


