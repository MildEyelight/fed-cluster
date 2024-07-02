import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import copy

# 检查GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 超参数
num_clients = 5
num_rounds = 100
pretrain_rounds = 10
local_epochs = 5
batch_size = 32
alpha = 0.99
noise_level = 10  # 噪声级别

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 分割数据给客户端
indices = np.arange(len(train_dataset))
np.random.shuffle(indices)
client_data_indices = np.array_split(indices[:len(train_dataset)//10], num_clients)
client_datasets = [Subset(train_dataset, indices) for indices in client_data_indices]
client_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class ResNetCIFAR10(nn.Module):
    def __init__(self):
        super(ResNetCIFAR10, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
    
    def forward(self, x):
        return self.resnet(x)

# 初始化客户端模型
client_models = [ResNetCIFAR10().to(device) for _ in range(num_clients)]

# 初始化服务器的全局模型
global_model = ResNetCIFAR10().to(device)

# 信誉值初始化
reputation = np.full(num_clients, 0.5)

# 定义训练函数
def train(model, loader, epochs):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    total_gradients = [torch.zeros_like(param) for param in model.parameters()]
    
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                for i, param in enumerate(model.parameters()):
                    total_gradients[i] += param.grad

    return total_gradients

# 定义测试函数
def test(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(loader.dataset)

# 定义加噪声攻击
def noise_attack(gradients, noise_level=0.1):
    with torch.no_grad():
        for grad in gradients:
            grad += noise_level * torch.randn_like(grad)

# 计算攻击识别率
def calculate_attack_detection_rate(identified_attackers, true_attackers):
    return len(set(identified_attackers) & set(true_attackers)) / len(true_attackers)

# 开始联邦学习过程
true_attackers = []
identified_attackers = []

# 优化器
optimizer = optim.SGD(global_model.parameters(), lr=0.0001, momentum=0.9)

for round in range(num_rounds):
    selected_clients = np.random.choice(num_clients, num_clients, replace=False)
    local_gradients = []
    local_data_sizes = []
    
    for client_idx in selected_clients:
        model = copy.deepcopy(global_model).to(device)
        gradients = train(model, client_loaders[client_idx], local_epochs)
        
        if round > pretrain_rounds:
            if round % num_clients == client_idx:
                noise_attack(gradients, noise_level)
                true_attackers.append(client_idx)
        
        gradients_flat = []
        for grad in gradients:
            gradients_flat.append(grad.view(-1))
        local_gradients.append(torch.cat(gradients_flat).to(device))
        local_data_sizes.append(len(client_datasets[client_idx]))
    
    local_gradients = torch.stack(local_gradients)
    local_data_sizes = np.array(local_data_sizes)
    total_data_size = np.sum(local_data_sizes)
    
    malicious_clients = []
    if round > pretrain_rounds:
        # 计算标准差
        std_devs = torch.std(local_gradients, dim=0)
        
        # 识别恶意客户端
        threshold = torch.mean(std_devs) + 3 * torch.std(std_devs)
        for i, gradient in enumerate(local_gradients):
            if torch.any(torch.abs(gradient - torch.mean(local_gradients, dim=0)) > threshold):
                malicious_clients.append(selected_clients[i])
                reputation[selected_clients[i]] = alpha * reputation[selected_clients[i]]
            else:
                reputation[selected_clients[i]] = alpha * reputation[selected_clients[i]] + (1 - alpha)
        
        identified_attackers.extend(malicious_clients)
    
    # 聚合非恶意客户端的梯度
    valid_gradients = [local_gradients[i] * local_data_sizes[i] / total_data_size for i in range(len(local_gradients)) if selected_clients[i] not in malicious_clients and reputation[selected_clients[i]] >= 0.5]
    if valid_gradients:
        weighted_gradient = torch.sum(torch.stack(valid_gradients), dim=0)
    
        index = 0
        for param in global_model.parameters():
            param.grad = weighted_gradient[index:index + param.numel()].view(param.size())
            index += param.numel()
        
        optimizer.step()

    # 在每轮训练后进行测试并打印测试精度
    test_accuracy = test(global_model, test_loader)
    print(f"Round {round + 1}/{num_rounds}, Test Accuracy: {test_accuracy:.4f}")
    if round > pretrain_rounds:
        print(f"true_attackers: {true_attackers[-1]}")
        print(f"malicious_clients: {malicious_clients}")
    print(f"reputation: {reputation}")

#TODO 攻击识别率的计算还有点问题
# attack_detection_rate = calculate_attack_detection_rate(identified_attackers, true_attackers)
# print(f"攻击识别率: {attack_detection_rate:.2f}")
