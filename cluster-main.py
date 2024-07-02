import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
from models.ResNet import ResNet

import numpy as np
from copy import deepcopy

import argparse  
#常量
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch = 10
groupNum = 5
clientNumPerGroup = 5
clusterGroup = []
class_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
models = []
attackInd = 0
noiseLevel = 10
alpha = 0.99

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def loadIfExist(model_name, model_dir, model_class):
    model_path = os.path.join(model_dir, f'model_{model_name}.pth') 
    if os.path.exists(model_path):  
        # 如果模型文件存在，则加载它  
        print(f"Loading model from {model_path}")  
        model = torch.load(model_path)  
    else:  
        # 如果模型文件不存在，则创建一个新的ResNet模型并保存它  
        print(f"Creating and saving model to {model_path}")  
        model = model_class()  # 假设我们不使用预训练的权重    
    return model

def loadClusterGroup():
    for i in range(groupNum):
        clusterGroup.append(loadIfExist(i, "./models", ResNet))


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

def noise_attack(gradients, noise_level=0.1):
    with torch.no_grad():
        for grad in gradients:
            grad += noise_level * torch.randn_like(grad)

def evalModel(model, test_loaders):
    model.eval()
    accuracy = []
    with torch.no_grad():
        for test_loader in test_loaders:
            correct = 0
            total = 0
            for x, y in test_loader:
                images, labels = x.to(device), y.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy.append(100 * correct / total)
    print(f"Model Accuracy: {accuracy}")
    return accuracy

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='设置是否允许攻击')  
    parser.add_argument('--allow_attack', action='store_true', help='允许攻击模式')  
    args = parser.parse_args()  
    ALLOW_ATTACK = args.allow_attack
    
    #dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loaders = []
    test_loaders = []
    for group in class_groups:
        test_indices = []
        for i in group:
            test_indices += [idx for idx, label in enumerate(testset.targets) if label == i]
        test_subset = torch.utils.data.Subset(testset, test_indices)
        test_loader = torch.utils.data.DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=2)
        test_loaders.append(test_loader)
    for group in class_groups:
        # 划分数据集
        train_indices = []
        for i in group:
            train_indices += [idx for idx, label in enumerate(trainset.targets) if label == i]
        train_subset = torch.utils.data.Subset(trainset, train_indices)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
        train_loaders.append(train_loader)
    
    loadClusterGroup()
    for ind, train_loader in enumerate(train_loaders):
        print("Start Group %d", ind)
        clusterModel = clusterGroup[ind]
        clusterModel = clusterModel.to(device)
        clientModels = []
        true_attackers = []
        identified_attackers = []
        reputation = np.full(clientNumPerGroup, 0.5)
        # 优化器
        optimizer = optim.SGD(clusterModel.parameters(), lr=0.0001, momentum=0.9)
        
        for round in range(epoch):
            selected_clients = list(range(clientNumPerGroup))
            local_gradients = []
            local_data_sizes = []
            
            for client_idx in selected_clients:
                model = deepcopy(clusterModel).to(device)
                gradients = train(model, train_loader, 1)
                
                if round % clientNumPerGroup == client_idx:
                    noise_attack(gradients, noiseLevel)
                    true_attackers.append(client_idx)
                
                gradients_flat = []
                for grad in gradients:
                    gradients_flat.append(grad.view(-1))
                local_gradients.append(torch.cat(gradients_flat).to(device))
                local_data_sizes.append(len(trainset)/len(class_groups))
            
            local_gradients = torch.stack(local_gradients)
            local_data_sizes = np.array(local_data_sizes)
            total_data_size = np.sum(local_data_sizes)
            
            malicious_clients = []
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
            if ALLOW_ATTACK:
                valid_gradients = [local_gradients[i] * local_data_sizes[i] / total_data_size for i in range(len(local_gradients))]
            else :
                valid_gradients = [local_gradients[i] * local_data_sizes[i] / total_data_size for i in range(len(local_gradients)) if selected_clients[i] not in malicious_clients and reputation[selected_clients[i]] >= 0.5]
            if valid_gradients:
                weighted_gradient = torch.sum(torch.stack(valid_gradients), dim=0)
            
                index = 0
                for param in clusterModel.parameters():
                    param.grad = weighted_gradient[index:index + param.numel()].view(param.size())
                    index += param.numel()
                
                optimizer.step()

            # 在每轮训练后进行测试并打印测试精度
            test_accuracy = evalModel(clusterModel, test_loaders)
            #print(f"Round {round + 1}/{epoch}, Test Accuracy: {test_accuracy:.4f}")
            print(f"true_attackers: {true_attackers[-1]}")
            print(f"malicious_clients: {malicious_clients}")
            print(f"reputation: {reputation}")
        print("Finsh Group %d", ind)
    print("Finsh FL training")
                    




