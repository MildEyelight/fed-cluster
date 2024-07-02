import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
from models.ResNet import ResNet
#宏定义
#模型保存路径
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_dir = './models'
model_save_prefix = "model_"

#保存加载模型的函数
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

def saveModel(model, model_name, model_dir):
    model_path = os.path.join(model_dir, f'model_{model_name}.pth')
    torch.save(model, model_path)
    
# 定义每两个类别一组的划分
class_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
models = []
# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集,并划分数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loaders = []
for group in class_groups:
    test_indices = []
    for i in group:
        test_indices += [idx for idx, label in enumerate(testset.targets) if label == i]
    test_subset = torch.utils.data.Subset(testset, test_indices)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=2)
    test_loaders.append(test_loader)

# 在每个模型上进行测试并输出结果
def evalModel(model):
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


# 训练和测试函数
def train(model, trainloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in trainloader:
            inputs, labels = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {running_loss / len(trainloader)}")
        evalModel(model)

# 划分数据集并训练模型
for ind, group in enumerate(class_groups):
    # 划分数据集
    train_indices = []
    for i in group:
        train_indices += [idx for idx, label in enumerate(trainset.targets) if label == i]
    train_subset = torch.utils.data.Subset(trainset, train_indices)
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)

    # 创建模型和优化器
    
    model = loadIfExist(ind, model_dir, ResNet)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练模型
    train(model, trainloader, optimizer, criterion, epochs=10)
    saveModel(model, ind, model_dir)