import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt

# Transformações para as imagens (normalização e redimensionamento)
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # Redimensiona as imagens
    transforms.ToTensor(),           # Converte as imagens para tensores
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalização
])

# Carregar os dados de treinamento e validação
train_data = datasets.ImageFolder('dataset/train', transform=transform)
val_data = datasets.ImageFolder('dataset/val', transform=transform)

# DataLoader para carregar os dados em batches
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Definir o modelo (ResNet-18)
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Modificando a camada final para 2 classes (cão e gato)

# Configurar o dispositivo (CUDA ou CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Função de treinamento
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

# Função de validação
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

# Treinamento do modelo
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

# Salvar o modelo treinado
torch.save(model.state_dict(), 'model_cats_and_dogs.pth')

# Exibir algumas previsões
def show_predictions(model, val_loader, device):
    model.eval()
    data_iter = iter(val_loader)
    inputs, labels = next(data_iter)
    inputs, labels = inputs.to(device), labels.to(device)
    
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    for i in range(4):
        ax = axes[i]
        ax.imshow(inputs[i].cpu().permute(1, 2, 0))
        ax.set_title(f"Pred: {predicted[i].item()}, True: {labels[i].item()}")
        ax.axis('off')
    
    plt.show()

show_predictions(model, val_loader, device)