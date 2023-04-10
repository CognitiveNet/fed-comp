import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split
from unet import UNet  # Substitua isso pela implementação correta do UNet

# Carregar o modelo UNet (ou outro modelo de segmentação)
model = UNet(num_classes=4)
model.eval()

# Definir transformações de dados e caminho do dataset
data_transforms = {
    'image': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'mask': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
}

data_dir = 'D:\Dataset\BID_Dataset'
full_dataset = SegmentationDataset(data_dir, data_transforms)  # Substitua isso pela implementação correta da classe SegmentationDataset

# Calcular os tamanhos de treinamento e validação
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Dividir o conjunto de dados em treinamento e validação
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Criar os carregadores de dados
batch_size = 32
num_workers = 4
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

dataloaders = {'train': train_loader, 'val': val_loader}

# Definir dispositivo (GPU se disponível)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Mover o modelo para o dispositivo
model.to(device)

# Definir a função de perda e otimizador
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# O restante do código permanece o mesmo, mas certifique-se de que os rótulos são máscaras e não rótulos de classe
