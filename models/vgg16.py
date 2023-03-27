import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import random_split

model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model.eval()

# Alterar a última camada do classificador para 4 classes
num_classes = 4
model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=num_classes)

# Definir transformações de dados e caminho do dataset
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'D:\Dataset\BID_Dataset'
full_dataset = datasets.ImageFolder(data_dir, data_transforms)

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
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Definir o número de épocas
num_epochs = 1

# Adicionar listas para armazenar valores de perda e acurácia - Para cada época.
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Adicionar listas para armazenar valores de perda e acurácia de cada batch - Pois está aprendendo muito rápido.
batch_train_losses = []
batch_val_losses = []
batch_train_accuracies = []
batch_val_accuracies = []


# Treinar o modelo
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        
        running_loss = 0.0
        running_corrects = 0

        # Obtenha o DataLoader e o conjunto de dados para a fase atual
        dataloader = dataloaders[phase]
        dataset = train_dataset if phase == 'train' else val_dataset

        # Adicionar um contador de batches
        batch_counter = 0

        # Iterar sobre o DataLoader usando índices
        for idx, (inputs, labels) in enumerate(dataloader):
            # Obter o caminho da imagem atual
            img_path, _ = dataset.dataset.samples[dataset.indices[idx]]

            # Imprimir o caminho da imagem
            # print(f"Processando a imagem: {img_path}")

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Calcular a perda e a acurácia do batch atual
            batch_loss = loss.item()
            batch_corrects = torch.sum(preds == labels.data).item()
            batch_acc = batch_corrects / inputs.size(0)

            # Imprimir a perda e a acurácia do batch atual
            # print(f"Batch {batch_counter}: Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}")


            # Essa parte é apenas para plotar o gráfico por batch, já que vou treinar por uma época apenas.
            if phase == 'train':
                batch_train_losses.append(batch_loss)
                batch_train_accuracies.append(batch_acc)
            else:
                batch_val_losses.append(batch_loss)
                batch_val_accuracies.append(batch_acc)

            # Incrementar o contador de batches
            batch_counter += 1

        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects.double() / len(dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # No final de cada época, armazenar os valores de perda e acurácia
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        val_losses.append(epoch_loss)
        val_accuracies.append(epoch_acc)