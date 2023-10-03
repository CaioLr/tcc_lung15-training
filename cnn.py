import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from skimage import io

from multiprocessing import Process, freeze_support

dir_path = os.path.dirname(os.path.realpath(__file__))

class LungDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = io.imread(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

def cnn_por_classe(start, end):
    for quant_classes in range(start, end):
        batch_sizes = [32,64]
    
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
                self.relu2 = nn.ReLU()
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.fc1 = nn.Linear(in_features=64*32*32, out_features=128)
                self.relu3 = nn.ReLU()
                self.fc2 = nn.Linear(in_features=128, out_features=quant_classes)
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.pool1(x)
                x = self.conv2(x)
                x = self.relu2(x)
                x = self.pool2(x)
                x = x.view(-1, 64*32*32)
                x = self.fc1(x)
                x = self.relu3(x)
                x = self.fc2(x)
                x = self.softmax(x)
                return x

        net = Net()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        
        for batch_size in batch_sizes:
            
            transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((128,128)),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomRotation((90,90),expand=False, center=None, fill=0),

            ])

            training_data = LungDataset(annotations_file = fr'{dir_path}\lung15\train_lung15_{quant_classes}.csv', img_dir = fr'{dir_path}\lung15\train', transform=transform)
            test_data = LungDataset(annotations_file = fr'{dir_path}\lung15\test_lung15_{quant_classes}.csv', img_dir = fr'{dir_path}\lung15\test', transform=transform)

            trainloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

            testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
            
            all_classes = ['normal','covid','pneumonia','atelectasia','massa','consolidacao','fibrosi','infiltracao','efisema','pneumotorax','edema','cardiomegalia','efusao','espessamento','nodulo']
            classes = []

            for i in range(quant_classes):
                classes.append(all_classes[i])  
                
            classes = tuple(classes)
            
            num_epochs = 5

            for epoch in range(num_epochs):
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()          
            
            PATH = f'./models/{quant_classes}/pulmao_{quant_classes}_{batch_size}.pth'
            torch.save(net.state_dict(), PATH)
            
            net = Net()
            net.load_state_dict(torch.load(PATH))
            
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Acurácia da rede neural  com {quant_classes} classes e batch_size = {batch_size}: {100 * correct / total} %')

if __name__ == '__main__':
    freeze_support()
    
    # Criando duas instâncias de Process para imprimir números em paralelo
    p1 = Process(target=cnn_por_classe, args=(7, 9))
    p2 = Process(target=cnn_por_classe, args=(9, 11))
    p3 = Process(target=cnn_por_classe, args=(11, 13))
    p4 = Process(target=cnn_por_classe, args=(13, 16))
    # Iniciando os processos
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    
    # Aguardando a finalização dos processos
    p1.join()
    p2.join()
    p3.join()
    p4.join()