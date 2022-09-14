import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 13
hidden_size = 3
num_classes = 4
num_epochs = 100
batch_size = 4
learning_rate = 0.01

class WineDataset(Dataset):
    
    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]].astype(np.longlong).flatten()) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# create dataset
dataset = WineDataset()



# create data spliter
def dataSplit(dataset, val_split=0.25, shuffle=False, random_seed=0):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler

val_split = 0.25
shuffle_dataset = True
random_seed= 30

train_sampler, valid_sampler = dataSplit(dataset=dataset, val_split=val_split, shuffle=shuffle_dataset, random_seed=random_seed)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)



# 1) model build
class FeedForwardNeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardNeuralNet, self).__init__()
        # define first layer
        self.l1 = nn.Linear(input_size, hidden_size)
        # activation function
        self.relu = nn.ReLU()
        # define second layer
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out


model = FeedForwardNeuralNet(input_size, hidden_size, num_classes)

# 2) loss and optimizer
# Cross Entropy
criterion = nn.CrossEntropyLoss()
# adam algorithm
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 3) Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (datas, labels) in enumerate(train_loader):
        
        # init optimizer
        optimizer.zero_grad()
        
        # forward -> backward -> update
        outputs = model(datas)
        
        loss = criterion(outputs, labels)
        
        loss.backward()

        optimizer.step()

        if (i + 1) % 19 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
            
            
# 4) Testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for datas, labels in val_loader:
        outputs = model(datas.float())
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')