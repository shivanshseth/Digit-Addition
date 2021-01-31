# -*- coding: utf-8 -*-

# Rapid prototyping notebook


# Commented out IPython magic to ensure Python compatibility.
# %%capture
# ! pip install pytorch-lightning
# ! pip install pytorch-lightning-bolts

import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

X = np.load('../Data/data0.npy')
y = np.load('../Data/lab0.npy')
for i in [1]:
    Xt = np.load('../Data/data' + str(i) + '.npy')
    yt = np.load('../Data/lab' + str(i) + '.npy')
    X = np.concatenate((X, Xt))
    y = np.concatenate((y, yt))
Xt = np.load('../Data/data2.npy')[:5000]
yt = np.load('../Data/lab2.npy')[:5000]
X = np.concatenate((X, Xt))
y = np.concatenate((y, yt))


class DigitAdditionDataset(Dataset):
    def __init__(self, X, y):
        self.x = X
        self.n_samples = X.shape[0]
        self.y = torch.Tensor(y).long()
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_train = torch.Tensor([[i] for i in X_train])
X_test = torch.Tensor([[i] for i in X_test])
batch_size = 300

traindataset = DigitAdditionDataset(X_train, y_train)
valdataset = DigitAdditionDataset(X_test, y_test)
#train = DataLoader(dataset=traindataset, batch_size=batch_size, shuffle=True, num_workers=40)
#val = DataLoader(dataset=valdataset, batch_size=batch_size, num_workers=40)


class Data(pl.LightningDataModule):
    def __init__(self, batch_size = 300):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage = None):
        self.train_data = traindataset
        self.val_data = valdataset

    def train_dataloader(self):
        return DataLoader(dataset=traindataset, batch_size=batch_size, shuffle=True, num_workers=40)
    
    def val_dataloader(self):
        return DataLoader(dataset=valdataset, batch_size=batch_size,  num_workers=40)

            
"""---

## Model
"""

class Lulz(pl.LightningModule):

    def __init__(self):
        self.ep_num = 0
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.layerR1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=5, stride=1, padding=2),
            nn.ReLU())

        #self.layerR2 = nn.Sequential(
        #    nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        #    nn.ReLU() 
        #    )

        #self.layerR3 = nn.Sequential(
        #    nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
        #    nn.ReLU() 
        #    )

        # (32, 40 , 168) -> (4, 40, 84)
        self.layerR4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)))

        # (4, 40, 84) -> (48, 40, 42)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)))

        # (48, 40, 42) -> (128, 22, 22)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=(4,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # (128, 22, 22) -> (192, 11, 11)
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # (192, 11, 11) -> (192, 12, 12)
        self.layer4 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=4, stride=1, padding=2),
            nn.ReLU())
        # (192, 12, 12) -> (128, 6, 6)
        self.layer5 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2))
            

        self.fc1 = nn.Linear(128*6*6, 128*6*6)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128*6*6, 2000)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(2000, 37)
        #self.res1 = nn.Linear(2000, 10)
        #self.res2 = nn.Linear(2000, 10)
        #self.res3 = nn.Linear(2000, 10)
        #self.res4 = nn.Linear(2000, 10)
        ## (10, 1, 4) -> (50, 1, 1)
        #self.lconv = nn.Conv2d(10, 50, kernel_size=(4,1),stride=1,padding=0)

    def forward(self, x):
        out = self.layerR1(x)
        #out = self.layerR2(out)
        #out = self.layerR3(out)
        out = self.layerR4(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        #print(out.shape) 
        out = F.relu(self.fc1(out))
        #print(out.shape)
        out = self.drop1(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
        
        

    def training_step(self, batch, batch_idx):
        # --------------------------
        images, label = batch
        outputs = self(images)
        loss = self.criterion(outputs, label)
        total = label.size(0)
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)
        correct = (predicted == label).sum().item()
        accuracy = correct/total * 100
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)
        #print("Training: loss:", loss.item(), "accuracy:", accuracy)
        #return {'loss': loss, 'accuracy': accuracy}
        return loss
        # --------------------------

    def validation_step(self, batch, batch_idx):
        # --------------------------
        images, label = batch
        outputs = self(images)
        loss = self.criterion(outputs, label)
        total = label.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == label).sum().item()
        accuracy = correct/total * 100
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        print("Validation", "accuracy:", accuracy)
        #return {'loss': loss, 'accuracy': accuracy}
        #return loss, accuracy
        # ---------#-----------------

    def training_epoch_end(self, outs): 
        print("Epoch:", self.ep_num, "Training: loss:", outs[0])
        self.ep_num+=1
    #    #print("Training:", outs)

    #def validation_epoch_end(self, outs):
    #    for out in outs:
    #        pass
    #    print("Validation: loss:", outs[0]['loss'].item(), "accuracy:", outs[0]['accuracy'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

"""---
## Train
NOTE: in colab, set progress_bar_refresh_rate high or the screen will freeze because of the rapid tqdm update speed.
"""
if __name__ == '__main__':
    # init model
    ae = Lulz()
    digits = Data()
    # Initialize a trainer
    trainer = pl.Trainer(progress_bar_refresh_rate=0, gpus=4, max_epochs=100, distributed_backend='ddp')
    # Train the model ⚡
    trainer.fit(ae, digits)


