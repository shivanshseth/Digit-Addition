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
for i in [1, 2]:
    Xt = np.load('../Data/data' + str(i) + '.npy')
    yt = np.load('../Data/lab' + str(i) + '.npy')
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = torch.Tensor([[i] for i in X_train])
X_test = torch.Tensor([[i] for i in X_test])
batch_size = 300

traindataset = DigitAdditionDataset(X_train, y_train)
valdataset = DigitAdditionDataset(X_test, y_test)
train = DataLoader(dataset=traindataset, batch_size=batch_size, shuffle=True, num_workers=40)
val = DataLoader(dataset=valdataset, batch_size=batch_size, num_workers=40)

"""---

## Model
"""

class Lulz(pl.LightningModule):

    def __init__(self):
        self.ep_num = 0
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
            )
        self.layer01 = nn.Sequential(                                                                                                                                        
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU() 
            )
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)))
        # (16, 40 , 168) -> (32, 40, 84)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)))
        # (32, 40, 84) -> (64, 40, 42)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=(4,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, 40, 42) -> (64, 44, 44) -> (64, 22, 22)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            

        # (64, 22, 22) -> (64, 11, 11)
        self.fc1 = nn.Linear(64*11*11, 2000)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(2000, 40)
        #self.res1 = nn.Linear(2000, 10)
        #self.res2 = nn.Linear(2000, 10)
        #self.res3 = nn.Linear(2000, 10)
        #self.res4 = nn.Linear(2000, 10)
        ## (10, 1, 4) -> (50, 1, 1)
        #self.lconv = nn.Conv2d(10, 50, kernel_size=(4,1),stride=1,padding=0)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer01(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        #print(out.shape) 
        out = F.relu(self.fc1(out))
        #print(out.shape)
        out = self.drop1(out)
        out = self.fc2(out)
        #res1 = self.res1(out)
        #res2 = self.res2(out)
        #res3 = self.res3(out)
        #res4 = self.res4(out)
        #resl = [res1, res2, res3, res4]
        #print(res1.shape)
        #out = torch.stack(resl)
        #print(out.shape)
        #out = torch.reshape(out.T, (10, 4, 1))
        #out = self.lconv(out)
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
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', accuracy, on_epoch=True)
        
        return {'loss': loss, 'accuracy': accuracy}
        #return loss
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
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', accuracy, on_epoch=True)
        return {'loss': loss, 'accuracy': accuracy}
        #return loss, accuracy
        return loss
        # ---------#-----------------

    def training_epoch_end(self, outs): 
        print("Epoch:", self.ep_num, "Training: loss:", outs[0]['loss'].item(), "accuracy:", outs[0]['accuracy'])
        self.ep_num+=1
        #print("Training:", outs)

    def validation_epoch_end(self, outs):
        for out in outs:
            pass
        print("Validation: loss:", outs[0]['loss'].item(), "accuracy:", outs[0]['accuracy'])

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

    # Initialize a trainer
    trainer = pl.Trainer(progress_bar_refresh_rate=0, gpus=4, max_epochs=100, distributed_backend='ddp', callbacks=[EarlyStopping(monitor='val_loss')])

    # Train the model ⚡
    trainer.fit(ae, train, val)


