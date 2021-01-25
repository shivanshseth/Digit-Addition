import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#loading data
X = np.load('../Data/data0.npy')
y = np.load('../Data/lab0.npy')
#print(y.shape)
#for i in [1, 2]: 
#    Xt = np.load('../Data/data' + str(i) +  '.npy')
#    yt = np.load('../Data/lab' + str(i) +'.npy')
#    X = np.concatenate((X, Xt))
#    y = np.concatenate((y, yt))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("split done")
## Padding to make it square
#tp = ((0, 0), (64, 64), (0, 0))
#X_train = np.pad(X_train, pad_width=tp, mode='constant', constant_values=0)
X_train = torch.Tensor([[i] for i in X_train])
#X_test = np.pad(X_test, pad_width=tp, mode='constant', constant_values=0)
X_test = torch.Tensor([[i] for i in X_test])
batch_size = 300

print("Converted to tensor")

class DigitAdditionDataset(Dataset):
    def __init__(self, X, y):
        self.x = X
        self.n_samples = X.shape[0]
        self.y = torch.Tensor(y).long()
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

traindataset = DigitAdditionDataset(X_train, y_train)
valdataset = DigitAdditionDataset(X_test, y_test)
trainloader = DataLoader(dataset=traindataset, batch_size=batch_size, shuffle=True, num_workers=1)
print("dataloader made")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            #nn.BatchNorm2d(8),
            nn.ReLU()
            )
        self.layer01 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            #nn.BatchNorm2d(16),
            nn.ReLU()
            )
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)))
        # (16, 40 , 168) -> (32, 40, 84)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)))
        # (32, 40, 84) -> (64, 40, 42)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=(4,3)),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, 40, 42) -> (64, 44, 44) -> (64, 22, 22)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, 22, 22) -> (64, 11, 11)
        self.fc1 = nn.Linear(64*11*11, 2000)
        self.fc2 = nn.Linear(2000, 50)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer01(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out



# In[153]:


model = Net()
model= nn.DataParallel(model)
model = model.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("model made")

# In[154]:


# Train the model
def train_model(model, trainloader, valdataset, num_epochs=100, saveweights=True, loadweights=False, weightsfile=None):
    print("starting train")
    total_step = len(trainloader)
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        for i, (images, label) in enumerate(trainloader):
            model.train()
            # Run the forward pass
            images = images.to(device)
            label = label.to(device)
            outputs = model(images)
            #print("OUTPUT DEVICE", outputs.device, label.device)
            loss = criterion(outputs, label)
            #train_loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track the accuracy
            total = label.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == label).sum().item()
            #train_acc_list.append(correct / total)

            
        print('Training: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
        train_acc_list.append(correct / total)
        train_loss_list.append(loss.item())

        model.eval()
        # Run the forward pass
        images, label = valdataset.x, valdataset.y
        images = images.to(device)
        label = label.to(device)
        outputs = model(images)
        #print("OUTPUT DEVICE", outputs.device, label.device)
        loss = criterion(outputs, label)
        val_loss_list.append(loss.item())

        # Track the accuracy
        total = label.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == label).sum().item()
        val_acc_list.append(correct / total)

        print('Validation: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                      (correct / total) * 100))
        if saveweights:
           torch.save(model, 'trained_model')
        
    plt.title("Curve:Loss") 
    plt.plot(range(len(train_loss_list)), train_loss_list, label="Train") 
    plt.plot(range(len(train_loss_list)), val_loss_list, label="Validation") 
    plt.xlabel("Iterations") 
    plt.ylabel("Loss")
    plt.savefig('Loss_curve.png')
    plt.title("Curve:Accuracy") 
    plt.plot(range(len(train_loss_list)), train_acc_list, label="Train") 
    plt.plot(range(len(train_loss_list)), val_acc_list, label="Validation") 
    plt.xlabel("Iterations") 
    plt.ylabel("Loss")
    plt.savefig('acc_curve.png')

train_model(model, trainloader, valdataset, 100) 
