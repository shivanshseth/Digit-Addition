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
for i in [1]:
    Xt = np.load('../Data/data' + str(i) + '.npy')
    yt = np.load('../Data/lab' + str(i) + '.npy')
    X = np.concatenate((X, Xt))
    y = np.concatenate((y, yt))
Xt = np.load('../Data/data2.npy')[:5000]
yt = np.load('../Data/lab2.npy')[:5000]
X = np.concatenate((X, Xt))
y = np.concatenate((y, yt))


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
valoader = DataLoader(dataset=valdataset, batch_size=batch_size, shuffle=True, num_workers=1)
trainloader = DataLoader(dataset=traindataset, batch_size=batch_size, shuffle=True, num_workers=1)

print("dataloader made")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
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
        out = self.drop2(out)
        out = self.fc3(out)
        return out
        
 
# In[153]:

model = Net()
model= nn.DataParallel(model)
model = model.cuda()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("model made")

# In[154]:

# Train the model
def train_model(model, trainloader, valoader, num_epochs=100, saveweights=True, eval_pass=False, weightsfile="./trained_model"):
    print("starting train")
    torch.cuda.empty_cache()
    if eval_pass:
        num_epochs = 1

    total_step = len(trainloader)
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        if not eval_pass:
            for i, (images, label) in enumerate(trainloader):
                model.train()
                # Run the forward pass
                images = images.cuda()
                label = label.cuda()
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
                del label
                del images
                #train_acc_list.append(correct / total)

                
            print('Training: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
            train_acc_list.append(correct / total)
            train_loss_list.append(loss.item())

        torch.cuda.empty_cache()
       
        for images, label in valoader:
            model.eval()
            # Run the forward pass
            images = images.cuda()
            label = label.cuda()
            outputs = model(images)
            #print("OUTPUT DEVICE", outputs.device, label.device)
            loss = criterion(outputs, label)

            # Track the accuracy
            total = label.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == label).sum().item()
        val_acc_list.append(correct / total) 
        val_loss_list.append(loss.item())
        print('Validation: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                      (correct / total) * 100))
    if saveweights:
        torch.save(model.state_dict(), './trained_model')
        
    plt.title("Curve:Loss") 
    plt.plot(range(len(train_loss_list)), train_loss_list, label="Train") 
    plt.plot(range(len(train_loss_list)), val_loss_list, label="Validation") 
    plt.xlabel("Iterations") 
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    plt.title("Curve:Accuracy") 
    plt.plot(range(len(train_loss_list)), train_acc_list, label="Train") 
    plt.plot(range(len(train_loss_list)), val_acc_list, label="Validation") 
    plt.xlabel("Iterations") 
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('acc_curve.png')

train_model(model, trainloader, valoader, 100) 
