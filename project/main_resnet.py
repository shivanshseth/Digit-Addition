import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import sys
import time
from data_preprocess import resnet_train_val_loader, resnet_test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

r = "resnet2"

# Train the model
def train_model(model, trainloader, valoader, num_epochs, criterion, optimizer, schedular, saveweights=True, eval_pass=False, weightsfile="./trained_model"):
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

            schedular.step()
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
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'output/model' + str(r) + '_' + str(epoch))
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
    plt.savefig('output/' + str(r) + 'loss_curve.png')
    plt.close()
    plt.title("Curve:Accuracy") 
    plt.plot(range(len(train_loss_list)), train_acc_list, label="Train") 
    plt.plot(range(len(train_loss_list)), val_acc_list, label="Validation") 
    plt.xlabel("Iterations") 
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('output/' + str(r) + 'acc_curve.png')
    return model

def test_model(model, testloader, criterion):
    total_correct = 0
    total_loss = 0
    n = 0
    for images, label in testloader:
        model.eval()
        # Run the forward pass
        images = images.cuda()
        label = label.cuda()
        outputs = model(images)
        loss = criterion(outputs, label)

            # Track the accuracy
        total = label.size(0)
        n += total
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == label).sum().item()
        total_correct += correct
        total_loss += loss.item()
    
    accuracy = total_correct / n * 100
    loss = total_loss / len(testloader)
    print("Test: Accuracy:", accuracy, "Loss:", loss)
    


if __name__ == '__main__':
    # init model
  print(sys.argv[1])

  if sys.argv[1] == 'train':

    # Loading Data

    if len(sys.argv) < 4:
      X = np.load('../Data/data0.npy')
      y = np.load('../Data/lab0.npy') 
      Xt = np.load('../Data/data1.npy')
      yt = np.load('../Data/lab1.npy')
      X = np.concatenate((X, Xt))
      y = np.concatenate((y, yt))
      Xt = np.load('../Data/data3.npy')
      yt = np.load('../Data/lab3.npy')
      k = np.random.choice(Xt.shape[0], 50000, replace=False)
      X = np.concatenate((X, Xt[k]))
      y = np.concatenate((y, yt[k]))
      Xt = np.load('../Data/data2.npy')[:6000]
      yt = np.load('../Data/lab2.npy')[:6000]
      X = np.concatenate((X, Xt))
      y = np.concatenate((y, yt))
    else:
      dataset_file = sys.argv[2]
      labels_file = sys.argv[3]
      X = np.load(dataset_file)
      y = np.load(labels_file)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #X_train = torch.Tensor([[i] for i in X_train])
    #X_test = torch.Tensor([[i] for i in X_test])
    #batch_size = 800

    #traindataset = DigitAdditionDataset(X_train, y_train)
    #valdataset = DigitAdditionDataset(X_test, y_test)
    #
    #valoader = DataLoader(dataset=valdataset, batch_size=batch_size, shuffle=True, num_workers=1)
    #trainloader = DataLoader(dataset=traindataset, batch_size=batch_size,  num_workers=1)
    tstart = time.time()
    print("Data-preprocessing...")
    trainloader, valoader = resnet_train_val_loader(X, y, 800) 
    print("Done", time.time() - tstart)
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 37)

    model= nn.DataParallel(model)
    model = model.cuda()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9)
    
    print("model made")

    model = train_model(model, trainloader, valoader, 100, criterion=criterion, optimizer=optimizer, schedular=exp_lr_scheduler) 
    

  if sys.argv[1] == 'test':

    if len(sys.argv) < 4:
      print('Syntax:', 'python main.py test <dataset_file> <label_file> <model_checkpoint>')
      sys.exit(1)

    chkpoint = 'output/best'
    if len(sys.argv) == 5:
      chkpoint = sys.argv[4]
    dataset_file = sys.argv[2]
    labels_file = sys.argv[3]
    X = np.load(dataset_file)[6000:]
    y = np.load(labels_file)[6000:]
    
    test = resnet_test_loader(X, y)
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 37)
    model= nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load(chkpoint))
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("model made")

    test_model(model, test, criterion) 
