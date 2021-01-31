#### Goals  
The goal of to successfully predict the sum of digits from an image with 4 handwritten.  

## Description   
Architecture

   | Name      | Type             | Params
------------------------------------------------
0  | criterion | CrossEntropyLoss | 0     
1  | layerR1   | Sequential       | 1.2 K 
2  | layerR4   | Sequential       | 76.9 K
3  | layer1    | Sequential       | 102 K 
4  | layer2    | Sequential       | 204 K 
5  | layer3    | Sequential       | 614 K 
6  | layer4    | Sequential       | 590 K 
7  | layer5    | Sequential       | 221 K 
8  | fc1       | Linear           | 21.2 M
9  | drop1     | Dropout          | 0     
10 | fc2       | Linear           | 9.2 M 
11 | drop2     | Dropout          | 0     
12 | fc3       | Linear           | 74.0 K
------------------------------------------------
32.3 M    Trainable params
0         Non-trainable params
32.3 M    Total params


## How to run   
First, install dependencies   
pip -r requirements.txt
```bash
# clone project   
git clone https://github.com/shivanshseth/Digit-Addition

 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module  
python main.py train [data_file] [labels_file]
```

## Testing
```bash
python main.py test data_file labels_file model_parameters_file
```
# Results
To view the training and validation curves:
```bash
tensorboard --logdir project/lightning_logs
```
