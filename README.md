#### Goals  
The goal of this model is to predict the sum of 4 handwritten digits from an 40x168 px image.

## Description   

#### Datasets
A dataset containing 30000 images and their corresponding labels was provided for the assignment. An additional dataset of 50000 data points was created by combining individual images from the MNIST Dataset. This dataset was used for training the model.

#### Models

Two different models have been trained on the dataset, one CNN from scratch and a pretrained "Resnet18" model finetuned on the dataset.

#### 1. CNN trained from scratch 

| Num | Name     | Type             | Params | 
------|----------|------------------|--------|
| 0  | criterion | CrossEntropyLoss | 0      |
| 1  | layerR1   | Sequential       | 1.2 K  |
| 2  | layerR4   | Sequential       | 76.9 K |
| 3  | layer1    | Sequential       | 102 K  |
| 4  | layer2    | Sequential       | 204 K  |
| 5  | layer3    | Sequential       | 614 K  |
| 6  | layer4    | Sequential       | 590 K  |
| 7  | layer5    | Sequential       | 221 K  |
| 8  | fc1       | Linear           | 21.2 M |
| 9  | drop1     | Dropout          | 0      |
| 10 | fc2       | Linear           | 9.2 M  |
| 11 | drop2     | Dropout          | 0      |
| 12 | fc3       | Linear           | 74.0 K |

32.3 M    Trainable params
0         Non-trainable params
32.3 M    Total params

##### Metrics

| Dataset | Accuracy | loss |
----------|----------|------|
| Training | 93.50%    | 0.1957|
| Validation | 94.00%  | 0.2135|
| Test       | 70.675% |  1.08 |

The difference in Validation and Test accuracy can be attributed to the difference in datasets used.

#### 2. Fine-tuned Resnet 18

#### Metrics 

| Dataset | Accuracy | loss |
----------|----------|------|
| Training | 100%    | 0.1957|
| Validation | 94.23%  | 0.453|
| Test       | 95.55%% |  0.394 |

The fine-tuned resnet model clearly out performs model 1. Even though the test set is somewhat different from the training and validation datasets, it generalises well enough.

## How to run   
First, install dependencies

```   
pip -r requirements.txt
# clone project   
git clone https://github.com/shivanshseth/Digit-Addition

# module folder
cd project
```

#### Running model - 1
```
# run module  
python main.py train [data_file] [labels_file]
```

#### Runnning model - 2 (fine-tuned resnet) 
```
# run module  
python main_resnet.py train [data_file] [labels_file]
```


Note: In absence of "data_file" and "data_label" arguments, the model will use files "../Data/data[1-2-3].npy" and "../Data/lab[1-2-3].npy". Assumptions are made about the size of these datasets so its better to provide data set explicitly.


## Testing
```bash
python main.py test data_file labels_file [model_parameters_file]
```
Note: In the absence of model_parameters_file, the model will use "output/best" by default.

# Results
The validation and training curves are stored in 
```
output/${r}_loss_curve.png
output/${r}_accuracy_curve.png
```
The value of "r" is defined at the beginning in "main.py"
```
