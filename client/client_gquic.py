from collections import OrderedDict

import sys
if len(sys.argv) < 5: #include run_file + number_client + client_index + port + scenario
    print("Not enough argument!")
    exit(0)
    
# sys.path.append('/home/bkcs/Lam_Quang/random')
sys.path.append('C:\\Users\\Admin\\Desktop\\Lab\\Lab_final\\random')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from time import time
import warnings
from randomUtils import *
import flwr as fl
import random
random.seed(99)
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Config
byte_number = "256"

PACKET_NUM = 20
NUM_FEATURE = int(byte_number)
NUM_CLASSES = 5

client_lr = 0.0001
BATCH_SIZE = 64
NUM_CLIENTS = int(sys.argv[1])
CLIENT_INDEX = int(sys.argv[2])
SCENARIO = int(sys.argv[4])

# Enum scenario
SCENARIO_IID = int(1)
SCENARIO_Quantity_Skew = int(2)
SCENARIO_Label_Skew = int(3)

# Load datasets
# train_dir = '../../../../GQUIC_small/Train/GQUIC_train_' + byte_number + '.feather'
# test_dir = '../../../../GQUIC_small/Test/GQUIC_test_' + byte_number + '.feather'
train_dir = '../GQUIC_small/Train/GQUIC_train_' + byte_number + '.feather'
test_dir = '../GQUIC_small/Test/GQUIC_test_' + byte_number + '.feather'
data = pd.read_feather(train_dir)
test = pd.read_feather(test_dir)

def most_frequent(List):
    return max(set(List), key=List.count)

def load_data_set(data, seed):
    flows = data.groupby('flow_id')['Label'].apply(list).to_dict()
    true_label = []
    for flow in flows:
        true_label.append(most_frequent(flows[flow]))

    true_label = np.array(true_label)
    true_dataset = data.drop(['Label', 'flow_id'], axis=1).to_numpy()/255
    true_dataset = true_dataset.reshape(-1, PACKET_NUM, NUM_FEATURE)
    true_dataset = np.expand_dims(true_dataset, -1)

    true_set = []
    for i in range(true_dataset.shape[0]):
        true_set.append(true_dataset[i].transpose(2, 0, 1))
    true_set = np.array(true_set)

    idx = np.arange(true_set.shape[0])
    np.random.seed(seed) 
    np.random.shuffle(idx)
    true_set = true_set[idx]
    true_label = true_label[idx]
    return true_set, true_label

x_train, y_train = load_data_set(data, 2103)
x_test, y_test = load_data_set(test, 33)

def crop(x_train, y_train):
    length = x_train.shape[0] // NUM_CLIENTS
    x_train = x_train[CLIENT_INDEX*length:(CLIENT_INDEX + 1)*length]
    y_train = y_train[CLIENT_INDEX*length:(CLIENT_INDEX + 1)*length]
    return x_train, y_train

def quantity_skew(x_train, y_train):
    lengths = quantitySkew(x_train.shape[0], NUM_CLIENTS)
    subX = []
    subY = []
    start_index = 0
    for num in lengths:
        subarrayX = x_train[start_index:start_index+num]
        subX.append(subarrayX)
        subarrayY = y_train[start_index:start_index+num]
        subY.append(subarrayY)
        start_index += num
    return subX[CLIENT_INDEX], subY[CLIENT_INDEX]

def label_skew(x_train, y_train):
    group_samples_X = [[] for _ in range(NUM_CLIENTS)]
    group_samples_Y = [[] for _ in range(NUM_CLIENTS)]
    
    # Group samples by label for train set
    for i, label in enumerate(y_train):
        if (random.random() > 0.5):
            group_id = label % NUM_CLIENTS  # Assign the group based on the label
        else:
            group_id = random.randint(0, NUM_CLIENTS - 1)
        group_samples_X[group_id].append(x_train[i])
        group_samples_Y[group_id].append(y_train[i])
    
    return np.array(group_samples_X[CLIENT_INDEX]), np.array(group_samples_Y[CLIENT_INDEX])

if (SCENARIO == SCENARIO_IID):
    x_train, y_train = crop(x_train, y_train)
    x_test, y_test = crop(x_test, y_test)
elif (SCENARIO == SCENARIO_Quantity_Skew):
    x_train, y_train = quantity_skew(x_train, y_train)
    x_test, y_test = quantity_skew(x_test, y_test)
else:
    x_train, y_train = label_skew(x_train, y_train)
    x_test, y_test = label_skew(x_test, y_test)

def to_tensor(x_train, y_train):
    tensor_x = torch.Tensor(x_train) # transform to torch tensor
    tensor_y = torch.Tensor(y_train)
    tensor_y = tensor_y.type(torch.LongTensor)

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    return my_dataset

train_set = to_tensor(x_train, y_train)
test_set = to_tensor(x_test, y_test)

def load_datasets(train_set, test_set):
    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_set, batch_size=BATCH_SIZE) 
    return trainloader, testloader

trainloader, testloader = load_datasets(train_set, test_set)

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(32, 32, (5, 5), padding=(2, 2))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2))
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 5 * 64, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, NUM_CLASSES)
        self.softmax = nn.Softmax(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
net = Net().to(DEVICE)

def train(net, trainloader, train_time, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=client_lr)
    net.train()
    data_iter = iter(trainloader)
    total, total_loss, correct = 0, 0.0, 0
    start = time()
    while True:
        if time() - start > train_time:
            break
        try:
            images, labels = next(data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # Reinitialize data_iter
            data_iter = iter(trainloader)
            images, labels = next(data_iter)

        # Train parameters
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Metric
        total += labels.size(0)
        total_loss += loss
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    return total_loss / total, correct / total

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    # Initialize metrics
    TP, FP, FN = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            
            for class_idx in range(NUM_CLASSES):
                true_class_mask = (labels == class_idx)
                predicted_class_mask = (predicted == class_idx)
                
                # True Positives (TP): Predicted as current class and actually belongs to the current class
                TP += torch.sum(predicted_class_mask & true_class_mask).item()
                # False Positives (FP): Predicted as current class but actually belongs to a different class
                FP += torch.sum(predicted_class_mask & ~true_class_mask).item()
                # False Negatives (FN): Predicted as a different class but actually belongs to the current class
                FN += torch.sum(~predicted_class_mask & true_class_mask).item()
     
    loss /= len(testloader.dataset)
    return loss, TP, FP, FN

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        val_loss, TP, FP, FN = test(net, trainloader)
        train_loss, train_acc = train(net, trainloader, train_time=config["train_time"])
        print("Training time: {}, Local loss: {}, Local acc: {}%".format(config["train_time"], train_loss, train_acc*100))
        return self.get_parameters(config={}), len(trainloader.dataset), {"val_loss": float(val_loss), "train_loss": float(train_loss), "TP": TP, "FP": FP, "FN": FN}

    def evaluate(self, parameters, config):
        return 0, 0, {}
    
# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:" + sys.argv[3], client=FlowerClient())
