from collections import OrderedDict

import sys
if len(sys.argv) < 5: #include run_file + number_client + client_index + port + scenario
    print("Not enough argument!")
    exit(0)
    
# sys.path.append('/home/bkcs/Lam_Quang/random')
sys.path.append('C:\\Users\\Admin\\Desktop\\Lab\\Lab_final\\random')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from time import time
import warnings
from randomUtils import *
import flwr as fl
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
random.seed(99)
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Config
client_lr = 0.001
BATCH_SIZE = 64
NUM_CLASSES = 10
NUM_CLIENTS = int(sys.argv[1])
CLIENT_INDEX = int(sys.argv[2])
SCENARIO = int(sys.argv[4])

# Enum scenario
SCENARIO_IID = int(1)
SCENARIO_Quantity_Skew = int(2)
SCENARIO_Label_Skew = int(3)

# Load datasets
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = CIFAR10("./data", train=True, download=True, transform=transform)
testset = CIFAR10("./data", train=False, download=True, transform=transform)

def printNumberOfSample(train_datasets):
    if (CLIENT_INDEX == 0):
        sample_record = []
        for client_index, _ in enumerate(train_datasets):
            client_record = []
            num_labels = {label: 0 for label in range(NUM_CLASSES)}
            for sample, label in train_datasets[client_index]:
                num_labels[label] += 1
            for key, value in num_labels.items():
                client_record.append(value)
            sample_record.append(client_record)
        sample_record = np.array(sample_record)
        
        # Print
        clients = [f'Client {i}' for i in range(NUM_CLIENTS)]
        classes = [f'Class {i}' for i in range(NUM_CLASSES)]
        fig, ax = plt.subplots(figsize=(10,6))

        # Plot each category as a stacked bar
        bottom = np.zeros(NUM_CLIENTS)
        colors = [generate_random_color() for _ in range(NUM_CLASSES)]
        for i in range(NUM_CLASSES):
            ax.bar(clients, sample_record[:, i], bottom=bottom, label=f'{classes[i]}', color=colors[i])
            bottom += sample_record[:, i]

        # Customize plot
        ax.set_xlabel('Clients')
        ax.set_ylabel('Samples')
        ax.set_title('Client datasets distribution')
        ax.legend()
        plt.xticks(rotation=45)
        this_dir = Path.cwd()
        save_dir = this_dir / "client_sample"
        label = None
        match SCENARIO:
            case 1:
                label = "iid"
            case 2:
                label = "quantity_skew"
            case _:
                label = "label_skew" 
                
        fig.savefig(save_dir / str("cifar10_" + label + ".png"))

trainloader = None
testloader = None
if (SCENARIO == SCENARIO_IID):
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    train_datasets = random_split(trainset, lengths, torch.Generator().manual_seed(99))

    partition_size = len(testset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    test_datasets = random_split(testset, lengths, torch.Generator().manual_seed(99))

    trainloader = DataLoader(train_datasets[CLIENT_INDEX], batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_datasets[CLIENT_INDEX], batch_size=BATCH_SIZE)
    printNumberOfSample(train_datasets)
    exit(0)
    
elif (SCENARIO == SCENARIO_Quantity_Skew):
    lengths = quantitySkew(len(trainset), NUM_CLIENTS)
    train_datasets = random_split(trainset, lengths, torch.Generator().manual_seed(99))

    lengths = quantitySkew(len(testset), NUM_CLIENTS)
    test_datasets = random_split(testset, lengths, torch.Generator().manual_seed(99))

    trainloader = DataLoader(train_datasets[CLIENT_INDEX], batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_datasets[CLIENT_INDEX], batch_size=BATCH_SIZE)
    printNumberOfSample(train_datasets)
    exit(0)
    
else:
    group_samples_train = [[] for _ in range(NUM_CLIENTS)]
    group_samples_test = [[] for _ in range(NUM_CLIENTS)]
    
    # Group samples by label for train set
    for sample, label in trainset:
        if (random.random() > 0.5):
            group_id = label % NUM_CLIENTS  # Assign the group based on the label
        else:
            group_id = random.randint(0, NUM_CLIENTS - 1)
        group_samples_train[group_id].append((sample, label))

    # Group samples by label for test set
    for sample, label in testset:
        if (random.random() > 0.5):
            group_id = label % NUM_CLIENTS  # Assign the group based on the label
        else:
            group_id = random.randint(0, NUM_CLIENTS - 1)
        group_samples_test[group_id].append((sample, label))
        
    trainloader = DataLoader(group_samples_train[CLIENT_INDEX], batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(group_samples_test[CLIENT_INDEX], batch_size=BATCH_SIZE)
    printNumberOfSample(group_samples_train)
    exit(0)
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net().to(DEVICE)

def train(net, trainloader, train_time, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=client_lr, momentum=0.9)
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
