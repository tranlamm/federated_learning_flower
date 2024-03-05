import sys
if len(sys.argv) < 6: #include run_file + number_client + port + fixed_time_per_round + total_time + save_file_name
    print("Not enough argument!")
    exit(0)
    
import pickle
import numpy as np
from pathlib import Path
from flwr.common import FitIns
import flwr as fl
from time import time

# Config
CUR_TOTAL_TIME = 0
INIT_TIME = int(sys.argv[3])
FIXED_TOTAL_TIME = int(sys.argv[4])
NUM_CLIENTS = int(sys.argv[1])
NUM_ROUNDS = 100000
FRACTION_FIT = 0.8
alpha = 0.01
train_time = []
val_loss = []
val_acc = []
reward = []

def saveFile():
    # Declare storage file
    global val_loss
    global val_acc
    global reward
    global train_time
    
    this_dir = Path.cwd()
    output_dir = this_dir / "flower_saved" / sys.argv[5]
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        
    # Lưu lại history vào file
    val_loss_file = str(output_dir) + '/val_loss.pkl'
    val_loss = np.array(val_loss)
    with open(val_loss_file, 'wb') as f:
        pickle.dump(val_loss, f)
        
    val_acc_file = str(output_dir) + '/val_acc.pkl'
    val_acc = np.array(val_acc)
    with open(val_acc_file, 'wb') as f:
        pickle.dump(val_acc, f)
        
    train_time_file = str(output_dir) + '/train_time.pkl'
    train_time = np.array(train_time)
    with open(train_time_file, 'wb') as f:
        pickle.dump(train_time, f)   
        
    reward_file = str(output_dir) + '/reward.pkl'
    reward = np.array(reward)
    with open(reward_file, 'wb') as f:
        pickle.dump(reward, f)

# Strategy
class CustomStrategy(fl.server.strategy.FedAvg):
    train_time = INIT_TIME

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        global CUR_TOTAL_TIME
        global FIXED_TOTAL_TIME
        print(CUR_TOTAL_TIME)
        if CUR_TOTAL_TIME >= FIXED_TOTAL_TIME:
            saveFile()
            print("END")
            sys.exit()
            
        # calculate reward
        total_example, total_loss = 0, 0
        TP, FP, FN = 0, 0, 0
        for _, res in results:
            TP += res.metrics["TP"]
            FP += res.metrics["FP"]
            FN += res.metrics["FN"]
            total_example += res.num_examples
            total_loss += res.num_examples * res.metrics["val_loss"]
        loss = total_loss / total_example
        curr_reward = -loss - alpha/self.train_time
        reward.append(curr_reward)
        val_loss.append(loss)
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)
        val_acc.append(f1_score)
        train_time.append(self.train_time)
        CUR_TOTAL_TIME += self.train_time
        
        parameters_aggregated, _ = super().aggregate_fit(server_round, results, failures)
        return parameters_aggregated, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        return None, {}

    def configure_fit(self, server_round: int, parameters, client_manager):
        config = {"train_time": self.train_time}
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

# Create FedAvg strategy
strategy = CustomStrategy(
    fraction_fit=FRACTION_FIT,  # Sample % of available clients for training
    fraction_evaluate=0,  # Sample % of available clients for evaluation
    min_fit_clients=int(NUM_CLIENTS*FRACTION_FIT),  # Never sample less than n clients for training
    min_evaluate_clients=0,  # Never sample less than n clients for evaluation
    min_available_clients=int(sys.argv[1]),  # Wait until all n clients are available
)

# Start Flower server
fl.server.start_server(
  server_address="0.0.0.0:" + sys.argv[2],
  config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
  strategy=strategy,
)
