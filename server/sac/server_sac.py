import sys
if len(sys.argv) < 5: #include run_file + number_client + port + total_time + save_file_name
    print("Not enough argument!")
    exit(0)
    
sys.path.append('/home/bkcs/Lam_Quang/sac')

from sklearn.preprocessing import normalize
import pickle
import numpy as np
from pathlib import Path
from flwr.common import FitIns
import flwr as fl
from sac import SACAgent

# Config
INIT_TIME = 100
MIN_TIME = 50
MAX_TIME = 150
CUR_TOTAL_TIME = 0
FIXED_TOTAL_TIME = int(sys.argv[3])
NUM_CLIENTS = int(sys.argv[1])
NUM_ROUNDS = 100000
FRACTION_FIT = 0.8
state_dims = 3 * int(NUM_CLIENTS * FRACTION_FIT) + 1
action_dims = 3
BATCH_SIZE = 64
alpha = 0.01

# Variable
agent = SACAgent(state_dims=state_dims, action_dims=action_dims, hidden_dims=64, batch_size=BATCH_SIZE)
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
   
    # Declare storage file
    this_dir = Path.cwd()
    output_dir = this_dir / "flower_saved" / sys.argv[4]
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
    pre_state = None
    pre_action = None

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        global CUR_TOTAL_TIME
        global FIXED_TOTAL_TIME
        if CUR_TOTAL_TIME >= FIXED_TOTAL_TIME:
            saveFile()
            print("END")
            sys.exit()
        
        parameters_aggregated, _ = super().aggregate_fit(server_round, results, failures)
        
        # get state
        fit_metrics = [(res.metrics["val_loss"], res.metrics["train_loss"], res.num_examples) for _, res in results]
        fit_metrics = np.array(fit_metrics)
        curr_state = np.concatenate((fit_metrics[:, 0], fit_metrics[:, 1], fit_metrics[:, 2], np.array([self.train_time])))
        curr_state = normalize(curr_state.reshape(1, -1))[0]
        
        # calculate reward
        total_example, total_loss = 0, 0
        TP, FP, FN = 0, 0, 0
        for _, res in results:
            TP += res.metrics["TP"]
            FP += res.metrics["FP"]
            FN += res.metrics["FN"]
            total_loss += res.num_examples * res.metrics["val_loss"]
            total_example += res.num_examples
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
            
        # add to experiment buffer
        if self.pre_state is not None:
            agent.train_on_transition(self.pre_state, self.pre_action, curr_reward, curr_state)
        self.pre_state = curr_state
        
        # predict next action
        action = agent.get_next_action(curr_state, False)
        # action mapping: 0 -> -1, 1 -> 0, 2 -> 1
        if action == 0:
            timeDiff = -1
        elif action == 1:
            timeDiff = 0
        else:
            timeDiff = 1
        
        nextTrainTime = self.train_time + timeDiff
        if nextTrainTime > MAX_TIME:
            nextTrainTime = MAX_TIME
        elif nextTrainTime < MIN_TIME:
            nextTrainTime = MIN_TIME
                
        self.pre_action = action
        self.train_time = nextTrainTime
        
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

