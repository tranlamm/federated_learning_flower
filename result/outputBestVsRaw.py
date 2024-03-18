import sys
if len(sys.argv) < 3:
    print("Not enough argument!")
    exit(0)
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

this_dir = Path.cwd()
isFull = int(sys.argv[1])
isCifar = int(sys.argv[2])
if not isCifar:
    output_dir = this_dir / "gquic256_saved_new"
    if (isFull):
        save_dir = this_dir / "final_result" / "full" / "gquic256"
    else:
        save_dir = this_dir / "final_result" / "zoom" / "gquic256"
    list_method = ["gquic256_raw_50", 
               "gquic_256_raw_100s_", 
               "gquic_256_raw_150s_", 
               "gquic256_ppo"]
else:
    output_dir = this_dir / "cifar10_saved_new"
    if (isFull):
        save_dir = this_dir / "final_result" / "full" / "cifar10"
    else:
        save_dir = this_dir / "final_result" / "zoom" / "cifar10"
    list_method = ["cifar_10_raw_50", 
               "cifar_10_raw_100", 
               "cifar_10_raw_150", 
               "cifar_10_sac"]

fig_loss, f_loss = plt.subplots(figsize=(10,6))
fig_acc, f_acc = plt.subplots(figsize=(10,6))
fig_reward, f_reward = plt.subplots(figsize=(10,6))
fig_time, f_time = plt.subplots(figsize=(10,6))

for method in list_method:
    specific_output_dir = output_dir / method
    loss_file = str(specific_output_dir) + '/val_loss.pkl'
    acc_file = str(specific_output_dir) + '/val_acc.pkl'
    time_file = str(specific_output_dir) + '/train_time.pkl'
    reward_file = str(specific_output_dir) + '/reward.pkl'

    with open(loss_file, 'rb') as f:
        loss = pickle.load(f)
        
    with open(acc_file, 'rb') as f:
        acc = pickle.load(f)

    with open(reward_file, 'rb') as f:
        reward = pickle.load(f) 
        
    with open(time_file, 'rb') as f:
        train_time = pickle.load(f)    
        
    round = np.arange(1, len(train_time) + 1)
    sum = 0
    time = []
    for x in train_time:
        sum += x
        time.append(sum)
    time = np.array(time)
    
    f_loss.plot(time, loss, label=method)
    f_acc.plot(time, acc, label=method)
    f_reward.plot(time, reward, label=method)
    f_time.plot(round, train_time, label=method)
    
f_loss.set_xlabel('Time')
f_loss.set_ylabel('Loss')
f_loss.set_title('Training Loss')
if not isFull:
    if not isCifar:
        f_loss.set_ylim(0.0148, 0.02)
    else:
        f_loss.set_ylim(-0.0005, 0.02)
f_loss.legend()
fig_loss.savefig(save_dir / "training_loss.png")

f_acc.set_xlabel('Time')
f_acc.set_ylabel('Acc')
f_acc.set_title('Training acc')
if not isFull:
    if not isCifar:
        f_acc.set_ylim(0.8, 1)
    else:
        f_acc.set_ylim(0.5, 1.01)
f_acc.legend()
fig_acc.savefig(save_dir / "training_acc.png")

f_reward.set_xlabel('Time')
f_reward.set_ylabel('Reward')
f_reward.set_title('Training reward')
if not isFull:
    if not isCifar:
        f_reward.set_ylim(-0.02, -0.014)
    else:
        f_reward.set_ylim(-0.02, 0.0001)
f_reward.legend()
fig_reward.savefig(save_dir / "training_reward.png")

# f_time.set_xlabel('Round')
# f_time.set_ylabel('Time')
# f_time.set_title('Training time per round')
# f_time.legend()
# fig_time.savefig(save_dir / "training_time_per_round.png")