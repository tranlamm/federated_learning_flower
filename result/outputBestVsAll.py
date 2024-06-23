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
font = 26
if not isCifar:
    output_dir = this_dir / "gquic256_saved_new"
    if (isFull):
        save_dir = this_dir / "final_result" / "full" / "gquic256_all"
    else:
        save_dir = this_dir / "final_result" / "zoom" / "gquic256_all"
    list_method = [
                "dql_gquic256_latest",
                "ddpg_gquic256_latest",
                "ppo_gquic256_latest",
                "sac_gquic256_latest",
                ]
else:
    output_dir = this_dir / "cifar10_saved_new"
    if (isFull):
        save_dir = this_dir / "final_result" / "full" / "cifar10_all"
    else:
        save_dir = this_dir / "final_result" / "zoom" / "cifar10_all"
    list_method = ["cifar10_dql_ucb1", 
               "cifar10_dql_epsilon_greedy", 
               "cifar10_dql_softmax", 
               "cifar_10_ddpg_epsilon_greedy",
               "cifar_10_ddpg_ucb1",
               "cifar_10_ppo",
               "cifar_10_sac"]
    font = 17

fig_loss, f_loss = plt.subplots(figsize=(14,8))
fig_acc, f_acc = plt.subplots(figsize=(14,8))
fig_reward, f_reward = plt.subplots(figsize=(14,8))
fig_time, f_time = plt.subplots(figsize=(14,8))

objName = {
    "dql_gquic256_latest" : "DQL",
    "ddpg_gquic256_latest" : "DDPG",
    "ppo_gquic256_latest" : "PPO",
    "sac_gquic256_latest" : "SAC",
    
    "cifar10_dql_ucb1" : "DQL_UCB1",
    "cifar10_dql_epsilon_greedy" : "DQL_EpsilonGreedy",
    "cifar10_dql_softmax" : "DQL_Softmax",
    "cifar_10_ddpg_epsilon_greedy" : "DDPG_EpsilonGreedy",
    "cifar_10_ddpg_ucb1" : "DDPG_UCB1",
    "cifar_10_ppo" : "PPO",
    "cifar_10_sac" : "SAC",
}

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
    index = 0
    for x in train_time:
        sum += x
        time.append(sum)
        if sum < 38000:
            index += 1
    time = np.array(time)
    
    print(objName[method] + " " + str(index) + " " + str(acc[index]) + " " + str(reward[index]))
    
    f_loss.plot(time, loss, label=objName[method])
    f_acc.plot(time, acc, label=objName[method])
    f_reward.plot(time, reward, label=objName[method])
    f_time.plot(round, train_time, label=objName[method])
    
    # LOSS
f_loss.set_xlabel('Time', fontsize="22")
f_loss.xaxis.set_tick_params(labelsize=17)
f_loss.xaxis.set_label_coords(0.5, -0.09)
f_loss.set_ylabel('Loss', fontsize="22")
f_loss.yaxis.set_tick_params(labelsize=17)
f_loss.yaxis.set_label_coords(-0.125, 0.5)
# f_loss.set_title('Training Loss')
if not isFull:
    if not isCifar:
        f_loss.set_ylim(0.0157, 0.020)
    else:
        f_loss.set_ylim(-0.001, 0.02)
f_loss.legend(loc = "upper right", fontsize=str(font))
fig_loss.savefig(save_dir / "training_loss.png")

    # ACCURACY 
f_acc.set_xlabel('Time', fontsize="22")
f_acc.xaxis.set_tick_params(labelsize=17)
f_acc.xaxis.set_label_coords(0.5, -0.09)
f_acc.set_ylabel('Accuracy', fontsize="22")
f_acc.yaxis.set_tick_params(labelsize=17)
f_acc.yaxis.set_label_coords(-0.1, 0.5)
# f_acc.set_title('Training acc')
if not isFull:
    if not isCifar:
        f_acc.set_ylim(0.75, 0.92)
    else:
        f_acc.set_ylim(0.4, 1.05)
f_acc.legend(loc = "lower right", fontsize=str(font))
fig_acc.savefig(save_dir / "training_acc.png")

    # REWARD 
f_reward.set_xlabel('Time', fontsize="22")
f_reward.xaxis.set_tick_params(labelsize=17)
f_reward.xaxis.set_label_coords(0.5, -0.09)
f_reward.set_ylabel('Reward', fontsize="22")
f_reward.yaxis.set_tick_params(labelsize=17)
f_reward.yaxis.set_label_coords(-0.125, 0.5)
# f_reward.set_title('Training reward')
if not isFull:
    if not isCifar:
        f_reward.set_ylim(-0.02, -0.0155)
    else:
        f_reward.set_ylim(-0.025, 0.001)
f_reward.legend(loc = "lower right", fontsize=str(font))
fig_reward.savefig(save_dir / "training_reward.png")

# f_time.set_xlabel('Round')
# f_time.set_ylabel('Time')
# f_time.set_title('Training time per round')
# f_time.legend()
# fig_time.savefig(save_dir / "training_time_per_round.png")