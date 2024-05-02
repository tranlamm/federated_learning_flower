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
    list_method = [ 
                "raw_guic256_50s", 
                "raw_guic256_100s", 
                "raw_gquic256_150s",
                "gquic_256_sac_new",
               ]
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
    
objName = {
    "raw_guic256_50s" : "Fixed 50 seconds training time per round",
    "raw_guic256_100s" : "Fixed 100 seconds training time per round",
    "raw_gquic256_150s" : "Fixed 150 seconds training time per round",
    "gquic_256_sac_new" : "Soft Actor Critic (SAC)",
}

fig_loss, f_loss = plt.subplots(figsize=(14,8))
fig_acc, f_acc = plt.subplots(figsize=(14,8))
fig_reward, f_reward = plt.subplots(figsize=(14,8))
fig_time, f_time = plt.subplots(figsize=(14,8))
fig_time2, f_time2 = plt.subplots(figsize=(14,8))

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
    
    tmp = 0
    cnt = 0
    listTime = []
    for i in range(80000):
        if cnt == len(train_time):
            break
        if tmp < train_time[cnt]:
            tmp += 1
        else:
            cnt += 1
            tmp = 0
        listTime.append(cnt)
    
    listTimeIdx = [i for i in range(0, len(listTime))]
    f_loss.plot(time, loss, label=objName[method])
    f_acc.plot(time, acc, label=objName[method])
    f_reward.plot(time, reward, label=objName[method])
    f_time.plot(round, train_time, label=objName[method])
    f_time2.plot(listTimeIdx, listTime, label=objName[method])
    
    # LOSS
f_loss.set_xlabel('Time')
f_loss.set_ylabel('Loss')
# f_loss.set_title('Training Loss')
if not isFull:
    if not isCifar:
        f_loss.set_ylim(0.0148, 0.02)
    else:
        f_loss.set_ylim(-0.0005, 0.02)
f_loss.legend()
fig_loss.savefig(save_dir / "training_loss.png")

    # ACCURACY 
f_acc.set_xlabel('Time', fontsize="24")
f_acc.xaxis.set_tick_params(labelsize=20)
f_acc.xaxis.set_label_coords(0.5, -0.09)
f_acc.set_ylabel('Accuracy', fontsize="24")
f_acc.yaxis.set_tick_params(labelsize=20)
f_acc.yaxis.set_label_coords(-0.1, 0.5)
# f_acc.set_title('Training acc')
if not isFull:
    if not isCifar:
        f_acc.set_ylim(0.75, 0.92)
    else:
        f_acc.set_ylim(0.5, 1.01)
f_acc.legend(loc = "lower right", fontsize="24")
fig_acc.savefig(save_dir / "training_acc.png")

    # REWARD 
f_reward.set_xlabel('Time', fontsize="24")
f_reward.xaxis.set_tick_params(labelsize=20)
f_reward.xaxis.set_label_coords(0.5, -0.09)
f_reward.set_ylabel('Reward', fontsize="24")
f_reward.yaxis.set_tick_params(labelsize=17)
f_reward.yaxis.set_label_coords(-0.125, 0.5)
# f_reward.set_title('Training reward')
if not isFull:
    if not isCifar:
        f_reward.set_ylim(-0.02, -0.0155)
    else:
        f_reward.set_ylim(-0.02, 0.0001)
f_reward.legend(loc = "lower right", fontsize="24")
fig_reward.savefig(save_dir / "training_reward.png")

# f_time.set_xlabel('Round')
# f_time.set_ylabel('Time')
# f_time.set_title('Training time per round')
# f_time.legend()
# fig_time.savefig(save_dir / "training_time_per_round.png")

# f_time2.set_xlabel('Time')
# f_time2.set_ylabel('Epoch')
# f_time2.set_title('Number epoch')
# f_time2.legend()
# fig_time2.savefig(save_dir / "epoch.png")