import sys
if len(sys.argv) < 2: #include run_file + file_visualize
    print("Not enough argument!")
    exit(0)
    
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

this_dir = Path.cwd()
output_dir = this_dir / "flower_saved" / str(sys.argv[1])
loss_file = str(output_dir) + '/val_loss.pkl'
acc_file = str(output_dir) + '/val_acc.pkl'
time_file = str(output_dir) + '/train_time.pkl'
reward_file = str(output_dir) + '/reward.pkl'

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

fig = plt.figure(figsize=(10, 6))
plt.plot(time, loss, label='loss')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig(output_dir / "training_loss.png")

fig = plt.figure(figsize=(10, 6))
plt.plot(time, acc, label='acc')
plt.xlabel('Time')
plt.ylabel('Acc')
plt.title('Training acc')
plt.legend()
plt.savefig(output_dir / "training_acc.png")

fig = plt.figure(figsize=(10, 6))
plt.plot(time, reward, label='reward')
plt.xlabel('Time')
plt.ylabel('Reward')
plt.title('Training reward')
plt.legend()
plt.savefig(output_dir / "training_reward.png")

fig = plt.figure(figsize=(10, 6))
plt.plot(round, train_time, label='time/round')
plt.xlabel('Round')
plt.ylabel('Time')
plt.title('Training time per round')
plt.legend()
plt.savefig(output_dir / "training_time_per_round.png")