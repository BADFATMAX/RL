import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import gymnasium
from stable_baselines3 import PPO

class myDataset(Dataset):
    def __init__(self, X, y):

      self.X = X
      self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_ = self.X[idx]
        y_ = self.y[idx]
        return x_, y_

train_data = dsets.CIFAR10(root = './data', train = True,
                        transform = transforms.ToTensor(), download = True)

test_data = dsets.CIFAR10(root = './data', train = False,
                       transform = transforms.ToTensor())

import numpy as np
train_samples = np.array(train_data.data).transpose((0,3,1,2)) # convert to [B, C, H, W]
train_labels = np.array(train_data.targets)

dataset = myDataset(X=train_samples, y=train_labels)
train_set, valid_set = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

train_loader = torch.utils.data.DataLoader(
  train_set,
  batch_size=32,
  shuffle=True,
  drop_last=True)

valid_loader = torch.utils.data.DataLoader(
  valid_set,
  batch_size=32,
  drop_last=True,
  shuffle=True)


# some environment params
# optimizer should be class
# because every NN building needs
# pass model.params() to optimizer,
# so we will recreate optimizer every
# time new NN will be created successfully
env_params = {
    'opt_cls': torch.optim.RAdam,
    'trn_ldr': train_loader,
    'vld_ldr': valid_loader,
    'crit': torch.nn.MSELoss(),
}
import gymnasium as gym
from gymnasium.envs.registration import register
register(
    id='gym_env/env_v1-v0',
    entry_point='gym_env.envs:Env',
    max_episode_steps=10,#200
)

env = gymnasium.make('gym_env/env_v1', **env_params)

# Proximal Policy Optimization
# doc: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

model = PPO("MultiInputPolicy", env, verbose=1)
print(model)
model.learn(1, progress_bar=True)
# print(model.learn(total_timesteps=1))