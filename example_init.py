import torchvision.datasets as dsets
from torchvision import transforms
import os
import shutil
def setup_params(fld):
    if os.path.exists(fld):
        shutil.rmtree(fld)
    os.makedirs(fld)
    train_data = dsets.CIFAR10(root = './data', train = True,
                        transform = transforms.ToTensor(), download = True)

    test_data = dsets.CIFAR10(root = f'./data', train = False,
                       transform = transforms.ToTensor())
    
    
    with open(f"{fld}/params.py", "w") as fp:
        fp.write("""
from torch.utils.data import Dataset
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data import Dataset
import torch
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
opt_cls = torch.optim.RAdam
trn_ldr = train_loader
vld_ldr = valid_loader
crit = torch.nn.MSELoss()
""")
    with open(f"{fld}/__init__.py", "w"):
        pass


def setup_learn(fld):
    with open(f"{fld}/learn.py", "w") as fp:
        fp.write(f"""
import torch
import gymnasium
from stable_baselines3 import PPO
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from {fld}.params import opt_cls
from {fld}.params import trn_ldr
from {fld}.params import vld_ldr
from {fld}.params import crit
env_params = {{
    'opt_cls': opt_cls,
    'trn_ldr': trn_ldr,
    'vld_ldr': vld_ldr,
    'crit': crit,
}}
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

# with open("example.env", "wb") as f:
    # pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


dirname = os.path.dirname(__file__)

from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, BaseCallback
# checkpoint_callback = CheckpointCallback(save_freq=1, save_path=f'{{dirname}}/logs/', name_prefix='ex_model')
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
    def _on_step(self) -> bool:
        PPO.save(self.model, f'{{dirname}}/model')
        if self.n_calls >= self.model._total_timesteps:
            return False
        else:
            return True

the_callback = CustomCallback()

model = PPO("MultiInputPolicy", env, verbose=1)
print(model)
model.learn(total_timesteps=6, progress_bar=True, callback=the_callback)
""")
    pass
    


if __name__ == "__main__":
    fld = "ex_nn_task"
    setup_params(fld)
    setup_learn(fld)