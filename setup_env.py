import os


def make_dirs(env_name: str):
    os.makedirs('gym_{}'.format(env_name))
    os.makedirs('gym_{}/envs'.format(env_name))
    pass


def make_files(env_name: str, stage_names):
    registry_init = 'gym_{}/__init__.py'.format(env_name)
    with open(registry_init, 'w') as f:
        #====================CHANGED gym to gymnasium
        f.write('from gymnasium.envs.registration import register\n\n')
        #====================
        for name in stage_names:
            register_str = "register(\n    id='{}',\n    entry_point='gym_{}.envs:{}',\n)\n".format(name,
                                                                                                    env_name,
                                                                                                    ''.join([s.capitalize()for s in name.split('_')]))
            f.write(register_str)
    envs_init = 'gym_{}/envs/__init__.py'.format(env_name)
    with open(envs_init, 'w') as f:
        for name in stage_names:
            import_str = "from gym_{}.envs.{} import {}\n".format(env_name, name,
                                                                  ''.join([s.capitalize() for s in name.split('_')]))
            f.write(import_str)

    head = """import gymnasium as gym
from gymnasium import spaces, utils
#from gymnasium.utils import seeding
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import sys
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Dataset


"""

    tail = """

  metadata = {'render_modes': ['human']}
  # actions agent can do
  # you can add here your layers
  actions_set = [
                # increase channels with kernel 3
                #'conv=channel_factor:2,kernel_size:3,stride:1,padding:0-',
                'conv=channel_factor:4,kernel_size:3,stride:1,padding:0-',
                'conv=channel_factor:8,kernel_size:3,stride:1,padding:0-',
                'conv=channel_factor:16,kernel_size:3,stride:1,padding:0-',
                # increase channels with kernel 5
                #'conv=channel_factor:2,kernel_size:5,stride:1,padding:0-',
                'conv=channel_factor:4,kernel_size:5,stride:1,padding:0-',
                'conv=channel_factor:8,kernel_size:5,stride:1,padding:0-',
                'conv=channel_factor:16,kernel_size:5,stride:1,padding:0-',
                # increase channels with kernel 7
                #'conv=channel_factor:2,kernel_size:7,stride:1,padding:0-',
                #'conv=channel_factor:4,kernel_size:7,stride:1,padding:0-',
                #'conv=channel_factor:8,kernel_size:7,stride:1,padding:0-',
                #'conv=channel_factor:16,kernel_size:7,stride:1,padding:0-',



                # decrease channels with kernel 3
                'conv=channel_factor:0.6,kernel_size:3,stride:1,padding:0-',
                'conv=channel_factor:0.8,kernel_size:3,stride:1,padding:0-',

                # decrease channels with kernel 5
                'conv=channel_factor:0.6,kernel_size:5,stride:1,padding:0-',
                'conv=channel_factor:0.8,kernel_size:5,stride:1,padding:0-',

                # decrease channels with kernel 7
                #'conv=channel_factor:0.6,kernel_size:7,stride:1,padding:0-',
                #'conv=channel_factor:0.8,kernel_size:7,stride:1,padding:0-',


                 'batchnorm=eps:0.00001-',

                 'avgpool=kernel_size:2,stride:2,padding:0-',
                 #'avgpool=kernel_size:3,stride:3,padding:0-',

                 'maxpool=kernel_size:2,stride:2,padding:0-',
                 #'maxpool=kernel_size:3,stride:3,padding:0-',

                 'dropout=p:0.1-',
                 'dropout=p:0.2-',
                 #'dropout=p:0.4-',
  ]

  # Some reward variables
  NN_CREATE_SUCCESS_REWARD = 5
  NN_CREATE_NOT_SUCCESS_PENALTY = -5

  # TODO add not fixed NN length
  DEPTH_REWARD_FACTOR = 0.1
  METRICS_OPTIMIZATION_FACTOR = 1
  NAN_NUM = 100

  def __init__(self, opt_cls=None, crit=None, trn_ldr=None, vld_ldr=None, render_mode=None):
      '''
      Initialization of environment variables
      '''

      # here is learning params
      self.NN_PARAMS = {
          'lr': 0.001,
          'num_classes': 10, # TODO adaptive num_classes
          'train_epochs': 10,
          'last_nets_metrics_memory_len': 10,
          'layers_amount': 5,
          'amount_of_metrics': 2,
      }

      # here contains NN learning metrics
      # uses for observations
      # shape=[MEMORY_LEN, N_METRICS, EPOCHS]
      self.NN_PARAMS['metrics'] = np.zeros((
          self.NN_PARAMS['last_nets_metrics_memory_len'],
          self.NN_PARAMS['amount_of_metrics'],
          self.NN_PARAMS['train_epochs'],
      ))

      # here contains last architectures
      # uses for observations
      # shape=[MEMORY, N_LAYERS]
      self.NN_PARAMS['last_nets_architectures'] = np.zeros((
             self.NN_PARAMS['last_nets_metrics_memory_len'],
             self.NN_PARAMS['layers_amount'],
          ))

      # Variables for NN
      self.Net = self.NN()
      self.train_dataloader = trn_ldr
      self.valid_dataloader = vld_ldr
      self.optimizer_class = opt_cls
      self.optimizer = None
      self.criterion = crit
      self.device = 'cuda'

      self.nngenerator = self.nnGenerator()

      self.last_obs = None

      # Action space describes what agent will give to environment
      # shape=[ACTION_SET_SIZE, N_LAYERS]
      self.action_space = spaces.MultiDiscrete(
        [len(self.actions_set)] * self.NN_PARAMS['layers_amount'],
         seed=42)

      # Observation space describes what agent
      # will take from enviromnent as observation
      # shape=dict{
      #  METRICS, shape as NN_PARAMS['metrics'],
      #  ARCHITECTURES, shape as NN_PARAMS['layers_amount'],
      # }
      self.observation_space = spaces.Dict(
          {
          'last_nets_metrics_memory': spaces.Box(
              low=0,
              high=100,
              shape=(self.NN_PARAMS['last_nets_metrics_memory_len'],
                     self.NN_PARAMS['amount_of_metrics'],
                     self.NN_PARAMS['train_epochs'],
                     )),
          'last_nets_architectures': spaces.Box(
              low=0,
              high=len(self.actions_set),
              shape=(self.NN_PARAMS['last_nets_metrics_memory_len'],
                     self.NN_PARAMS['layers_amount'],
                     ))
          }
      )

      # some variable for collecting statistics

      self.statistics = {
        'episode_rewards': [],
        'global_rewards': [],
        'made_steps': [],
      }

      self.seed()
      assert render_mode is None or render_mode in self.metadata["render_modes"]
      self.render_mode = render_mode


  class NN(nn.Module):
    '''
    NN template
    Agent's net will be put to self.layers

    '''

    def __init__(self):
      super().__init__()

      self.layers = nn.Sequential()

    def forward(self, x):
      x = self.layers(x)
      return x

    def __call__(self, x):
      return self.forward(x)

  def set_train_dataloader(self, train_dataloader):
    self.train_dataloader = train_dataloader

  def set_valid_dataloader(self, valid_dataloader):
    self.valid_dataloader = valid_dataloader

  def set_criterion(self, criterion):
    self.criterion = criterion

  def set_optimizer(self, optimizer):
    self.optimizer_class = optimizer

  def train(self):
    '''
    Simple NN training loop
    You can add your metrics here

    # TODO add smart metrcis choosing

    return: all train/valid metrics
    '''

    train_losses = []
    valid_losses = []

    for i in tqdm(range(0, self.NN_PARAMS['train_epochs'] + 1)):
        losses = []
        for X, Y in self.train_dataloader:
            X = X.float().to(self.device)
            Y = Y.float().to(self.device)
            preds = self.Net(X)
            preds, _ = torch.max(preds,1)
            loss = self.criterion(preds, Y)
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        #print("Train Loss : {:.6f}".format(torch.tensor(losses).mean()))
        if i != 0:
            train_losses.append(torch.tensor(losses).mean())

        with torch.no_grad():
            losses = []
            for X, Y in self.valid_dataloader:
                X = X.float().to(self.device)
                Y = Y.float().to(self.device)
                preds = self.Net(X)
                preds, _ = torch.max(preds,1)
                loss = self.criterion(preds,Y)
                losses.append(loss.item())
            #print("Valid Loss : {:.6f}".format(torch.tensor(losses).mean()))
            if i != 0:
                valid_losses.append(torch.tensor(losses).mean())

    # update all required NN metrics
    #print(len(train_losses), train_losses)
    #print(len(valid_losses), valid_losses)

    return np.array([train_losses, valid_losses])

  class nnGenerator():
    '''
       nnGenerator makes manipulations with
       agent's action (preproc for training)
    '''

    def __init__(self):

      self.text_layers_dict = dict({})
      self.nn_len = -1

    def action_to_text(self, action, action_set):
      '''
      Converts MultiDiscrete action to action_text
      return: res - str
      '''

      res = ""
      for i in action:
        res = res + action_set[int(i)]
      return res

    def parse_action(self, action, action_set):
      '''
      parse action_text
      parsed data puts to self.text_layers_dict
      result dict format: 'layer_name': (id, {params})
          {
            'conv1': (1, {'channel_factor': '123', 'padding': '0', ...}),
            'dropout2': (2, {'p': '0.2'}),
            ...
          }
      return None
      '''

      action = self.action_to_text(action, action_set)
      self.text_layers_dict = dict({})
      #print(action)
      if action[-1] == '-':
        action = action[:-1]
      text_layers = action.split('-')
      id = 0
      for text_layer in text_layers:
        tmp = text_layer.split('=')
        layer_name, layer_params = tmp[0], tmp[1].split(',')
        layer_params_dict = dict({})
        for param in layer_params:
          param = param.split(':')
          param_name, param_value = param[0], param[1]
          layer_params_dict[param_name] = param_value
        self.text_layers_dict[layer_name + str(id)] = (id, layer_params_dict)
        id += 1
      #print(self.text_layers_dict)

    def get_text_layers_dict(self):
      return self.text_layers_dict

    def get_nn_len(self):
      return self.nn_len

    def conv_output_shape(self, h, w, kernel_size=1, stride=1, pad=0, dilation=1):
      '''
      Calculates output of layer by input shape [B, C, H, W]
      return: h - int, w - int
      '''

      h = math.floor( ((h + (2 * pad) - ( dilation * (kernel_size - 1) ) - 1 )/ stride) + 1)
      w = math.floor( ((w + (2 * pad) - ( dilation * (kernel_size - 1) ) - 1 )/ stride) + 1)
      return h, w

    def generateNN(self, n_classes, test_batch): # -> nn.Sequential
      '''
      most important part of that class
      Build Neural Network
      Algorithm:
      1) extract layer params
      2) create layer
      3) assert NN with that layer will be correct
      4) append layer to the backbone
      5) repeat 1-4 for all layers
      6) append linear classifier

      n_classes: amount of classification classes
      test_batch: test data for input data shape extraction
      return success_state: bool (True if created successfully), net: nn.Sequential
      '''
      success_state = False
      backbone = nn.Sequential()
      classifier = nn.Sequential()
      optimizer = None
      try:
        data_shape = np.array(test_batch).shape # [B, C, H, W]
        last_shape = data_shape
        for layer_name in self.text_layers_dict.keys():
            layer = None
            layer_params= self.text_layers_dict[layer_name][1] # [0] - id
            if layer_name.find('conv') >= 0:
              # 'conv=channel_factor:2,kernel_size:5,stride:1,padding:0-',
              kernel_size = int(layer_params['kernel_size'])
              channel_factor = float(layer_params['channel_factor'])
              stride = int(layer_params['stride'])
              padding = int(layer_params['padding'])
              dilation = 1

              activation = nn.ReLU(inplace=True)

              in_chan = last_shape[1]

              assert(in_chan <= last_shape[2] and in_chan <= last_shape[3])
              assert(last_shape[3] > kernel_size and last_shape[2] > kernel_size)
              out_chan = math.floor(in_chan * channel_factor)
              assert(out_chan > 0)

              backbone.append(nn.Conv2d(in_chan, out_chan, kernel_size, stride, padding, dilation))
              backbone.append(activation)

              h, w = self.conv_output_shape(last_shape[2], last_shape[3], kernel_size, stride, padding, dilation)
              last_shape = (last_shape[0], out_chan, h, w)

            elif layer_name.find('batchnorm') >= 0:
              eps = float(layer_params['eps'])
              in_chan = last_shape[1]
              backbone.append(nn.BatchNorm2d(in_chan, eps))

            elif layer_name.find('avgpool') >= 0:
              # 'avgpool=kernel_size:2,stride:2,padding:0-',
              kernel_size = int(layer_params['kernel_size'])
              stride = int(layer_params['stride'])
              padding = int(layer_params['padding'])
              dilation = 1
              in_chan = last_shape[1]
              out_chan = in_chan

              assert(in_chan <= last_shape[2] and in_chan <= last_shape[3])
              assert(last_shape[3] > kernel_size and last_shape[2] > kernel_size)

              backbone.append(nn.AvgPool2d(kernel_size, stride, padding))

              h, w = self.conv_output_shape(last_shape[2], last_shape[3], kernel_size, stride, padding, dilation)
              last_shape = (last_shape[0], out_chan, h, w)


            elif layer_name.find('maxpool') >= 0:
              # 'maxpool=kernel_size:2,stride:2,padding:0-',
              kernel_size = int(layer_params['kernel_size'])
              stride = int(layer_params['stride'])
              padding = int(layer_params['padding'])
              dilation = 1
              in_chan = last_shape[1]
              out_chan = in_chan
              assert(in_chan <= last_shape[2] and in_chan <= last_shape[3])
              assert(last_shape[3] > kernel_size and last_shape[2] > kernel_size)

              backbone.append(nn.MaxPool2d(kernel_size, stride, padding))

              h, w = self.conv_output_shape(last_shape[2], last_shape[3], kernel_size, stride, padding, dilation)
              last_shape = (last_shape[0], out_chan, h, w)

            elif layer_name.find('dropout') >= 0:
              p = float(layer_params['p'])
              backbone.append(nn.Dropout2d(p))
        linear_in_shape = last_shape[1] * last_shape[2] * last_shape[3]
        classifier = nn.Linear(linear_in_shape, n_classes)
        success_state = True
        #print('NN build successfull!')

      except Exception as e:
        #print('NN build failed!')
        #print(str(e))
        pass

      net = nn.Sequential()
      net.append(backbone)
      net.append(nn.Flatten(start_dim=1))
      net.append(classifier)
      self.text_layers_dict = dict({})
      return success_state, net


  def seed(self, seed=None):
      from gymnasium.utils import seeding
      self.np_random, seed = seeding.np_random(seed)
      #seed = 42
      return [seed]


  def calc_reward(self, nn_created_correctly_flag, nn_len, last_train_metrics):
    '''
       calculate agent reward

       nn_created_correctly_flag: bool,
       if True - NN was built successfully,
       we can calculate other parts of reward,
       otherwise - agent takes NN_CREATE_NOT_SUCCESS_PENALTY only

       nn_len: int,
       that var needed for depth decreasing reward
       # Not used (future) #

       last_train_metrcis: np.array,
       contains last training loop metrics
       for metrics_optimization_reward

       return reward: float, sum of all reward parts

    '''

    reward = 0
    # TODO reward for decreasing nn depth
    optimal_depth_reward = 0
    # reward by metrics
    metrics_optimization_reward = 0
    # reward for successfull nn creation
    creation_successfull_reward = 0
    if nn_created_correctly_flag == True:
      # do not reward agent if creation is not succeed
      creation_successfull_reward += self.NN_CREATE_SUCCESS_REWARD

      last_metrics = self.NN_PARAMS['metrics'][-1]
      last_train_metrics = np.nan_to_num(x=last_train_metrics, nan=self.NAN_NUM)
      last_train_metrics = np.minimum(last_train_metrics,
                          np.ones(shape=(self.NN_PARAMS['amount_of_metrics'], self.NN_PARAMS['train_epochs'])) * self.NAN_NUM
                          )

      tmp_r = np.min(last_metrics,axis=1) - np.min(np.array(last_train_metrics), axis=1)


      self.NN_PARAMS['metrics'] = np.roll(self.NN_PARAMS['metrics'], -1, axis=0)
      self.NN_PARAMS['metrics'][-1] = last_train_metrics
      metrics_optimization_reward = np.sum(self.METRICS_OPTIMIZATION_FACTOR * tmp_r)

      optimal_depth_reward = (self.NN_PARAMS['layers_amount'] - nn_len)
      optimal_depth_reward *= self.DEPTH_REWARD_FACTOR

    else:
      creation_successfull_reward += self.NN_CREATE_NOT_SUCCESS_PENALTY

    reward += optimal_depth_reward
    reward += metrics_optimization_reward
    reward += creation_successfull_reward

    return reward

  def get_test_batch(self):
    '''
        return: batch: np.array, one batch
        for input_shape in NN building algorithm

    '''
    batch = None
    for  b, _ in self.train_dataloader:
      batch = b
    return batch

  def create_obs(self):
    '''
       return obs: dict, new observation
    '''

    obs = {
          'last_nets_metrics_memory': self.NN_PARAMS['metrics'],
          'last_nets_architectures': self.NN_PARAMS['last_nets_architectures']

          }
    return obs

  def step(self, action):
      '''
          Main environment function

          takes action, creates NN, train NN, calc new obs and reward

          action: list, shape of action is like action_space

      return:
      obs: np.array,
      reward: float,
      done = False, end of agent training flag, needed when your actions
      achieved some finish state
      info: dict, you may need to add some extra information, put it here

      Algorithm
      1) parse action
      2) generate NN
      3) update optimizer, prepare for training
      4) train NN, collect metrics
      5) calculate reward
      6) collect statistics
      7) create new observation
      8) return obs, reward, done, info

      '''
      reward = 0
      done = False
      info = {}
      self.nngenerator.parse_action(action, self.actions_set)
      test_b = self.get_test_batch()
      success_state, net = self.nngenerator.generateNN(n_classes=self.NN_PARAMS['num_classes'], test_batch=test_b)

      new_metrics = None
      if success_state == True: # NN created_correctly
        # OUTPUT NN to server
        done = True
        self.NN_PARAMS['last_nets_architectures'] = np.roll(self.NN_PARAMS['last_nets_architectures'], -1, axis=0)
        self.NN_PARAMS['last_nets_architectures'][-1] = action
        self.Net.layers = net
        #display(self.Net)
        self.render()
        self.Net = self.Net.to(self.device)
        self.optimizer = self.optimizer_class(self.Net.parameters(), lr=self.NN_PARAMS['lr'])
        new_metrics = self.train()


      reward = self.calc_reward(success_state,
                                self.nngenerator.get_nn_len(),
                                new_metrics,
                                )


      current_obs = self.create_obs()

      self.last_obs = current_obs

      print('Reward: ', reward)
      self.episode_reward = reward
      self.current_it += 1
      return current_obs, self.episode_reward, None, done, info


  def reset(self, seed=None, options=None):
      '''
      Reset the env,
      Set all changed in training proccess variables to zero
      (or noise, dependse on your realization)

      seed: list, list of random seeds (depricated)
      options: list, additional options (future)
      '''
      super().reset(seed=seed)
      self.Net = self.NN()

      current_obs = {
          'last_nets_metrics_memory': np.zeros((
              self.NN_PARAMS['last_nets_metrics_memory_len'],
              self.NN_PARAMS['amount_of_metrics'],
              self.NN_PARAMS['train_epochs'],
             ) ),
          'last_nets_architectures': np.zeros((
             self.NN_PARAMS['last_nets_metrics_memory_len'],
             self.NN_PARAMS['layers_amount'],
          ) )

          }
      self.NN_PARAMS['metrics'] = np.zeros((
          self.NN_PARAMS['last_nets_metrics_memory_len'],
          self.NN_PARAMS['amount_of_metrics'],
          self.NN_PARAMS['train_epochs'],
      ))
      self.NN_PARAMS['last_nets_architectures'] = np.zeros((
             self.NN_PARAMS['last_nets_metrics_memory_len'],
             self.NN_PARAMS['layers_amount'],
          ))

      self.last_obs = current_obs
      self.episode_reward = 0
      self.current_it = 1

      self.statistics['episode_rewards'] = []
      self.statistics['made_steps'] = []

      return current_obs, {'none': None}



  def render(self, mode=None):
    '''
    method for visualisation your observation
    example: render method for labirint task contains some
    visualization of map, agent, finish point, etc.
    '''
    #TODO visualization
    print("in render")

    pass


  def close(self):
      '''
          environment destructor
      '''
      # TODO
      pass
    """
    for name in stage_names:
        env_class = 'class {}(gym.Env):'.format(''.join([s.capitalize() for s in name.split('_')]))
        with open('gym_{}/envs/{}.py'.format(env_name, name), 'w') as f:
            f.write(head + env_class + tail)
    pass


def make_setup(env_name):
    with open('setup.py', 'w') as f:
        tmp = [
            'from setuptools import setup\n\n',
            "setup(name='gym_{}',\n".format(env_name),
            "    version='0.0.1',\n",
            "    install_requires=['gym']  # And any other dependencies foo needs\n",
            ")\n",
        ]
        f.writelines(tmp)


if __name__ == '__main__':
    env_name = 'env'# input('Please type your env\'s name (string):')
    env_stage_num = 1 #int(input('Type the stage number of your Env (int):'))

    print(
        'Please type yout stage names\n(start with alphabet, lowercase and number, split by underscore(_), e.g. my_env_v1)')
    env_stage_names = ['env']
    #for i in range(env_stage_num):
    #    env_stage_names.append(input("Stage {}s name:".format(i)))

    make_dirs(env_name)
    make_files(env_name, env_stage_names)
    #generate_setup = input('Generate `setup.py`?[y/n]')

    generate_setup = 'y'
    if generate_setup.lower() == 'y':
        make_setup(env_name)

    from stable_baselines3 import *
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    # import gymnasium as gym
    from gymnasium.envs.registration import register

    "registering enviroment"
    register(
        id='gym_env/env_v1-v0',
        entry_point='gym_env.envs:Env',
        max_episode_steps=200,
    )