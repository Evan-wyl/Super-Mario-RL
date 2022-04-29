# -*- codeing = utf-8 -*-
# @Time : 2022/4/22 20:15
# @Author : Evan_wyl
# @File : dqn.py

import pandas as pd
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


# -*- codeing = utf-8 -*-
# @Time : 2022/4/23 21:32
# @Author : Evan_wyl
# @File : ddqn.py

import torch
from torch import nn
from torch import functional as F
from torch import optim

import numpy as np
import pandas as pd
from collections import deque
import copy
import random

import sys
sys.path.append("..")
from params.CONSTANT import MAX_LEN
from params.CONSTANT import BACH_SIZE
from params.CONSTANT import EXPLORATION_RATE
from params.CONSTANT import EXPLORATION_RATE_MIN
from params.CONSTANT import EXPLORATION_RATE_DECAY
from params.CONSTANT import SYNC_EVERY
from params.CONSTANT import BURNIN
from params.CONSTANT import LEARN_EVERY
from params.CONSTANT import GAMMA
from params.CONSTANT import LEARNING_RATE


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, get:{h}")
        if w != 84:
            raise ValueError(f"Expecting input wight: 84, get:{w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c,  out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,  stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


class Mario(object):
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        self.net = DQN(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = EXPLORATION_RATE
        self.exploration_rate_decay = EXPLORATION_RATE_DECAY
        self.exploration_rate_min = EXPLORATION_RATE_MIN
        self.curr_step = 0

        self.save_every = 5e5

        self.memory = deque(maxlen=MAX_LEN)
        self.batch_size = BACH_SIZE

        self.gamma = GAMMA

        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()

        self.burnin = 1e4
        self.learn_every = 3
        self.sync_every = 1e4

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward,  done):
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor(reward)
            done = torch.tensor(done)

        self.memory.append((state, next_state,  action, reward, done))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        # stack:Concatenates a sequence of tensors along a new dimension.
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="target")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate = self.exploration_rate), save_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)

        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

