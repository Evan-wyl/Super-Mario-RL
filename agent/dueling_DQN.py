# -*- codeing = utf-8 -*-
# @Time : 2022/4/25 10:33
# @Author : Evan_wyl
# @File : dueling_DQN.py

import torch
from torch import nn
from torch import optim
from torch import functional as F

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
from params.CONSTANT import LEARNING_RATE
from params.CONSTANT import GAMMA
from params.CONSTANT import LEARNING_RATE


class DuelingNetWork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingNetWork, self).__init__()

        c, h, w = input_dim
        if h != 84:
            raise ValueError(f"Expecting input height: 84, get:{h}")
        if w != 84:
            raise ValueError(f"Expecting input weight: 84, get:{w}")

        self.cov1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.cov2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.cov3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.l0_a = nn.Linear(3136, 512)
        self.l1_a = nn.Linear(512, output_dim)

        self.l0_v = nn.Linear(3136, 512)
        self.l1_v = nn.Linear(512, 1)

    def forward(self, input):
        x = self.cov1(input)
        x = nn.ReLU(x)
        x = self.cov2(x)
        x = nn.ReLU(x)
        x = self.cov3(x)
        x = nn.ReLU(x)
        x = x.Flatten()

        a = self.l0_a(x)
        a = nn.ReLU(a)
        a = self.l1_a(a)

        v = self.l0_v(x)
        v = nn.ReLU(v)
        v = self.l1_v(v)
        return a + v - a.mean()


class Mario(object):
    def __init__(self, state_dim, action_dim, save_dir):
        super(Mario, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        self.net = DuelingNetWork(self.state_dim, self.action_dim)
        self.target = copy.deepcopy(self.net)
        for p in self.target.parameters():
            p.requires_grad = False
        if self.use_cuda:
            self.net = self.net.to(device="cuda")
            self.target = self.target.to(device="cuda")

        self.exploration_rate = EXPLORATION_RATE
        self.exploration_rate_decay = EXPLORATION_RATE_DECAY
        self.exploration_rate_min = EXPLORATION_RATE_MIN
        self.cur_step = 0

        self.save_every = 5e5

        self.memory = deque(maxlen=MAX_LEN)
        self.batch_size= BACH_SIZE

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
            action_values = self.net(state)
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.cur_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        state = state.__array__
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
            reward = torch.tensor([reward])
            done = torch.tensor([done])
        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.unsqueeze(), reward.unsqueeze(), done.unsqueeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state)[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # next_state_Q = self.net(next_state)
        next_state_Q = self.target(next_state)
        nex_Q = torch.max(next_state_Q, dim=1)
        # best_action = torch.argmax(next_state_Q, axis=1)
        # nex_Q = self.target(next_state)[
        #     np.arange(0, self.batch_size), best_action
        # ]
        return (reward + (1 - done.float()) * self.gamma * nex_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def syn_Q_target(self):
        self.target.load_state_dict(self.net.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"mario_net{int(self.cur_step / self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path)
        print(f"MarioNet saved to {save_path} at step {self.cur_step}")

    def learn(self):
        if self.cur_step % self.sync_every == 0:
            self.syn_Q_target()

        if self.cur_step % self.save_every == 0:
            self.save()

        if self.cur_step < self.burnin:
            return None, None

        if self.cur_step % self.learn_every == 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
