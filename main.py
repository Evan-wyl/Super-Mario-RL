# -*- codeing = utf-8 -*-
# @Time : 2022/4/23 11:31
# @Author : Evan_wyl
# @File : main.py

import numpy as np
from pathlib import Path
import datetime

import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


import sys
sys.path.append("..")
from agent.ddqn import Mario
# from agent.dqn import Mario
# from agent.dueling_DDQN import Mario
# from agent.dueling_DQN import Mario
from metricLogger import MetricLogger
from params.CONSTANT import EPISODES, AGENT

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super(SkipFrame, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward,  done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(GrayScaleObservation, self).__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape,  dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


if __name__ == '__main__':
    env = gym_super_mario_bros.make("SuperMarioBros-v0")

    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    #
    # done = True
    # for step in range(10000):
    #     if done:
    #         state = env.reset()
    #     state, reward, done, info = env.step(env.action_space.sample())
    #     env.render()
    # env.close()

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA:  {use_cuda}")
    print()

    save_dir = Path(f"checkpoints/{AGENT}") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    # episodes = 10
    for e in range(EPISODES):
        state = env.reset()

        while True:
            action = mario.act(state)

            next_state, reward, done, info = env.step(action)

            mario.cache(state, next_state, action, reward, done)

            q, loss = mario.learn()

            logger.log_step(reward, loss, q)

            state = next_state

            env.render()

            if done or info["flag_get"]:
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.cur_step)

