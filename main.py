# -*- codeing = utf-8 -*-
# @Time : 2022/4/23 11:31
# @Author : Evan_wyl
# @File : main.py

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import sys

sys.path.append("..")
from agent.dqn import DQN


if __name__ == '__main__':
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    done = True
    for step in range(10000):
        if done:
            state = env.reset()
        stat, reward, done, info = env.step(env.action_space.sample())
        env.render()
    env.close()
