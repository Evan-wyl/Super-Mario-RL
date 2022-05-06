# -*- codeing = utf-8 -*-
# @Time : 2022/4/23 12:10
# @Author : Evan_wyl
# @File : CONSTANT.py

# AGENT = "DQN"
AGENT = "DDQN"
# AGENT = "DUELING_DQN"
# AGENT = "DUELING_DDQN"

MAX_LEN = 25000
BACH_SIZE = 32
LEARNING_RATE = 0.00025
# LEARNING_RATE = 6.25e-5 # Dueling_DDQN
EPISODES = 40000

GAMMA = 0.9
EXPLORATION_RATE = 1
EXPLORATION_RATE_DECAY = 0.99999975
EXPLORATION_RATE_MIN = 0.1

SAVE_EVERY = 5e5
BURNIN = 1e4  # min. experience before training
LEARN_EVERY = 3 # no. of experiences between updates to Q_online
SYNC_EVERY = 1e4 # no. of experiences between Q_target & Q_online sync