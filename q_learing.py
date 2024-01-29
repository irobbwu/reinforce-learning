import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.envs.toy_text import frozen_lake
from torch.utils.tensorboard import SummaryWriter
import collections
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--episode', type=int, default=None)
args = parser.parse_args()

class Agent:
    def __init__(self, env):
        self.env = env
        self.state, info = self.env.reset()

        # 状态，行为，是否终止保存
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.terminates = collections.defaultdict(bool)

        # q_value,state_value保存
        self.q_table = collections.defaultdict(float)
        self.values = collections.defaultdict(float)

    ## 随机模拟实验
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, terminited, truncted, info = self.env.step(action)
            # print(self.state, action, new_state)
            self.terminates[(self.state, action, new_state)] = terminited
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            if terminited:
                self.state, info = self.env.reset()
            else:
                self.state = new_state

    ## 计算 行动值
    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            val = reward + GAMMA * self.values[tgt_state]
            action_value += (count / total) * val
        return action_value

    def q_table_update(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                self.q_table[(state, action)] = self.calc_action_value(state, action)

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            self.values[state] = max(
                [
                    self.q_table[(state, action)]
                    for action in range(self.env.action_space.n)
                ]
            )

    def select_action(self, state):
        return int(
            np.argmax(
                [
                    self.q_table[(state, action)]
                    for action in range(self.env.action_space.n)
                ]
            )
        )

    def episode(self, n):
        # 初始状态
        self.play_n_random_steps(1000)
        env = self.env

        # 可视化
        env.render_mode = "human"
        display_size = 512
        env.window_size = (display_size,display_size)
        env.cell_size = (
                    env.window_size[0] // env.ncol,
                    env.window_size[1] // env.nrow,
                )
        env = gym.wrappers.RecordVideo(env, video_folder="video")
        
        state, info = self.env.reset()
        total_rewards = 0
        
        for e in range(n):
            # print(state)

            # 计算q_value
            agent.q_table_update()

            # 选择最优策略
            action = self.select_action(state)
            new_state, reward, terminated, truncted, info = env.step(action)

            total_rewards+=reward
            
            # 更新 state
            state = new_state
            
            # 更新 v
            self.value_iteration()

            if terminated:
                state, info = self.env.reset()
        return total_rewards


if __name__ == "__main__":
    ENV_NAME = "FrozenLake-v1"
    GAMMA = 0.8
    
    env = frozen_lake.FrozenLakeEnv(is_slippery=False)
    env.spec = gym.spec("FrozenLake-v1")
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    # env = gym.wrappers.RecordVideo(env, video_folder="video")
    
    agent = Agent(env)
    agent.episode(args.episode)
    env.close()