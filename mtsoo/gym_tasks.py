import gym
from gym.spaces.discrete import Discrete
import longdpole
import numpy as np


class GymTaskSet():
    def __init__(self, names):
        self.names = names
        self.envs = [gym.make(name) for name in self.names]

    def run_env(self, sf, agent):
        env = self.envs[sf]
        env.seed(0)
        obs = env.reset()
        total_reward = 0
        done = False
        for i in range(100000):
            action = agent.get_action(obs.astype(np.float32))
            if type(env.action_space) == Discrete:
                no_act = env.action_space.n
                act = np.argmax(action[:no_act])
            else:
                act = action[0]
            obs, reward, done, _ = env.step(act)
            total_reward += reward
            if done:
                break
        return -total_reward
