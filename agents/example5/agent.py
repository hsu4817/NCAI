import random
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import math

from ExampleAgent import ExampleAgent
from .dqn import ReplayBuffer, DQN

class Agent(ExampleAgent):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

        self.buffer_size = 5000
        self.batch_size = 64
        self.gamma = 0.99

        self.policy = DQN()
        self.target = DQN()
        self.buffer = ReplayBuffer(self.buffer_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters())
        
        self.path = './agents/example5/policy.pt'
        if self.flags.mode != 'train':
            self.policy.load_state_dict(torch.load(self.path))
    
    def preprocess_map(self, obs):
        pre = []

        available = [ord('.'), ord('#')]
        unavailable = [ord(' '), ord('`')]
        door_or_wall = [ord('|'), ord('-')]

        chars = obs['chars']
        colors = obs['colors']
        for y in range(21):
            pre_line = []
            for x in range(79):
                char = chars[y][x]
                color = colors[y][x]
                
                pre_char = 1.0
                if char in unavailable:
                    pre_char = 0.0
                elif char in door_or_wall and color == 7:
                    pre_char = 0.0
                elif char == ord('#') and color == 6:
                    pre_char = 0.0
                pre_line.append(pre_char)
            pre.append(pre_line)
        
        return np.array(pre).astype(np.float32)
    
    def preprocess_stat(self, obs):
        blstats = obs['blstats']
        x = blstats[0]
        y = blstats[1]
        hp = float(blstats[10])

        return np.array([x, y, hp,]).astype(np.float32)

    def get_action(self, env, obs):
        pre_map = self.preprocess_map(obs)
        pre_stat = self.preprocess_stat(obs)

        pre_map = torch.from_numpy(pre_map).float().unsqueeze(0)
        pre_stat = torch.from_numpy(pre_stat).float().unsqueeze(0)

        with torch.no_grad():
            q = self.policy(pre_map, pre_stat)
        
        _, action = q.max(1)
        return action.item()
    
    def optimize_td_loss(self):
        glyphs, stats, next_glyphs, next_stats, actions, rewards, dones = self.buffer.sample(self.batch_size)
        glyphs = torch.from_numpy(glyphs).float()
        stats = torch.from_numpy(stats).float()
        next_glyphs = torch.from_numpy(next_glyphs).float()
        next_stats = torch.from_numpy(next_stats).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        dones = torch.from_numpy(dones).float()

        with torch.no_grad():
            q_next = self.policy(next_glyphs, next_stats)
            _, action_next = q_next.max(1)
            q_next_max = self.target(next_glyphs, next_stats)
            q_next_max = q_next_max.gather(1, action_next.unsqueeze(1)).squeeze()
        
        q_target = rewards + (1 - dones) * self.gamma * q_next_max
        q_curr = self.policy(glyphs, stats)
        q_curr = q_curr.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(q_curr, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, env):
        episode_rewards = [0.0]
        average_rewards = []
        losses = []

        obs = env.reset()
        for time_step in range(0, self.flags.max_steps):
            pre_map = self.preprocess_map(obs)
            pre_stat = self.preprocess_stat(obs)
            
            eps_threshold = math.exp(-len(episode_rewards)/50)
            if random.random() <= eps_threshold:
                action = env.action_space.sample()
            else:
                action = self.get_action(env, obs)

            new_obs, reward, done, info = env.step(action)

            episode_rewards[-1] += reward

            if done:
                obs = env.reset()
            else:
                new_pre_map = self.preprocess_map(new_obs)
                new_pre_stat = self.preprocess_stat(new_obs)
                self.buffer.push(pre_map,
                                pre_stat,
                                new_pre_map,
                                new_pre_stat,
                                action,
                                reward,
                                float(done))
                obs = new_obs
            
            if time_step > 1000:
                loss = self.optimize_td_loss()
                losses.append(loss)

            if time_step > 1000 and time_step % 1000 == 0:
                self.target.load_state_dict(self.policy.state_dict())
                torch.save(self.policy.state_dict(), self.path)
            
            if done:
                num_episodes = len(episode_rewards)
                print("********************************************************")
                print("Average loss: {}".format(np.array(losses).mean()))
                print("Total Steps: {}".format(time_step))
                print("Episodes: {}".format(num_episodes))
                print("Reward: {}".format(episode_rewards[-1]))
                print("********************************************************")
                episode_rewards.append(0.0)
                losses = []