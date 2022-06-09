import random
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import math

from ExampleAgent import ExampleAgent
from .a2c import A2C

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class Agent(ExampleAgent):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

        self.gamma = 0.95
        self.actor_loss_coef = 1.0
        self.critic_loss_coef = 0.5
        self.entropy_loss_coef = 0.01

        self.a2c = A2C()
        self.optimizer = torch.optim.Adam(self.a2c.parameters())
        
        self.path = './agents/example6/policy.pt'
        if self.flags.mode != 'train':
            self.a2c.load_state_dict(torch.load(self.path))
    
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
        actor, critic = self.get_actor_critic(env, obs)
        
        action = actor.sample().unsqueeze(1)
        return action
    
    def get_actor_critic(self, env, obs):
        pre_map = self.preprocess_map(obs)
        pre_stat = self.preprocess_stat(obs)

        pre_map = torch.from_numpy(pre_map).float().unsqueeze(0)
        pre_stat = torch.from_numpy(pre_stat).float().unsqueeze(0)

        actor, critic = self.a2c(pre_map, pre_stat)
        return actor, critic
    
    def optimize_td_loss(self, log_probs, critics, entropies, returns):        
        log_probs = torch.cat(log_probs)
        critics = torch.FloatTensor(critics)
        entropies = torch.cat(entropies)
        
        advantages = returns - critics
        
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = entropies.mean()
        
        loss = self.actor_loss_coef * actor_loss + self.critic_loss_coef * critic_loss - self.entropy_loss_coef * entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.a2c.parameters(), 1.0)
        self.optimizer.step()

        return loss

    def train(self, env):
        episode_rewards = [0.0]
        average_rewards = []
        log_probs, critics, rewards, dones, entropies = [], [], [], [], []

        obs = env.reset()
        for time_step in range(0, self.flags.max_steps):       
            action = self.get_action(env, obs)
            actor, critic = self.get_actor_critic(env, obs)

            new_obs, reward, done, info = env.step(action)

            episode_rewards[-1] += reward

            log_prob = actor.log_prob(action.squeeze(1))
            entropy = actor.entropy()
            
            log_probs.append(log_prob)
            critics.append(critic.squeeze())
            rewards.append(reward)
            dones.append(done)
            entropies.append(entropy)

            if done:
                obs = env.reset()
            else:                
                obs = new_obs
                continue
            
            with torch.no_grad():
                _, new_critic = self.get_actor_critic(env, new_obs)

                returns = []
                for t in reversed(range(len(rewards))):
                    r = rewards[t] + self.gamma * new_critic * (1.0 - dones[t])
                    returns.insert(0, r)
                returns = torch.FloatTensor(returns)

            loss = self.optimize_td_loss(log_probs, critics, entropies, returns)

            if done:
                num_episodes = len(episode_rewards)
                print("********************************************************")
                print("Loss: {}".format(loss))
                print("Total Steps: {}".format(time_step))
                print("Episodes: {}".format(num_episodes))
                print("Reward: {}".format(episode_rewards[-1]))
                print("********************************************************")
                writer.add_scalar('loss/train', loss, num_episodes)
                writer.add_scalar('reward/train', episode_rewards[-1], num_episodes)
                episode_rewards.append(0.0)
                log_probs, critics, rewards, dones, entropies = [], [], [], [], []
                
                torch.save(self.a2c.state_dict(), self.path)