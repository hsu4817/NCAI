import random
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import math

from ExampleAgent import ExampleAgent
from .a2c_lstm import A2C_LSTM
from collections import deque

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(ExampleAgent):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

        self.gamma = 0.999
        self.actor_loss_coef = 1.0
        self.critic_loss_coef = 0.5
        self.entropy_loss_coef = 0.01

        self.a2c_lstm = A2C_LSTM().to(device)
        self.optimizer = torch.optim.Adam(self.a2c_lstm.parameters())

        self.h_t = torch.zeros(1, 128).clone().to(device) #lstm cell의 dimension과 맞춰준다.
        self.c_t = torch.zeros(1, 128).clone().to(device) #lstm cell의 dimension과 맞춰준다.
        
        self.path = './agents/example7/policy.pt'
        if self.flags.mode != 'train':
            self.a2c_lstm.load_state_dict(torch.load(self.path))

    def get_action(self, env, obs):
        actor, critic = self.get_actor_critic(env, obs)
        
        action = actor.sample().unsqueeze(1)
        return action
    
    def get_actor_critic(self, env, obs):
        observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
        observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)

        actor, critic, self.h_t, self.c_t = self.a2c_lstm(observed_glyphs, observed_stats, self.h_t, self.c_t)
        return actor, critic
    
    def optimize_td_loss(self, log_probs, critics, entropies, returns):        
        log_probs = torch.cat(log_probs).to(device)
        critics = torch.FloatTensor(critics).to(device)
        entropies = torch.cat(entropies).to(device)
        
        advantages = returns - critics
        
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = entropies.mean()
        
        loss = self.actor_loss_coef * actor_loss + self.critic_loss_coef * critic_loss - self.entropy_loss_coef * entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.a2c_lstm.parameters(), 40.0)
        self.optimizer.step()

    def train(self):
        env = self.env
        
        num_episodes = 0
        episode_scores = deque([], maxlen=100)
        episode_dungeonlv = deque([], maxlen=100)
        episode_explv = deque([], maxlen=100)
        episode_steps = deque([], maxlen=100)

        log_probs, critics, rewards, dones, entropies = [], [], [], [], []
        episode_reward = 0

        obs = env.reset()
        for time_step in range(self.flags.max_steps):
            old_score = obs['blstats'][9]
            old_dlv = obs['blstats'][12]
            old_elv = obs['blstats'][18]
            old_steps = obs['blstats'][20]

            action = self.get_action(env, obs)
            actor, critic = self.get_actor_critic(env, obs)

            new_obs, reward, done, info = env.step(action)

            log_prob = actor.log_prob(action.squeeze(1))
            entropy = actor.entropy()
            
            log_probs.append(log_prob)
            critics.append(critic.squeeze())
            rewards.append(np.tanh(reward/100))
            dones.append(done)
            entropies.append(entropy)

            if done:
                num_episodes += 1
                episode_scores.append(old_score)
                episode_dungeonlv.append(old_dlv)
                episode_explv.append(old_elv)
                episode_steps.append(old_steps)

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
                returns = torch.FloatTensor(returns).to(device)

            self.optimize_td_loss(log_probs, critics, entropies, returns)

            if done:
                print("Elapsed Steps: {}%".format((time_step+1)/self.flags.max_steps*100))
                print("Episodes: {}".format(num_episodes))
                print("Last 100 Episode Mean Score: {}".format(sum(episode_scores)/len(episode_scores)))
                print("Last 100 Episode Mean Dungeon Lv: {}".format(sum(episode_dungeonlv)/len(episode_dungeonlv)))
                print("Last 100 Episode Mean Exp Lv: {}".format(sum(episode_explv)/len(episode_explv)))
                print("Last 100 Episode Mean Step: {}".format(sum(episode_steps)/len(episode_steps)))
                print()
                
                writer.add_scalar('Last 100 Episode Mean Score', sum(episode_scores)/len(episode_scores), time_step+1)
                writer.add_scalar('Last 100 Episode Mean Dungeon Lv', sum(episode_dungeonlv)/len(episode_dungeonlv), time_step+1)
                writer.add_scalar('Last 100 Episode Mean Exp Lv', sum(episode_explv)/len(episode_explv), time_step+1)
                writer.add_scalar('Last 100 Episode Mean Step', sum(episode_steps)/len(episode_steps), time_step+1)

                log_probs, critics, rewards, dones, entropies = [], [], [], [], []
                
                torch.save(self.a2c_lstm.state_dict(), self.path)