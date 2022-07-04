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

        self.a2c_lstm = A2C_LSTM().to(device)
        self.optimizer = torch.optim.Adam(self.a2c_lstm.parameters(), lr=0.01)

        self.h_t = torch.zeros(1, 128).clone().to(device) #lstm cell의 dimension과 맞춰준다.
        self.c_t = torch.zeros(1, 128).clone().to(device) #lstm cell의 dimension과 맞춰준다.

        def lr_lambda(epoch):
            return 1 - min(epoch*32*80, self.flags.max_steps) / self.flags.max_steps
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        self.path = './agents/example7/policy.pt'
        if self.flags.mode != 'train':
            self.a2c_lstm.load_state_dict(torch.load(self.path))

    def get_action(self, env, obs):
        actor, critic = self.get_actor_critic(env, obs)
        
        action = torch.multinomial(F.softmax(actor, dim=1), num_samples=1)
        return action
    
    def get_actor_critic(self, env, obs):
        observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
        observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)

        actor, critic, self.h_t, self.c_t = self.a2c_lstm(observed_glyphs, observed_stats, self.h_t, self.c_t)
        return actor, critic
    
    def optimize_td_loss(self, actors, actions, critics, returns):
        actors = torch.cat(actors).to(device)
        actions = torch.cat(actions).to(device)        
        critics = torch.FloatTensor(critics).to(device)
        advantages = returns - critics

        #compute actor loss
        cross_entropy = F.nll_loss(
            F.log_softmax(actors, dim=-1),
            target=torch.flatten(actions),
            reduction="none",
        )
        cross_entropy = cross_entropy.view_as(advantages)
        actor_loss = torch.sum(cross_entropy * advantages.detach())

        #compute critic loss
        critic_loss = 0.5 * torch.sum(advantages**2)

        #compute entropy loss
        policy = F.softmax(actors, dim=-1)
        log_policy = F.log_softmax(actors, dim=-1)
        entropy_loss = torch.sum(policy * log_policy)
        
        loss = actor_loss + critic_loss + entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.a2c_lstm.parameters(), 40.0)
        self.optimizer.step()
        self.scheduler.step()

    def train(self):
        env = self.env
        
        num_episodes = 0
        episode_scores = deque([], maxlen=100)
        episode_dungeonlv = deque([], maxlen=100)
        episode_explv = deque([], maxlen=100)
        episode_steps = deque([], maxlen=100)

        time_step = 0
        max_steps_per_episode = 32*80

        obs = env.reset()
        while time_step < self.flags.max_steps:
            actors, actions, critics, rewards, dones = [], [], [], [], []

            for mini_step in range(max_steps_per_episode):
                old_score = obs['blstats'][9]
                old_dlv = obs['blstats'][12]
                old_elv = obs['blstats'][18]
                old_steps = obs['blstats'][20]

                action = self.get_action(env, obs)
                actor, critic = self.get_actor_critic(env, obs)

                new_obs, reward, done, info = env.step(action)
                
                actors.append(actor)
                actions.append(action)
                critics.append(critic.squeeze())
                rewards.append(np.tanh(reward/100))
                dones.append(done)

                if done:
                    num_episodes += 1
                    episode_scores.append(old_score)
                    episode_dungeonlv.append(old_dlv)
                    episode_explv.append(old_elv)
                    episode_steps.append(old_steps)

                    obs = env.reset()
                else:                
                    obs = new_obs

                if mini_step == max_steps_per_episode-1:
                    time_step += max_steps_per_episode

                    with torch.no_grad():
                        _, new_critic = self.get_actor_critic(env, new_obs)

                        returns = []
                        r = new_critic
                        for t in reversed(range(len(rewards))):
                            r = rewards[t] + self.gamma * r * (1.0 - dones[t])
                            returns.insert(0, r)
                        returns = torch.FloatTensor(returns).to(device)

                    self.optimize_td_loss(actors, actions, critics, returns)

                    print("Elapsed Steps: {}%".format((time_step)/self.flags.max_steps*100))
                    print("Episodes: {}".format(num_episodes))
                    print("Last 100 Episode Mean Score: {}".format(sum(episode_scores)/len(episode_scores) if episode_scores else 0))
                    print("Last 100 Episode Mean Dungeon Lv: {}".format(sum(episode_dungeonlv)/len(episode_dungeonlv) if episode_dungeonlv else 0))
                    print("Last 100 Episode Mean Exp Lv: {}".format(sum(episode_explv)/len(episode_explv) if episode_explv else 0))
                    print("Last 100 Episode Mean Step: {}".format(sum(episode_steps)/len(episode_steps) if episode_steps else 0))
                    print()
                    
                    writer.add_scalar('Last 100 Episode Mean Score', sum(episode_scores)/len(episode_scores) if episode_scores else 0, time_step+1)
                    writer.add_scalar('Last 100 Episode Mean Dungeon Lv', sum(episode_dungeonlv)/len(episode_dungeonlv) if episode_dungeonlv else 0, time_step+1)
                    writer.add_scalar('Last 100 Episode Mean Exp Lv', sum(episode_explv)/len(episode_explv) if episode_explv else 0, time_step+1)
                    writer.add_scalar('Last 100 Episode Mean Step', sum(episode_steps)/len(episode_steps) if episode_steps else 0, time_step+1)
                    
                    torch.save(self.a2c_lstm.state_dict(), self.path)