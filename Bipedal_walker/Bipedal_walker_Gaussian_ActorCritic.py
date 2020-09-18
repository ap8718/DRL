#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 22:55:55 2020

@author: arunansupatra
"""


'''
use seperate network for each action
'''

import gym
import numpy as np
import matplotlib.pyplot as plt

class Network():
    def __init__(self, network_config):
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")
        self.actor_layer_sizes = np.array([self.state_dim, self.num_hidden_units, 2*self.num_actions])
        self.critic_layer_sizes = np.array([self.state_dim, self.num_hidden_units, self.num_actions])
        
        self.rand_generator = np.random.RandomState(network_config.get("seed"))
        
        self.actor_net = [dict() for i in range(0, len(self.actor_layer_sizes) - 1)]
        self.critic_net = [dict() for i in range(0, len(self.critic_layer_sizes) - 1)]
        
        for i in range(0, len(self.actor_layer_sizes) - 1):
            self.actor_net[i]['W'] = self.init_saxe(self.actor_layer_sizes[i], self.actor_layer_sizes[i + 1])
            self.actor_net[i]['b'] = np.zeros((1, self.actor_layer_sizes[i + 1]))
        
        for i in range(0, len(self.critic_layer_sizes) - 1):
            self.critic_net[i]['W'] = self.init_saxe(self.critic_layer_sizes[i], self.critic_layer_sizes[i + 1])
            self.critic_net[i]['b'] = np.zeros((1, self.critic_layer_sizes[i + 1]))
        
        self.actor_alpha = 0.0001
        self.critic_alpha = 0.0005
    
    def init_saxe(self, rows, cols):
        tensor = self.rand_generator.normal(0, 1, (rows, cols))
        if rows < cols:
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph
        
        if rows < cols:
            tensor = tensor.T
        return tensor
    
    def get_actor_mu_sig(self, s):
        W0, b0 = self.actor_net[0]['W'], self.actor_net[0]['b']
        psi_m = np.dot(s, W0) + b0
        x = np.maximum(0, psi_m)
        W1, b1 = self.actor_net[1]['W'], self.actor_net[1]['b']
        mu_sig = np.dot(x, W1) + b1
        # np.nan_to_num(mu_sig, copy=False, neginf=0.0)
        mu_sig[:, self.num_actions:] = np.exp(mu_sig[:, self.num_actions:])
        print("mu_sig is:")
        print(mu_sig)
        print('\n')
        
        return mu_sig
    
    def get_critic_value(self, s):
        W0, b0 = self.critic_net[0]['W'], self.critic_net[0]['b']
        psi_m = np.dot(s, W0) + b0
        x = np.maximum(0, psi_m)
        W1, b1 = self.critic_net[1]['W'], self.critic_net[1]['b']
        v = np.dot(x, W1) + b1
        
        return v
    
    def update_actor_weights(self, delta, s, actor_mu_grad, actor_sigma_grad):
        actor_grad = np.concatenate((actor_mu_grad, actor_sigma_grad))
        actor_grad = actor_grad.reshape((1, len(actor_grad)))
        # np.nan_to_num(actor_grad, copy=False)
        print('actor_grad is:')
        print(actor_grad)
        print('\n')
        delta = delta[0]
        W0, b0 = self.actor_net[0]['W'], self.actor_net[0]['b']
        psi_m = np.dot(s, W0) + b0
        x = np.maximum(0, psi_m)
        x = x.reshape((len(x[0]),1))
        s = s.reshape((len(s[0]),1))
        for i in delta:
            self.actor_net[1]['W'] += self.actor_alpha * i * actor_grad * x
            self.actor_net[0]['W'] += self.actor_alpha * i * s
        # if np.all(self.actor_net[1]['W'] == float('nan')):
        #     self.actor_net[1]['W'] = self.init_saxe(self.actor_layer_sizes[1], self.actor_layer_sizes[2])
        
        
    def update_critic_weights(self, delta, s):
        W0, b0 = self.actor_net[0]['W'], self.actor_net[0]['b']
        psi_m = np.dot(s, W0) + b0
        x = np.maximum(0, psi_m)
        x = x.reshape((len(x[0]),))
        s = s.reshape((len(s[0]),))
        
        delta = delta[0]
        for j in range(self.num_actions):
            for i in delta:
                self.critic_net[1]['W'][:,j] += self.critic_alpha * i * x
                self.critic_net[0]['W'][:,j] += self.critic_alpha * i * s
        
class Agent():
    
    def __init__(self, agent_config):
        self.network = Network(agent_config['network_config'])
        self.num_actions = agent_config['network_config']['num_actions']
        self.discount = agent_config['gamma']
        self.rand_generator = np.random.RandomState(agent_config.get("seed"))
        
        self.last_state = None
        self.last_action = None
        
        self.sum_rewards = 0.
        self.avg_reward = 0.
        self.avg_reward_alpha = 2**(-6)
        self.episode_steps = 0
    
    def policy(self, state):
        mu_sig = self.network.get_actor_mu_sig(state)
        mu_sig = np.reshape(mu_sig, (2*self.num_actions, 1))
        mu_vec = mu_sig[:self.network.num_actions]
        sigma_vec = mu_sig[self.network.num_actions:]
        actions = np.zeros((self.network.num_actions))
        for i in range(len(actions)):
            actions[i] = self.rand_generator.normal(mu_vec[i], sigma_vec[i])
        # np.nan_to_num(actions, copy=False)
        return actions
    
    def start(self, state):
        self.sum_rewards = 0.
        self.avg_reward = 0.
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state)
        self.last_action = self.last_action.reshape((self.network.num_actions, 1))
        return self.last_action
    
    def step(self, reward, state):
        self.episode_steps += 1
        self.sum_rewards += reward
        
        print("reward is:")
        print(reward)
        print('\n')
        
        delta = reward - self.avg_reward + self.network.get_critic_value(state) - self.network.get_critic_value(self.last_state)
        
        print("delta is:")
        print(delta)
        print('\n')
        
        self.avg_reward += self.avg_reward_alpha * delta
        
        self.network.update_critic_weights(delta, self.last_state)
        
        mu_sig = self.network.get_actor_mu_sig(state) #prints mu_sig
        mu_sig = np.reshape(mu_sig, (2*self.num_actions, 1))
        mu_vec = mu_sig[:self.network.num_actions]
        sigma_vec = mu_sig[self.network.num_actions:]
        # print("sigma_vec is:")
        # print(sigma_vec)
        # print('\n')
        # np.nan_to_num(self.last_action, copy=False)
        mu_grad = np.tanh(((1/sigma_vec)**2) * (self.last_action - mu_vec))
        sigma_grad = np.tanh(((1/sigma_vec)**2) * ((self.last_action - mu_vec)**2) - 1)
        
        self.network.update_actor_weights(delta, self.last_state, mu_grad, sigma_grad) #prints actor_grad
        
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state) #prints mu_sig
        self.last_action = self.last_action.reshape((self.network.num_actions, 1))
        print("last_action is:")
        print(self.last_action)
        print('\n')
        
        return self.last_action
    
    def end(self, reward):
        self.episode_steps += 1
        self.sum_rewards += reward
        
        state = np.zeros_like(self.last_state)
        
        delta = reward - self.avg_reward + self.network.get_critic_value(state) - self.network.get_critic_value(self.last_state)
        
        self.avg_reward += self.avg_reward_alpha * delta
        
        self.network.update_critic_weights(delta, self.last_state)
        
        mu_sig = self.network.get_actor_mu_sig(state)
        mu_sig = np.reshape(mu_sig, (2*self.num_actions, 1))
        mu_vec = mu_sig[:self.network.num_actions]
        sigma_vec = mu_sig[self.network.num_actions:]
        
        
        mu_grad = ((1/sigma_vec)**2) * (self.last_action - mu_vec)
        sigma_grad = ((1/sigma_vec)**2) * ((self.last_action - mu_vec)**2) - 1
        
        self.network.update_actor_weights(delta, self.last_state, mu_grad, sigma_grad)

env = gym.make('BipedalWalker-v3')
state = env.reset()

agent_parameters = {
    'network_config': {
        'state_dim': 24,
        'num_hidden_units': 256,
        'num_actions': 4
    },
    'optimizer_config': {
        'step_size': 1e-3,
        'beta_m': 0.9, 
        'beta_v': 0.999,
        'epsilon': 1e-8
    },
    'replay_buffer_size': 50000,
    'minibatch_sz': 8,
    'num_replay_updates_per_step': 4,
    'gamma': 0.99,
    'tau': 0.001
}

agent = Agent(agent_parameters)
state = env.reset()

action = agent.start(state)
state, reward, done, l = env.step(action)
r_sum = 0
rewards = []
ep = 0
loops = 0
while ep < 500:
    # env.render()
    loops += 1
    print("loop number:")
    print(loops)
    print('\n')
    action = agent.step(reward, state)
    state, reward, done, l = env.step(action)
    r_sum += reward
    if done:
        agent.end(reward)
        state = env.reset()
        agent.start(state)
        rewards.append(r_sum)
        r_sum = 0
        ep += 1
        print('_______________________________')
        print("ep number:")
        print(ep)
        print('_______________________________')
        print('\n')
env.close()
plt.figure()
plt.plot(rewards)
plt.title('Total rewards per episode')
plt.savefig('rewards.png')