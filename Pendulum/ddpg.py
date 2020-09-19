#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:26:31 2020

@author: arunansupatra
"""

import gym
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype = bool)
        
    def store_transition(self, state, action, reward, n_state, done):
        index = self.mem_ctr % self.mem_size
        
        self.state_memory[index] = state
        self.new_state_memory[index] = n_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        self.mem_ctr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        n_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, n_states, dones

class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=512, fc2_dims = 512,
                 name='critic', chkpt_dir = 'tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name+'_ddpg.h5')
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)
        
    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        
        q = self.q(action_value)
    
        return q

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=512, fc2_dims = 512,
                 name='actor', chkpt_dir = 'tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name+'_ddpg.h5')
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')
        
    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob)
        
        return mu

class Agent:
    def __init__(self, alpha=0.001, beta=0.002, input_dims = [8], env=None,
                 gamma=0.99, n_actions=2, max_size=1000000, tau=0.005, 
                 fc1=512, fc2=512, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        
        self.actor = ActorNetwork(n_actions = n_actions, name='actor')
        self.critic = CriticNetwork(n_actions = n_actions, name='critic')
        self.target_actor = ActorNetwork(n_actions = n_actions, name='actor')
        self.target_critic = CriticNetwork(n_actions = n_actions, name='critic')
        
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=alpha))

        self.update_network_parameters(tau=1)
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)
        
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def save_model(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)
        
    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)
        
    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype = tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],
                                        mean=0.0, stddev=0.1)
        #note that if env has action >1 we have to multiply by
        #max action at some point
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        
        return actions[0]
    
    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        
        state, action, reward, next_state, done =  self.memory.sample_buffer(self.batch_size)
        
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        n_states = tf.convert_to_tensor(next_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(n_states)
            critic_value_ = tf.squeeze(self.target_critic(n_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions),1)
            target = reward + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)
        
        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
        
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))
        
        self.update_network_parameters()
        

env = gym.make('Pendulum-v0')
agent = Agent(input_dims=env.observation_space.shape, env=env,
              n_actions = env.action_space.shape[0])
n_games = 250

figure_file = 'plots/pendulum.png'

best_score = env.reward_range[0]
score_history = []
load_checkpoint = False

if load_checkpoint:
    n_steps = 0
    while n_steps <=agent.batch_size:
        observation = env.reset()
        action = env.action_space.sample()
        observation_, reward, done, info = env.step(action)
        agent.remember(observation, action, reward, observation_, done)
        n_steps += 1
    agent.learn()
    agent.load_models()
    evaluate = True
else:
    evaluate = False

for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation, evaluate)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.remember(observation, action, reward, observation_, done)
        if not load_checkpoint:
            agent.learn()
        observation = observation_
        
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    
    if avg_score > best_score:
        best_score = avg_score
        if not load_checkpoint:
            agent.save_model()
    
    print('episode: ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
    
if not load_checkpoint:
    x = [i+1 for i in range(n_games)]
    plt.plot(x, score_history)
    plt.title('Pendulum score at every iteration')
    plt.xlabel('iterations')
    plt.ylabel('score')
    plt.savefig(figure_file)

