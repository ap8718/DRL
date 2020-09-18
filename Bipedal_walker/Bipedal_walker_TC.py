#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 21:16:04 2020

@author: arunansupatra
"""

import numpy as np
import gym
import tiles3 as tc
import matplotlib.pyplot as plt

class PendulumTileCoder:
    def __init__(self, iht_size=4096, num_tilings=32, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the tiles are the same
                            
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles 
        self.iht = tc.IHT(iht_size)
    
    def get_tiles(self, state):
        """
        Takes in an angle and angular velocity from the pendulum environment
        and returns a numpy array of active tiles.
        
        Arguments:
        angle -- float, the angle of the pendulum between -np.pi and np.pi
        ang_vel -- float, the angular velocity of the agent between -2*np.pi and 2*np.pi
        
        returns:
        tiles -- np.array, active tiles
        
        """
        
        ### Set the max and min of angle and ang_vel to scale the input (4 lines)
        
        ### START CODE HERE ###
        PARAM_MIN = -1
        PARAM_MAX = 1
        ### END CODE HERE ###

        
        ### Use the ranges above and self.num_tiles to set angle_scale and ang_vel_scale (2 lines)
        
        
        ### START CODE HERE ###
        param_scale = self.num_tiles / (PARAM_MAX - PARAM_MIN)
        
        ### END CODE HERE ###
        scaled_state = state * param_scale
        
        # Get tiles by calling tc.tileswrap method
        # wrapwidths specify which dimension to wrap over and its wrapwidth
        tiles = tc.tileswrap(self.iht, self.num_tilings, scaled_state, wrapwidths=[self.num_tiles, False])
                    
        return np.array(tiles)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
    
class Agent(): #BaseAgent
    def __init__(self):
        self.rand_generator = None

        self.actor_step_size = None
        self.critic_step_size = None
        self.avg_reward_step_size = None

        self.tc = None

        self.avg_reward = None
        self.critic_w = None
        self.actor_w = None

        self.actions = None

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None
    
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) state aggregation agent.

        Assume agent_info dict contains:
        {
            "iht_size": int
            "num_tilings": int,
            "num_tiles": int,
            "actor_step_size": float,
            "critic_step_size": float,
            "avg_reward_step_size": float,
            "num_actions": int,
            "seed": int
        }
        """

        # set random seed for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed")) 

        iht_size = agent_info.get("iht_size")
        num_tilings = agent_info.get("num_tilings")
        num_tiles = agent_info.get("num_tiles")

        # initialize self.tc to the tile coder we created
        self.tc = PendulumTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        # set step-size accordingly (we normally divide actor and critic step-size by num. tilings (p.217-218 of textbook))
        self.actor_step_size = agent_info.get("actor_step_size")/num_tilings
        self.critic_step_size = agent_info.get("critic_step_size")/num_tilings
        self.avg_reward_step_size = agent_info.get("avg_reward_step_size")

        self.num_actions = agent_info.get("num_actions")
        self.actions = list(range(agent_info.get("num_actions")))

        # Set initial values of average reward, actor weights, and critic weights
        # We initialize actor weights to three times the iht_size. 
        # Recall this is because we need to have one set of weights for each of the three actions.
        self.avg_reward = 0.0
        self.actor_mu_w = np.zeros((iht_size, len(self.actions)))
        self.actor_sig_w = np.zeros((iht_size, len(self.actions)))
        self.critic_w = np.zeros(iht_size)

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None
    
    def agent_policy(self, active_tiles):
        """ policy of the agent
        Args:
            active_tiles (Numpy array): active tiles returned by tile coder
            
        Returns:
            The action selected according to the policy
        """
        
        mu_vec = self.actor_mu_w[active_tiles].sum(axis=0)
        mu_vec = np.reshape(mu_vec, (self.num_actions, 1))
        
        print('mu_vec is:')
        print(mu_vec)
        print('\n')
        
        np.nan_to_num(self.actor_sig_w[active_tiles], posinf=0., neginf=0,)
        sigma_vec = np.exp(self.actor_sig_w[active_tiles].sum(axis=0))
        sigma_vec = np.reshape(sigma_vec, (self.num_actions, 1))
        
        print('sigma_vec is:')
        print(sigma_vec)
        print('\n')
        
        actions = np.zeros((self.num_actions))
        for i in range(len(actions)):
            actions[i] = self.rand_generator.normal(mu_vec[i], sigma_vec[i])
        # np.nan_to_num(actions, copy=False)
        # actions = np.tanh(actions)
        return actions
    
    def start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """

        # angle, ang_vel = state

        ### Use self.tc to get active_tiles using angle and ang_vel (2 lines)
        # set current_action by calling self.agent_policy with active_tiles
        # active_tiles = ?
        # current_action = ?

        ### START CODE HERE ###
        active_tiles = self.tc.get_tiles(state)
        current_action = self.agent_policy(active_tiles)
        ### END CODE HERE ###

        self.last_action = current_action
        self.prev_tiles = np.copy(active_tiles)

        return self.last_action


    def step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step based on 
                                where the agent ended up after the
                                last step.
        Returns:
            The action the agent is taking.
        """

        ### Use self.tc to get active_tiles using angle and ang_vel (1 line)
        # active_tiles = ?    
        ### START CODE HERE ###
        active_tiles = self.tc.get_tiles(state)
        current_action = self.agent_policy(active_tiles)
        ### END CODE HERE ###

        ### Compute delta using Equation (1) (1 line)
        # delta = ?
        ### START CODE HERE ###
        delta = reward - self.avg_reward + self.critic_w[active_tiles].sum() - self.critic_w[self.prev_tiles].sum()
        ### END CODE HERE ###

        ### update average reward using Equation (2) (1 line)
        # self.avg_reward += ?
        ### START CODE HERE ###
        self.avg_reward += self.avg_reward_step_size * delta
        ### END CODE HERE ###

        # update critic weights using Equation (3) and (5) (1 line)
        # self.critic_w[self.prev_tiles] += ?
        ### START CODE HERE ###
        print('critic update is:')
        print(self.critic_step_size * delta)
        print('\n')
        self.critic_w[self.prev_tiles] += self.critic_step_size * delta
        ### END CODE HERE ###

        # update actor weights using Equation (4) and (6)
        # We use self.softmax_prob saved from the previous timestep
        # We leave it as an exercise to verify that the code below corresponds to the equation.
        
        mu_vec = self.actor_mu_w[self.prev_tiles].sum(axis=0)
        mu_vec = np.reshape(mu_vec, (self.num_actions, 1))     
        sigma_vec = np.exp(self.actor_sig_w[self.prev_tiles].sum(axis=0))
        sigma_vec = np.reshape(sigma_vec, (self.num_actions, 1))
        # print('sigma_vec is:')
        # print(sigma_vec)
        # print('\n')
        
        for a in self.actions:
            # print('sigma weights before update were:')
            # print(self.actor_sig_w[self.prev_tiles, a])
            # print('\n')
            
            print('mu update is:')
            print(self.actor_step_size * delta * (1. / sigma_vec[a])**2 * (self.last_action[a]-mu_vec[a]))
            print('\n')
            self.actor_mu_w[self.prev_tiles, a] += self.actor_step_size * delta * (1. / sigma_vec[a])**2 * (self.last_action[a]-mu_vec[a])
            
            print('sigma update is:')
            print(self.actor_step_size * delta * (((self.last_action[a]-mu_vec[a]) / sigma_vec[a])**2 - 1))
            print('\n')
            self.actor_sig_w[self.prev_tiles, a] += self.actor_step_size * delta * (((self.last_action[a]-mu_vec[a]) / sigma_vec[a])**2 - 1)
            
        self.prev_tiles = active_tiles
        self.last_action = current_action
        
        print('action is:')
        print(self.last_action)
        print('\n')

        return self.last_action


    def agent_message(self, message):
        if message == 'get avg reward':
            return self.avg_reward

agent_parameters = {
    "num_tilings": 32,
    "num_tiles": 8,
    "actor_step_size": 2**(-2),
    "critic_step_size": 2**1,
    "avg_reward_step_size": 2**(-6),
    "num_actions": 4,
    "iht_size": 4096
}

env = gym.make('BipedalWalker-v3')
agent = Agent()
agent.agent_init(agent_parameters)
state = env.reset()

action = agent.start(state)
state, reward, done, l = env.step(action)
r_sum = 0
rewards = []
ep = 0
loops = 0
while ep < 1000:
    env.render()
    loops += 1
    print("loop number:")
    print(loops)
    print('\n')
    action = agent.step(reward, state)
    state, reward, done, l = env.step(action)
    r_sum += reward
    if done:
        # agent.end(reward)
        state = env.reset()
        action = agent.start(state)
        rewards.append(r_sum)
        r_sum = 0
        ep += 1
        print('_______________________________')
        print("ep number:")
        print(ep)
        print('_______________________________')
        print('\n')
        state, reward, done, l = env.step(action)
        
env.close()
plt.figure()
plt.plot(rewards)
plt.title('Total rewards per episode')
plt.savefig('rewards.png')
