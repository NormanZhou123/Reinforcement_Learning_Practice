#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017

  agent does *no* learning, selects actions randomly from the set of legal actions

"""

from utils import rand_in_range
from rl_glue import *  # Required for RL-Glue
import numpy as np

last_action = None  # last_action: NumPy array

num_actions = 10
Q_a = None
ini = 0 # initial value of Q_a
epsilon = 0 # probability of the epsilon case


def agent_init():
    global last_action, Q_a, ini

    Q_a = np.zeros(10) + ini

    last_action = np.zeros(1)  # generates a NumPy array with size 1 equal to zero


def agent_start(this_observation):  # returns NumPy array, this_observation: NumPy array
    global last_action

    last_action[0] = rand_in_range(num_actions)

    local_action = np.zeros(1)
    local_action[0] = rand_in_range(num_actions)

    return local_action[0]


def agent_step(reward, this_observation):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    global last_action, epsilon

    local_action = np.zeros(1)
    if last_action == None:
        local_action[0] = agent_start(this_observation)
        last_action[0] = local_action[0]
    # local_action[0] = rand_in_range(num_actions)
    else:
        # might do some learning here
        # define a step size of alpha=0.1
        # find the value of reward for all 10 bandits
        # Qn = Qn-1 - alpha(Rn-1 - Qn-1)
        # Qn->local_action Qn-1->last_action Rn-1->reward of last_action
        alpha = 0.1
        action_num = int(last_action[0])
        Q_a[action_num] = Q_a[action_num] + alpha * (reward - Q_a[action_num])
        # local_action[0] = np.argmax(Q_a)

    select_option = np.array([0, 1])
    option = np.random.choice(select_option, p=[epsilon, 1 - epsilon])

    if option == 0:
        last_action[0] = rand_in_range(num_actions) # epsilon case

    else:
        last_action[0] = np.argmax(Q_a) # 1-epsilon case

    return last_action


def agent_end(reward):  # reward: floating point
    # final learning update at end of episode
    return


def agent_cleanup():
    # clean up
    return


def agent_message(inMessage):  # returns string, inMessage: string
    # might be useful to get information from the agent
    global last_action, ini, epsilon

    if inMessage == "what is your name?":
        return "my name is skeleton_agent!"
    elif inMessage == 0:
        ini = 5
        epsilon = 0
    elif inMessage == 1:
        ini = 0
        epsilon = 0.1

    # else
    return "I don't know how to respond to your message"
