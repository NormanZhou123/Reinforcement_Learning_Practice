#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle

policy = None
Q = None
G = None
Returns = None
pair = None
pair_num = None
Total_num = None
count = None

def agent_init():
    global policy, G, Q, pair_num, Returns,Total_num
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    Q =np.zeros((99,99))
    policy = np.zeros(99)
    Returns = np.zeros((99, 99))
    pair_num = np.zeros((99, 99))
    Total_num = np.ones((99,99))
    G= np.zeros((99, 99))

    for state in range(1, 100):
        action = min(state, 100-state)
        policy[state-1] = action

    #initialize the policy array in a smart way

def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    global pair_num
    action = rand_in_range(min(state[0], 100-state[0]))+1
    #pair[0] = [state[0], action]
    pair_num[state[0]-1][action-1] += 1
    #count += 1
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: floating point
    """
    # select an action, based on Q
    global G, Q, last_action, last_state, pair, pair_num, count
    action = np.argmax(Q[state[0]-1])+1  # zero bet not allowed
    # G[last_state[0]-1][last_action[0]-1] += reward + Q[state[0]-1][action-1]
    #pair[count] = [state[0], action]
    #count += 1
    pair_num[state[0]-1][action-1] += 1
    G[state[0]-1][action-1] += reward

    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    global G, pair, pair_num, Q,Returns, policy, Total_num
    # do learning and update pi
    np.seterr(divide='ignore', invalid='ignore')
    Returns += (pair_num*reward)
    Returns += G
    Total_num += pair_num
    pair_num = np.zeros((99,99))
    G = np.zeros((99,99))

    Q = Returns/Total_num

    for s in range(1, 100):
        opt_action = np.argmax(Q[s-1])+1
        policy[s-1] = opt_action

    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Q
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return pickle.dumps(np.max(Q, axis=1), protocol=0)
    else:
        return "I don't know what to return!!"

