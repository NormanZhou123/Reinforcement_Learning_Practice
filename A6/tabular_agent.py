from utils import rand_in_range, rand_un
import numpy as np
import pickle

alpha = 0.5
w = None
last_state = None
gamma = 1
x = None


def agent_init():
    global w, last_state, x
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    x = np.zeros((1000,1000))
    w = np.zeros(1000)
    last_state = np.zeros(1)
    for i in range(1000):
        x[i][i] = 1


def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global last_state

    action = np.random.binomial(1, 0.5)
    last_state = state

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: floating point
    """

    global last_state, w, x

    action = np.random.binomial(1, 0.5)

    w += alpha*(gamma*w[state[0]-1] - w[last_state[0]-1])* x[last_state[0]-1]

    last_state = state

    return action


def agent_end(reward):

    global w, x, last_state
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    w += alpha*(reward - w[last_state[0]-1]) * x[last_state[0]-1]


    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global w
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return w
    else:
        return "I don't know what to return!!"