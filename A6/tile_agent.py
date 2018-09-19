from utils import rand_in_range, rand_un
import numpy as np
import pickle
from tiles3 import tiles, IHT

w = None
last_state = None
gamma = 1
iht = None
numTilings = 50
alpha = 0.01/numTilings
Value_func = None
state1 = None


def agent_init():
    global w, last_state, iht, Value_func, state1
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    w = np.zeros(1200)
    last_state = np.zeros(1)
    iht = IHT(3000)
    Value_func = np.zeros(1000)
    state1 = np.zeros(1)


def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global last_state, state1

    action = np.random.binomial(1, 0.5)
    state1 = state/200
    last_state = state

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: floating point
    """

    global last_state, w, Value_func, state1

    current_state = np.zeros(1)
    current_state[0] = float(state[0] / 200.0)
    feature = np.zeros(1200)
    last_feature = np.zeros(1200)
    feature_list = tiles(iht, 50, current_state)
    last_feature_list = tiles(iht, 50, state1)

    for i in feature_list:
        feature[i] = 1

    for i in last_feature_list:
        last_feature[i] = 1

    action = np.random.binomial(1, 0.5)

    w += alpha*(gamma * np.dot(w, feature)
                - np.dot(w, last_feature)) * last_feature
    Value_func[last_state[0]-1] = np.dot(w, last_feature)


    state1 = state/200
    last_state = state

    return action


def agent_end(reward):

    global w, Value_func
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    last_feature = np.zeros(1200)
    last_feature_list = tiles(iht, 50, state1)
    for i in last_feature_list:
        last_feature[i] = 1

    w += alpha * (reward - np.dot(w, last_feature)) * last_feature
    Value_func[last_state[0]-1] = np.dot(w, last_feature)

    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Value_func
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return Value_func
    else:
        return "I don't know what to return!!"