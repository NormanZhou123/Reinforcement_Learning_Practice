from utils import rand_in_range, rand_un
import numpy as np
import pickle

Q = None
epsilon = 0.1
alpha = 0.5
last_action = None
last_state = None


def agent_init():
    global Q, last_action, last_state
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    Q = np.zeros((7, 10, 8))  # change it to 9 for 9 actions
    last_action = np.zeros(1)
    last_state = np.zeros(2)


def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global Q, last_action, epsilon, last_state

    select_option = np.array([0, 1])
    option = np.random.choice(select_option, p=[epsilon, 1 - epsilon])
    x = state[0]
    y = state[1]

    if option == 0:
        action_num = rand_in_range(8)  # change this to 9 to rand in 9 actions

    else:
        action_num = np.argmax(Q[y][x])

    last_action[0] = action_num
    action = last_action[0]
    last_state = state

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: floating point
    """

    global Q, last_action, epsilon, alpha, last_state

    select_option = np.array([0, 1])
    option = np.random.choice(select_option, p=[epsilon, 1 - epsilon])
    x = state[0]
    y = state[1]

    if option == 0:
        action_num = rand_in_range(8)  # change this to 9 to rand in 9 actions

    else:
        action_num = np.argmax(Q[y][x])

    Q[last_state[1]][last_state[0]][last_action[0]] += \
        alpha*(reward+Q[y][x][action_num]-Q[last_state[1]][last_state[0]][last_action[0]])

    last_action[0] = action_num
    last_state = state
    action = last_action[0]

    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    global Q, last_action, alpha, last_state

    Q[last_state[1]][last_state[0]][last_action[0]] += alpha * (reward - Q[last_state[1]][last_state[0]][last_action[0]])

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