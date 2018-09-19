from utils import rand_norm, rand_in_range, rand_un
import numpy as np

STEP_RANGE = 100
current_state = None
start_position = 500
end_position = [0,1001]

def env_init():

    global current_state

    current_state = np.zeros(1)


def env_start():
    """ returns numpy array """
    global current_state

    current_state[0] = start_position

    return current_state


def env_step(action):
    """
        Arguments
        ---------
        action : int
            the action taken by the agent in the current state

        Returns
        -------
        result : dict
            dictionary with keys {reward, state, isTerminal} containing the results
            of the action taken
    """
    global current_state

    move = np.random.randint(1, STEP_RANGE + 1)
    terminal = False

    new_state = np.zeros(1)
    if action not in [0,1]:
        print "Invalid action taken!!"
        exit(1)

    if action == 0:
        action = -1
    elif action == 1:
        action = 1

    new_state[0] = current_state[0] + move * action
    new_state[0] = max(min(new_state, 1001), 0)
    current_state = new_state

    if current_state[0] == end_position[0]:
        terminal = True
        reward = -1

    elif current_state[0] == end_position[1]:
        terminal = True
        reward = 1
    else:
        terminal = False
        reward = 0

    result = {"reward": reward, "state": current_state, "isTerminal": terminal}

    return result


def env_cleanup():
    #
    return


def env_message(in_message): # returns string, in_message: string
    """
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    return ""
