from utils import rand_norm, rand_in_range, rand_un
import numpy as np

current_state = None
start_position = [0, 3]
end_position = [8,5]
obs = [[2,2],[2,3],[2,4],[5,1],[7,3],[7,4],[7,5]]

def env_init():

    global current_state

    current_state = np.zeros(1)


def env_start():
    """ returns numpy array """
    global current_state

    current_state = start_position

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

    move = None
    terminal = False

    if action == 0:
        move = [0,1]
    elif action == 1:
        move = [0,-1]
    elif action == 2:
        move = [-1,0]
    elif action == 3:
        move = [1,0]

    else:
        print "Invalid action taken!!"
        exit(1)
    current_x = current_state[0]
    current_y = current_state[1]

    new_x = current_x+move[0]
    new_y = current_y+move[1]

    if new_x >= 0 and new_y >=0 and new_x <=8 and new_y <= 5:
        if [new_x,new_y] not in obs:
            current_state = [new_x, new_y]

    if current_state == end_position:
        terminal = True

    if terminal is True:
        reward = 1
    else:
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
