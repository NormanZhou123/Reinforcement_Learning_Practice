import numpy as np
import pickle
import sys
import time
from rl_glue import *  # Required for RL-Glue
import math
from rndmwalk_policy_evaluation import compute_value_function
import matplotlib.pyplot as plt


def tabularAgent(V):
    RLGlue("rndmwalk_env", "tabular_agent")
    rmsve = np.zeros(2000)

    for i in range(10):
        print("run", i)
        RL_init()
        for m in range(2000):
            RL_episode(10000)
            valueFunction = RL_agent_message("ValueFunction")
            rmsve[m] += math.sqrt(np.sum((V - valueFunction) * (V - valueFunction)) / 1000.0)

        RL_cleanup()
    return rmsve/10


def tileAgent(V):
    RLGlue("rndmwalk_env", "tile_agent")
    rmsve = np.zeros(2000)

    for i in range(10):
        print("run", i)
        RL_init()
        for m in range(2000):
            RL_episode(10000)
            valueFunction = RL_agent_message("ValueFunction")
            rmsve[m] += math.sqrt( np.sum((V - valueFunction)*(V - valueFunction))/1000.0 )

        RL_cleanup()
    return rmsve/10



def aggregationAgent(V):
    RLGlue("rndmwalk_env", "aggregation_agent")
    rmsve = np.zeros(2000)

    for i in range(10):
        print("run", i)
        RL_init()
        for m in range(2000):
            RL_episode(10000)
            valueFunction = RL_agent_message("ValueFunction")
            rmsve[m] += math.sqrt(np.sum((V - valueFunction) * (V - valueFunction)) / 1000.0)

        RL_cleanup()
    return rmsve/10


if __name__ == '__main__':

    true_value = np.zeros(1000)
    true_value = compute_value_function()[1:]

    y1 = tabularAgent(true_value)
    y2 = tileAgent(true_value)
    y3 = aggregationAgent(true_value)

    plt.plot(y1, label = "tabular")
    plt.plot(y2, label = "tile")
    plt.plot(y3, label = "aggregation")

    plt.legend()
    plt.show()



