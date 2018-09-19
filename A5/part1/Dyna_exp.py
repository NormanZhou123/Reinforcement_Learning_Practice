from rl_glue import *
RLGlue("Dyna_env", "Dyna_agent")
import matplotlib.pyplot as plt
import numpy as np


def send_message(n):
    RL_agent_message(n)
    return

if __name__ == '__main__':
    xlist = []
    ylist = []
    max_steps = 1000
    time_steps = 0
    data1 = np.zeros(51)
    data2 = np.zeros(51)
    data3 = np.zeros(51)


    send_message(0)
    for run in range(10):
        np.random.seed(run)
        RL_init()
        num_episodes = 1
        while num_episodes <= 50:
            RL_episode(2000)
            steps = RL_num_steps()
            data1[num_episodes] += steps
            num_episodes += 1
        RL_cleanup()
    data1 = data1 / 10
    plt.plot(data1, label="n=0")


    send_message(5)
    for run in range(10):
        np.random.seed(run)
        RL_init()
        num_episodes = 1
        while num_episodes <= 50:
            RL_episode(2000)
            steps = RL_num_steps()
            data2[num_episodes] += steps
            num_episodes += 1
        RL_cleanup()
    data2 = data2 / 10
    plt.plot(data2, label="n=5")


    send_message(50)
    for run in range(10):
        np.random.seed(run)
        RL_init()
        num_episodes = 1
        while num_episodes <= 50:
            RL_episode(2000)
            steps = RL_num_steps()
            data3[num_episodes] += steps
            num_episodes += 1
        RL_cleanup()
    data3 = data3 / 10
    plt.plot(data3, label="n=50")

    plt.ylim(0,800)
    plt.xlim(1,50)
    plt.xlabel('Episodes')
    plt.ylabel('Step per episode')
    plt.legend()
    plt.show()
    RL_cleanup()
