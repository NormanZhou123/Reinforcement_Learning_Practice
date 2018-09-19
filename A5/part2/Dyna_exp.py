from rl_glue import *
RLGlue("Dyna_env", "Dyna_agent")
import matplotlib.pyplot as plt
import numpy as np


def send_message(n):
    RL_agent_message(n)
    return


def change_alpha(alpha):
    RL_agent_message(alpha)
    return

if __name__ == '__main__':
    xlist = []
    ylist = []
    max_steps = 1000
    time_steps = 0
    data2 = np.zeros(51)
    alpha_list = [0.03125,0.0625,0.125,0.25,0.5,1.0]

    for i in range(6):
        change_alpha(alpha_list[i])
        send_message(5)
        for run in range(10):
            np.random.seed(100)
            RL_init()
            num_episodes = 1
            while num_episodes <= 50:
                RL_episode(2000)
                steps = RL_num_steps()
                data2[i] += steps
                num_episodes += 1
            RL_cleanup()
            data2 /= 50
        print data2[i]
        print alpha_list[i]
        ylist.append(data2[i])
        xlist.append(alpha_list[i])
    plt.plot(xlist, ylist)


    plt.xlabel('Alpha')
    plt.ylabel('Number of steps')
    plt.ylim(1, 100)
    plt.legend()
    plt.show()

    RL_cleanup()
