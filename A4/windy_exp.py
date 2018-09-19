from rl_glue import *
RLGlue("windy_env", "windy_agent")
import matplotlib.pyplot as plt

if __name__ == '__main__':
    xlist = [0]
    ylist = [0]
    num_episodes = 0
    max_steps = 8000
    time_steps = 0
    RL_init()
    RL_episode(max_steps)
    time_steps += RL_num_steps()
    num_episodes = RL_num_episodes()
    xlist.append(time_steps-1)
    ylist.append(0)
    xlist.append(time_steps)
    ylist.append(num_episodes)

    while time_steps <= 8000:
        RL_episode(max_steps)
        time_steps += RL_num_steps()
        num_episodes = RL_num_episodes()
        xlist.append(time_steps)
        ylist.append(num_episodes)
    plt.plot(xlist, ylist)
    plt.xlim([0, 8000])
    plt.xlabel('time steps')
    plt.ylabel('episode')
    plt.legend()
    plt.show()
    RL_cleanup()
