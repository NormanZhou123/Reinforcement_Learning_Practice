#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""

from tiles3 import tiles, IHT

from rl_glue import *  # Required for RL-Glue
RLGlue("mountaincar", "sarsa_lambda_agent")

import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def get_value():
    fout = open('value', 'w')
    w, iht = RL_agent_message("ValueFunction")
    q = np.zeros((50,50))
    for i in range(50):
        for j in range(50):
            pos = 8 * (-1.2 + (i * 1.7 / 50)) / (0.5 + 1.2)
            vel = 8 * (-0.07 + (j * 0.14 / 50)) / (0.07 + 0.07)
            state = [pos, vel]
            values = []
            for a in range(3):
                feature = np.zeros(1944)
                feature_list = tiles(iht, 8, state, [a])

                for k in feature_list:
                    feature[k] = 1
                values.append(-np.dot(w, feature))

            height = np.max(values)
            q[i][j] = height
            fout.write(repr(height) + '')
        fout.write('\n')
    fout.close()

    return q


if __name__ == "__main__":
    num_episodes = 1000
    num_runs = 1
    steps = np.zeros([num_runs,num_episodes])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = []
    y = []
    for i in range(50):
        x.append(-1.2 + (1.7 / 50) * i)
        y.append(-0.07 + (0.14 / 50) * i)
    x, y = np.meshgrid(x, y)

    for r in range(num_runs):
        print "run number : ", r
        RL_init()
        for e in range(num_episodes):
            # print '\tepisode {}'.format(e+1)
            RL_episode(0)
            #steps[r,e] = RL_num_steps()
            v = get_value()

    ax.set_xticks([-1.2, 0.5])
    ax.set_yticks([-0.07, 0.07])
    ax.set_zticks([0, np.amax(v)])
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')
    ax.plot_wireframe(x, y, v)
    plt.show()