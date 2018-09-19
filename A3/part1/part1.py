import numpy as np

V = np.zeros(101)
policy = np.zeros(99)
ph = 0.25
theta = 10**-100
action_value = np.zeros(1)
sweep = 0
Value = np.zeros((99, 4))
def Iteration():
    global V, theta, sweep
    loop = True
    while loop == True:
        delta = 0
        for s in range(1, 100):
            list1 = []
            v = V[s]
            for action in range(1, min(s, 100-s)+1):
                if s + action >= 100:
                    action_value[0] = ph * (1 + 1 * V[action + s]) + (1 - ph) * (0 + 1 * V[s - action])
                else:
                    action_value[0] = ph * (0 + 1 * V[action + s]) + (1 - ph) * (0 + 1 * V[s - action])

                list1.append(action_value[0])
            V[s] = max(list1)
            policy[s-1] = list1.index(max(list1))+1
            delta = max(delta, abs(v-V[s]))
            if sweep < 3:
                Value[s - 1][sweep] = V[s]
            else:
                Value[s - 1][3] = V[s]
        if delta < theta:
            loop = False
        sweep += 1

if __name__ == '__main__':
    Iteration()
    #print policy
    np.save("Value_function", Value)
    np.save("policy", policy)
