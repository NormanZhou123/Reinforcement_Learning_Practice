import numpy as np
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2


if __name__ == "__main__":
    V1 = np.load('Value_function.npy')
    plt1.show()
    plt1.plot(V1,label = "")
    plt1.xlim([0,100])
    plt1.xticks([1,25,50,75,99])
    plt1.xlabel('Capital')
    plt1.ylabel('Value estimates')
    plt1.legend()
    plt1.show()

    V2 = np.load('policy.npy')
    plt2.show()
    plt2.plot(V2, label="")
    plt2.xlim([0, 100])
    plt2.xticks([1, 25, 50, 75, 99])
    plt2.xlabel('Capital')
    plt2.ylabel('Stake')
    plt2.legend()
    plt2.show()