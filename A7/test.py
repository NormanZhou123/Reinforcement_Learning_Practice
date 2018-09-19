import numpy as np
from utils import rand_un

x=[]
y=[]
for i in range(50):
    x.append(-1.2+ (1.7/50)*i)
    y.append(-0.07+(0.14/50)*i)

print x
print y

x = np.arange(-1.2, 0.5, 1.7 / 50)
y = np.arange(-0.07, 0.07, 0.14 / 50)
print x
print y



