import math
import numpy as np
from matplotlib import pyplot as plt

random_state = 234
np.random.seed(random_state)

pi_estimates = []
cnt = 0

for i in range(10000):
    
    d = np.random.rand()/2
    theta = np.random.rand() * math.pi
    
    if d < np.sin(theta)/2 :
        cnt += 1
        
    P = cnt / (i+1)  # Avoid division by zero
    pi_estimates.append(P)
print("cnt:", cnt)
print("P:", cnt/10000)#1/pi
plt.plot(pi_estimates)
plt.axhline(math.pi)
plt.show()

