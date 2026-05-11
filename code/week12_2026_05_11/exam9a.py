import numpy as np
import random
from matplotlib import pyplot as plt

pop=[1,2,3,4,5,6,7]
x=4; n=500
random.seed(234)
counter=0
trajectory=[x]
for i in range(n):
    if 1<x<7:
        proposed=random.choice([x-1,x+1])
    elif x==1:
        proposed=random.choice([x,x+1])
    elif x==7:
        proposed=random.choice([x-1,x])
    if proposed/x>=1:
        counter+=1 
        x = proposed
    elif random.random()<proposed/x:
        counter+=1 
        x = proposed
    trajectory.append(x)
plt.plot(trajectory, [i for i in range(n+1)])
plt.xlabel('State')
plt.ylabel('Time step')
plt.title('Trajectory of the Markov Chain')
plt.show()