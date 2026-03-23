import math
import numpy as np
from matplotlib import pyplot as plt

random_state = 234
np.random.seed(random_state)
no_switch_est = []

switch = 0
no_switch = 0
for i in range(10000):
    car = np.random.randint(1,4)
    first_pick = np.random.randint(1,4)
    if car == first_pick:
        no_switch += 1
        no_switch_est.append(no_switch / (i+1))
    else:
        open = 6 - car - first_pick
        second_pick = 6 - first_pick - open
        if second_pick == car:
            switch += 1
            no_switch_est.append(switch / (i+1))
        else:
            no_switch += 1
            no_switch_est.append(no_switch / (i+1))
        
    P = no_switch / (i+1)  # Avoid division by zero
    no_switch_est.append(P)
print("cnt:", no_switch)
print("P:", no_switch/10000)

print("cnt:", switch)
print("P:", switch/10000)

plt.plot(no_switch_est)
plt.axhline(1/3)
"""
為什麼是 6?

假設三扇門編號為 1、2、3。

你選了 first_pick

汽車在 car

Monty 要打開一扇 不是你選的、也不是汽車的門

那麼剩下的一扇門就可以用這個公式算：

剩下的門
=
6-(你選的門+Monty打開的門)剩下的門=6-(你選的門+Monty打開的門)
為什麼是 6?

1 + 2 + 3 = 6

如果你知道兩個門的編號，第三個門就是 6 - sum(已知兩門)
"""