# 4 doors
import numpy as np

swwin = 0; nswin = 0
np.random.seed(234)
for i in range(100000):
    car = np.random.randint(1,5)
    first = np.random.randint(1,5)
    if car == first:
        nswin += 1
    else:
        ava = [1,2,3,4];ava.remove(car);ava.remove(first)
        opened = ava[np.random.randint(0,2)]
        ava = [1,2,3,4];ava.remove(first);ava.remove(opened)
        switched = ava[np.random.randint(0,2)]
        if switched == car:
            swwin += 1
        else:
            nswin += 1
print(swwin)
print(3/8)#3/4*1/2