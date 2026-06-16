import numpy as np
from matplotlib import pyplot as plt

data=np.random.normal(0,1,size=100000)
plt.hist(data, bins=10)