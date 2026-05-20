import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, multivariate_normal

zz=[]
for x2 in np.linspace(-2, 2, 11):
    for x1 in np.linspace(-2, 3, 11):
        zz.append(x1**2+x2**2)
zz=np.array(zz).reshape(11,11)
print(zz)
plt.contour(zz)
plt.title('Contour plot of x1^2+x2^2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

plt.contour(zz, [1, 2])
plt.title('Contour plot of x1^2+x2^2 with contour levels 1 and 2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()