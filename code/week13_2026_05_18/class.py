import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, multivariate_normal

truemu=[1,1]; trueSig=[[3,0.4],[0.4,3]];
Ns=300; propSig=[[0.5,0],[0,0.5]]; wlst=[];
np.random.seed(234)
w=[np.random.rand(), np.random.rand()];
for i in range(Ns):
    wtilde=multivariate_normal.rvs(w,propSig);
    r=multivariate_normal.pdf(wtilde, truemu, trueSig)/multivariate_normal.pdf(w, truemu, trueSig)
    if r>=1:
        w=wtilde
    elif np.random.rand()<r:
        w=wtilde
    wlst.append(w)
print(len(wlst))

wlst=np.array(wlst)
plt.scatter(wlst[:,0], wlst[:,1], s=0.2)
plt.xlabel('w1'); plt.ylabel('w2');
plt.title('MCMC sampling of w')

plt.show()

#mu=np.array([np.mean(wlst[:,0]), np.mean(wlst[:,1])])
#S=((wlst-mu).T@(wlst-mu))/Ns

w1=np.arange(-5,5,0.1);
w2=np.arange(-5,5,0.1);
zz=[]
for y in w2:
    for x in w1:
        zz.append([multivariate_normal.pdf([x, y], truemu, trueSig)])
zz=np.array(zz)

zz=zz.reshape(len(w2),len(w1))
plt.contour(w1, w2, zz)
plt.scatter(wlst[:,0], wlst[:,1], s=0.2)
plt.xlabel('w1'); plt.ylabel('w2');
plt.title('MCMC sampling of w with true distribution contour')
plt.show()

x1=np.arange(-5,5,0.1);
x2=np.arange(-5,5,0.1);
zz=[]
for y in x2:
    for x in x1:
        zz.append(np.mean([
    1 / (1 + np.exp(-(wlst[i][0]*x + wlst[i][1]*y))) >= 0.9
    for i in range(len(wlst))
]))
zz=np.array(zz)
zz=zz.reshape(len(x2),len(x1))
plt.contour(x1, x2, zz, [0.9], colors='red', linewidths=2)
plt.scatter(wlst[:,0], wlst[:,1], s=0.2)
plt.xlabel('w1'); plt.ylabel('w2');
plt.title('MCMC sampling of w with decision boundary contour')

plt.show()