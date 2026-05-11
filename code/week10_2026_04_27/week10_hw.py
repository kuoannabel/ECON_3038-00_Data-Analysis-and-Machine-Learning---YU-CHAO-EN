import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, multivariate_normal
#What is w0 of the MAP  (小數點以下兩位) ?*11.04
#What is w1 of the MAP  (小數點以下兩位) ?* -0.05
#在 Bayesian linear regression：

#MAP = posterior mean = muw

#但 posterior 會隨資料不同而改變
#explain : This code is an implementation of Bayesian linear regression. It uses the Olympic 100m sprint data to demonstrate how the prior distribution over the weights (w0 and w1) is updated to the posterior distribution after observing the data. The code first defines the prior distribution as a multivariate normal distribution with mean mu0 and covariance SIG0. Then, it calculates the posterior distribution using the observed data (X and t) and the likelihood of the data given the weights. Finally, it visualizes the prior and posterior distributions using contour plots. The code also shows how the posterior distribution changes as more data points are added, illustrating the concept of Bayesian updating.

data=np.array(pd.read_csv('C:\\Users\\kuoan\\OneDrive\\Desktop\\data\\ECON_3038-00_Data-Analysis-and-Machine-Learning---YU-CHAO-EN\\code\\week10_2026_04_27\\olympic100m.csv', header=None))  # x's are from 0 to 28
tall=data[:,1]; Xall=np.transpose(np.array([np.ones(len(data)),data[:,0]])); 
sig=np.sqrt(10); mu0=[0,0]; SIG0=[[100,0], [0,5]]; xnew=[1,29];
print("mu0:", mu0) # Prints the mean of the prior distribution over the weights (w0 and w1). In this case, it is set to [0, 0], indicating that before observing any data, we have no specific belief about the values of the weights.
# Fig 3-19

w0=np.linspace(-20,20,1001);
w1=np.linspace(-6,6,1001);
prior=np.array([multivariate_normal.pdf([w0[i],w1[j]], mu0, SIG0) for j in range(len(w1)) for i in range(len(w0))])
prior=prior.reshape(1001,1001)# Reshapes the prior array to match the dimensions of the w0 and w1 grids for contour plotting.
print("prior:", prior) # Prints the values of the prior distribution over the weights (w0 and w1) for each combination of weights in the grid defined by w0 and w1. This array contains the probability density values of the prior distribution at each point in the weight space.
W0, W1 = np.meshgrid(w0, w1)
plt.contour(W0,W1,prior)
plt.gca().set_aspect(1)
plt.title('Prior distribution over weights (w0 and w1)')
plt.xlabel('w0')
plt.ylabel('w1')
plt.show()

X=Xall[:2]; t=tall[:2];
SIGw=np.linalg.inv(X.T@X/sig**2+np.linalg.inv(SIG0))# Calculates the covariance matrix of the posterior distribution over the weights (w0 and w1) using the formula for Bayesian linear regression. It takes into account the observed data (X) and the prior covariance (SIG0) to compute the updated covariance (SIGw) for the posterior distribution.
muw=SIGw@(X.T@t/sig**2+np.linalg.inv(SIG0)@mu0)
print("muw:", muw)# Prints the mean of the posterior distribution over the weights (w0 and w1) after observing the first two data points. This mean represents the most likely values of the weights given the observed data and the prior distribution.
w0=np.linspace(-20,20,1001);# the range of w's alters in Fig3-20
w1=np.linspace(-6,6,1001);# the range of w's alters in Fig3-20


posterior=np.array([multivariate_normal.pdf([w0[i],w1[j]], muw, SIGw) for j in range(len(w1)) for i in range(len(w0))]).reshape(1001,1001)
W0, W1 = np.meshgrid(w0, w1)
plt.contour(W0,W1,posterior)
plt.contour(W0,W1,prior)
plt.title('Posterior distribution after observing 2 data points')
plt.xlabel('w0')
plt.ylabel('w1')
plt.show()

X=Xall; t=tall;
SIGw=np.linalg.inv(X.T@X/sig**2+np.linalg.inv(SIG0))
muw=SIGw@(X.T@t/sig**2+np.linalg.inv(SIG0)@mu0)
print("posterior muw:", muw)# Prints the mean of the posterior distribution over the weights (w0 and w1) after observing all data points. This mean represents the most likely values of the weights given the observed data and the prior distribution.
w0=np.linspace(7,15,1001); # the range of w's alters in Fig3-21
w1=np.linspace(-0.5,0.5,1001);
posterior=np.array([multivariate_normal.pdf([w0[i],w1[j]], muw, SIGw) for j in range(len(w1)) for i in range(len(w0))]).reshape(1001,1001)# the range of w's alters in Fig3-21 one line for loop for calculating the posterior distribution over the weights (w0 and w1) using the multivariate normal probability density function. The loop iterates over the grid of w0 and w1 values, calculates the probability density for each combination of weights, and stores it in the posterior array. Finally, it reshapes the posterior array to match the dimensions of the w0 and w1 grids for contour plotting.

W0, W1 = np.meshgrid(w0, w1)
plt.contour(W0,W1,posterior)
plt.contour(W0,W1,prior)
posterior = posterior.reshape(1001*1001, )  # Reshapes the posterior array to match the dimensions of the w0 and w1 grids for contour plotting.
w0 = w0.reshape(1001*1001, )  # Reshapes the w0 array to match the dimensions of the posterior array for contour plotting.
w0[posterior == np.max(posterior)] # Finds the value of w0 that corresponds to the maximum value of the posterior distribution, which represents the most likely value of w0 given the observed data and the prior distribution.
print("w0 of the MAP:", w0[posterior == np.max(posterior)]) # Prints the value of w0 that corresponds to the maximum value of the posterior distribution, which represents the most likely value of w0 given the observed data and the prior distribution.
w1 = w1.reshape(1001*1001, )  # Reshapes    the w1 array to match the dimensions of the posterior array for contour plotting.
w1[posterior == np.max(posterior)] # Finds the value of w1 that corresponds to the maximum value of the posterior distribution, which represents the most likely value of w1 given the observed data and the prior distribution.
print("w1 of the MAP:", w1[posterior == np.max(posterior)]) # Prints the value of w1 that corresponds to the maximum value of the posterior distribution, which represents the most likely value of w1 given the observed data and the prior distribution.
plt.title('Posterior distribution after observing all data points')
plt.xlabel('w0')
plt.ylabel('w1')
plt.show()