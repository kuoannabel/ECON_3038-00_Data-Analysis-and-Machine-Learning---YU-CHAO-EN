import numpy as np
from matplotlib import pyplot as plt

prior=[];
for theta in range(501):
    prior.append(theta*2/(1000*500))
prior[1]
len(prior)
for theta in range(501,1001):
    prior.append(2/1000-(theta-500)*2/(1000*500))
prior[-2]
len(prior)
plt.plot(prior)
prior[499:-499]
for i in range(2,115):
    if sum(prior[i:-i])<0.95:
        print(i-1); break

lkh=[];
for theta in range(1001):
    lkh.append((theta/1000)**1 * (1-theta/1000)**(4-1))
len(lkh)
plt.plot(lkh)

num=np.array(prior)*np.array(lkh);
den=sum(num);
posterior=num/den;
plt.plot(posterior)

limit=[]; diff=[];
for i in range(200):
    for j in range(i+1,len(posterior)):
        if sum(posterior[i:j+1])>=0.95 and __________________<0.95:
            limit.append([i, j])
            diff.append(abs(posterior[i]-posterior[j]))
for i in range(len(diff)):
    if diff[i]==_______________:
        left=limit[i][0]; right=limit[i][1];
left
right
sum(posterior[left: right+1])
sum(posterior[left: right])