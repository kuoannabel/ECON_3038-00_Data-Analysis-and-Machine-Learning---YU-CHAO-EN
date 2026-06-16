import numpy as np
from matplotlib import pyplot as plt
#Exam N = 10
# Assigned n = 9
#Based on N=10 and the  assigned n, (the left limit of HDI)*1000=? (請填整數)
#Based on N=10 and the  assigned n, (the right limit of HDI)*1000=? (請填整數)

#this is the code for calculating the posterior distribution of theta given the data
prior=[];
for theta in range(501):
    prior.append(theta*2/(1000*500))
print("prior[1]:", prior[1])

print("prior[499]:", prior[499], "prior[500]:", prior[500])#the prior distribution is a linear function that increases from 0 to 1 as theta goes from 0 to 500, and then decreases from 1 to 0 as theta goes from 500 to 1000, the prior distribution is symmetric around theta=500, and the maximum value of the prior distribution is 1 at theta=500
len(prior)
print("Length of prior:", len(prior))
for theta in range(501,1001):
    prior.append(2/1000-(theta-500)*2/(1000*500))
prior[-2]
len(prior)
plt.plot(prior)
plt.show()


prior[499:-499]
for i in range(2,115):
    if sum(prior[i:-i])<0.95:
        print(i-1); break
        
N = 10
num = 4
lkh=[];

for theta in range(1001):
    lkh.append((theta/1000)**num * (1-theta/1000)**(N-num))#the likelihood function is a binomial distribution with parameters n=10 and k=9, where theta is the probability of success, the likelihood function is proportional to theta^k * (1-theta)^(n-k), which is the probability of observing k successes in n trials given the probability of success theta 在給定成功機率 θ 的情況下，觀察到 k 次成功（以及 n−k 次失敗）的機率。
len(lkh)
plt.plot(lkh)
plt.show()


num = np.array(prior) * np.array(lkh);
den = sum(num);
posterior = num / den;
plt.plot(posterior)

plt.legend()
plt.show()



limit=[]; diff=[];

for i in range(200):
    for j in range(i+1,len(posterior)):
        if sum(posterior[i:j+1])>=0.95 and sum(posterior[i:j])<0.95:#the sum of the posterior distribution from i to j is greater than or equal to 0.95, but the sum from i to j-1 is less than 0.95, then we can say that the interval [i, j] contains 95% of the posterior distribution
            limit.append([i, j])#the limit of the interval that contains 95% of the posterior distribution
            diff.append(abs(posterior[i]-posterior[j]))#the difference between the posterior distribution at the left and right limits of the interval, we want to find the interval that has the smallest difference, which means that the posterior distribution is more concentrated in that interval
for i in range(len(diff)):
    if diff[i]==min(diff):#find the interval that has the smallest difference between the posterior distribution at the left and right limits
        print(limit[i])#print the limits of the interval that contains 95% of the posterior distribution with the smallest difference between the posterior distribution at the left and right limits
        left=limit[i][0]; right=limit[i][1];
print(f"Left limit: {left}, Right limit: {right}")#the limits of the interval that contains 95% of the posterior distribution with the smallest difference between the posterior distribution at the left and right limits

print(sum(posterior[left: right+1]))
print(sum(posterior[left: right]))
cdf = np.cumsum(posterior)

#show HDI and equal tail interval
plt.plot(posterior)
plt.axvline(left, color='red', label='HDI')
plt.axvline(right, color='blue', label='HDI')
#plt.axvline(right, color='green', label='Equal tail')
plt.legend()
plt.show()