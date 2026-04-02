import numpy as np
from matplotlib import pyplot as plt
#After relaxing np.range(200), based on N=10 and z=4 (the left limit of HDI)*1000=?
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
plt.title("Prior Distribution")
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
plt.title("Likelihood Function")
plt.show()


num = np.array(prior) * np.array(lkh);
den = sum(num);
posterior = num / den;
plt.plot(posterior)
plt.title("Posterior Distribution")
plt.legend()
plt.show()



limit=[]; diff=[];

#是的，你理解的方向是對的：題目說「relaxing np.range(200)」就是要你把搜尋區間從只掃描前 200 個 θ 點，放寬到整個 posterior 支援範圍。
#在這段程式碼中，我們從 posterior 的第一個點開始，逐漸增加 j 的值，直到找到一個區間 [i, j]，使得 posterior 在這個區間內的總和大於或等於 0.95，但在區間 [i, j-1] 的總和小於 0.95。這樣我們就找到了包含 95% posterior 分布的區間。接著，我們計算這個區間左右端點的 posterior 值之差，並將其存儲在 diff 列表中。最後，我們找到 diff 中最小的值，對應的區間就是我們要找的 HDI 區間。
theta = []
for i in range(1001):#the theta values corresponding to the posterior distribution, we want to find the theta values that correspond to the limits of the HDI interval
    theta.append(i)
    
theta = np.array(theta)
max_index = np.argmax(posterior)#the index of the maximum value of the posterior distribution, which corresponds to the theta value that has the highest posterior probability, we want to search for the HDI interval around this theta value, because the HDI interval should contain the theta value with the highest posterior probability
max_theta = theta[max_index]
#搜尋到posterior中間就可以了，因為posterior是對稱的，所以從posterior的最大值開始往兩邊搜尋就可以找到包含95% posterior分布的區間，這樣就不需要搜尋整個posterior了，這樣可以節省計算時間，因為posterior的值在最大值附近會比較大，所以在最大值附近搜尋就可以找到包含95% posterior分布的區間了
for i in range(max_theta):
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
def plot_hdi_and_equal_tail(posterior, left, right):
    plt.plot(posterior)
    plt.axvline(left, color='red', label='HDI')
    plt.axvline(right, color='blue', label='HDI')
    plt.title("Posterior Distribution with HDI")
    #plt.axvline(right, color='green', label='Equal tail')
    plt.legend()
    plt.show()

#show prior, likelihood, posterior and HDI
def plot_distributions(prior, lkh, posterior, left, right):
    plt.plot(prior, label="Prior")
    plt.plot(lkh, label="Likelihood")
    plt.plot(posterior, label="Posterior")

    plt.axvline(left, color='red', linestyle='--', label='HDI left')
    plt.axvline(right, color='blue', linestyle='--', label='HDI right')

    plt.legend()
    plt.xlabel("theta (x1000)")
    plt.title("Bayesian Updating and 95% HDI")

    plt.show()

plot_hdi_and_equal_tail(posterior, left, right)
plot_distributions(prior, lkh, posterior, left, right)