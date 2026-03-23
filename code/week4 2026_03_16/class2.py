import numpy as np

prior = []
for i in range(11):
    if i / 10 <= 0.5:
        prior.append(i / 10 * 2 / 5)
    else:
        prior.append(0.2 + (i / 10 - 0.5) * (-2/5))
        
likelihood = []
for i in range(11):
        likelihood.append(i / 10)
        
num = []
for i in range(11):
    num.append(prior[i] * likelihood[i])
    
posterior = np.array(num) / sum(num)# ex:num = [2,4,6] [2,4,6] / 12 = [1,2,3]
print(posterior)
#bayes theorem: P(A|B) = P(B|A) * P(A) / P(B) = prior * likelihood / sum(prior * likelihood) = num / sum(num)