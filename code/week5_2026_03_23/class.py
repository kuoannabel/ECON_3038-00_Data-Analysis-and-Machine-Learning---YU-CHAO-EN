import numpy as np 
import sympy as sp
from matplotlib import pyplot as plt
p = sp.Symbol('p')
n = 10; y = 7
lkh = sp.factorial(n) / (sp.factorial(y)*sp.factorial(n-y)) * p**y * (1-p)**(n-y)#Factorial 的中文是「階乘」。在數學中，這是一個正整數 的階乘（記作 ），指所有小於或等於  的正整數之乘積，例如 5! = 5 × 4 × 3 × 2 × 1 = 120。 


pp = np.linspace(0,1,11)
lkhv = []; #用來存放對應於 pp 中每個 p 值的 lkh 函數的值。
for i in range(len(pp)):
    lkhv.append(lkh.subs(p,pp[i]))#subs 是 sympy 中用於替換符號的函數。在這裡，lkh.subs(p, pp[i]) 的意思是將 lkh 函數中的符號 p 替換為 pp[i] 的值，從而計算出對應於 pp 中每個 p 值的 lkh 函數的值。
lkhv2 = [];
for x in pp:
    lkhv2.append(lkh.subs(p,x))#這段程式碼的作用與前一段類似，都是將 lkh 函數中的符號 p 替換為 pp 中的每個值 x，並將結果存儲在 lkhv2 列表中。這樣做的目的是計算出對應於 pp 中每個 p 值的 lkh 函數的值，並將其存儲在 lkhv2 列表中，以便後續使用或繪圖。
#lkhv == lkhv2
plt.plot(pp,lkhv)
plt.xlabel('p')
plt.ylabel('Likelihood')
plt.title('Likelihood Function')
plt.show()