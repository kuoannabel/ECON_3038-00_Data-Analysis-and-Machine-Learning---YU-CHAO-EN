import numpy as np
import sympy as sp

x = sp.symbols('x')
f = 6 * x * ( 1 - x ) # Ex 1. p x = 6x(1 − x) for x ∈ [0,1].
f_integrated = sp.integrate(f, (x, 0, 1))
print(f)
print(f_integrated)

#normal distribution, also known as the Gaussian distribution: 積分
f2 = sp.symbols('f2')
sigma = sp.symbols('sigma')#標準差
mu = sp.symbols('mu')#平均值
mu = 0
sigma = 1
f2 = 1 / (sigma * sp.sqrt(2 * sp.pi)) * sp.exp(- (x - mu)**2 / (2 * sigma**2)) # Ex 2. p x = 1 σ √(2π) exp(−(x−μ)²/(2σ²)) for x ∈ R.
f2_integrated = sp.integrate(f2, (x, -np.inf, np.inf)).evalf() # Ex 3. Compute the probability that x ∈ [0,1] for the distribution in Ex 2.
f2_integrated_x = sp.integrate(f2 * x, (x, -np.inf, np.inf)).evalf()#verify that E x = μ for the standard normal distribution (μ = 0,σ = 1)? 為什麼要乘 𝑥 因為期望值是 加權平均 (weighted average)。
print(f2)
print(f2_integrated_x) 

f3 = sp.symbols('f3')
mu = 4
sigma = 3.5

f3 = 1 / (sigma * sp.sqrt(2 * sp.pi)) * sp.exp(-(x - mu)**2 / (2 * sigma**2))
"""
這裡建立了一個 常態分布：

平均值 μ = 4 → 分布中心在 4

標準差 σ = 3.5 → 分布寬度控制曲線的胖瘦

x → 隨機變數

sp.exp(...) → 指數函數，用來生成鐘形曲線
"""
f3_integrated = sp.integrate(f3 * (x - mu)**2, (x, -np.inf, np.inf)).evalf()
"""
(x - mu)**2 → 計算每個值和平均值的偏差平方

乘上 f3 → 加權平均

積分 → 全部可能值加總

.evalf() → 得到數值結果

對常態分布,這個結果應該等於:𝜎2=3.5**2=12.25
variance = E[(x - μ)²] = σ² = std²
"""

print(f3_integrated)