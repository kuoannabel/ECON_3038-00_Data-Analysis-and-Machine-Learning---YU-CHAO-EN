import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, multivariate_normal

# 1. 初始化與 MCMC Metropolis 抽樣 (Ns 設為 10000)
truemu = [1, 1]
trueSig = [[3, 0.4], [0.4, 3]]
Ns = 10000
propSig = [[0.5, 0], [0, 0.5]]
wlst = []

np.random.seed(234)
w = [np.random.rand(), np.random.rand()]

for i in range(Ns):
    wtilde = multivariate_normal.rvs(w, propSig)
    r = multivariate_normal.pdf(wtilde, truemu, trueSig) / multivariate_normal.pdf(w, truemu, trueSig)
    if r >= 1:
        w = wtilde
    elif np.random.rand() < r:
        w = wtilde
    wlst.append(w)

wlst = np.array(wlst)
print(f"成功抽取樣本數: {len(wlst)}")

# =====================================================================
# 2. 核心：單獨計算特定點 (x1=5, x2=-5) 的貝氏預測機率
# =====================================================================
target_x1 = 5.0
target_x2 = -5.0

# 帶入投影片的公式：對 10000 個樣本計算 Sigmoid 後取平均
predictive_prob = np.mean([
    1 / (1 + np.exp(-(wlst[i][0] * target_x1 + wlst[i][1] * target_x2))) 
    for i in range(len(wlst))
])

# 印出結果（保留小數點後三位以利四捨五入觀察）
print("\n--------------------------------------------------")
print(f"點 ({target_x1}, {target_x2}) 的預測機率為: {predictive_prob:.3f}")
print(f"四捨五入至小數點後兩位: {predictive_prob:.2f}")
print("--------------------------------------------------\n")

# =====================================================================
# 3. 繪圖驗證 (為了讓程式跑快一點，我們只用前 500 個樣本來畫決策邊界)
# =====================================================================
x1 = np.arange(-5, 5, 0.1)
x2 = np.arange(-5, 5, 0.1)
zz = []
# 減量繪圖以加速
wlst_sub = wlst[:500] 

for y in x2:
    for x in x1:
        zz.append(np.mean([1 / (1 + np.exp(-(wlst_sub[i][0]*x + wlst_sub[i][1]*y))) >= 0.9 for i in range(len(wlst_sub))]))

zz = np.array(zz).reshape(len(x2), len(x1))

# 畫出 0.5 的決策邊界
plt.contour(x1, x2, zz, [0.9], colors='red', linewidths=2)
# 標註我們要測量的點 (5, -5) 
plt.scatter(target_x1, target_x2, color='green', marker='X', s=100, label='Target Point (5, -5)')
plt.scatter(wlst_sub[:, 0], wlst_sub[:, 1], s=0.5, alpha=0.5, label='MCMC Samples')

plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Decision Boundary and Target Point')
plt.legend()
plt.grid(True)
plt.show()