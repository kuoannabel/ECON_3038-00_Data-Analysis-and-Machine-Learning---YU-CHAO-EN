import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal

# 讀取資料
# 注意：請確保路徑正確，或將檔案放在與程式碼相同的資料夾
file_path = 'C:\\Users\\kuoan\\OneDrive\\Desktop\\data\\ECON_3038-00_Data-Analysis-and-Machine-Learning---YU-CHAO-EN\\code\\week10_2026_04_27\\olympic100m.csv'
data = np.array(pd.read_csv(file_path, header=None))

tall = data[:, 1]
Xall = np.transpose(np.array([np.ones(len(data)), data[:, 0]]))
sig = np.sqrt(10)
mu0 = [0, 0]
SIG0 = [[100, 0], [0, 5]]

# --- 第一部分：Prior 分佈 ---
w0 = np.linspace(-20, 20, 1001)
w1 = np.linspace(-6, 6, 1001)
W0, W1 = np.meshgrid(w0, w1)

# 使用更加優化的方式計算 PDF (避免雙重迴圈)
pos = np.dstack((W0, W1))
prior = multivariate_normal.pdf(pos, mean=mu0, cov=SIG0)

plt.figure()
plt.contour(W0, W1, prior)
plt.gca().set_aspect('equal')
plt.title('Prior distribution over weights (w0 and w1)')
plt.xlabel('w0')
plt.ylabel('w1')
plt.show()

# --- 第二部分：觀察 2 個數據點後的 Posterior ---
X = Xall[:2]
t = tall[:2]
SIGw = np.linalg.inv(X.T @ X / sig**2 + np.linalg.inv(SIG0))
muw = SIGw @ (X.T @ t / sig**2 + np.linalg.inv(SIG0) @ mu0)

posterior2 = multivariate_normal.pdf(pos, mean=muw, cov=SIGw)

plt.figure()
plt.contour(W0, W1, posterior2)
plt.contour(W0, W1, prior, alpha=0.3) # 繪製淡淡的 Prior 當背景
plt.title('Posterior distribution after observing 2 data points')
plt.xlabel('w0')
plt.ylabel('w1')
plt.show()

# --- 第三部分：觀察所有數據點後的 Posterior 與 MAP 計算 ---
X = Xall
t = tall
SIGw = np.linalg.inv(X.T @ X / sig**2 + np.linalg.inv(SIG0))
muw = SIGw @ (X.T @ t / sig**2 + np.linalg.inv(SIG0) @ mu0)
print("posterior muw (Analytical MAP):", muw)

# 重新定義範圍以觀察細節 (Fig 3-21)
w0_fine = np.linspace(7, 15, 1001)
w1_fine = np.linspace(-0.5, 0.5, 1001)
W0_f, W1_f = np.meshgrid(w0_fine, w1_fine)
pos_f = np.dstack((W0_f, W1_f))
posterior_all = multivariate_normal.pdf(pos_f, mean=muw, cov=SIGw)

# --- 修正報錯的部分 ---
# 我們需要將網格 (W0_f, W1_f) 拉平，而不是將原始的 w0_fine 拉平
posterior_flat = posterior_all.flatten()
W0_flat = W0_f.flatten()
W1_flat = W1_f.flatten()

# 尋找機率最大值索引
max_idx = np.argmax(posterior_flat)

map_w0 = W0_flat[max_idx]
map_w1 = W1_flat[max_idx]

print(f"w0 of the MAP: {map_w0:.2f}")
print(f"w1 of the MAP: {map_w1:.2f}")

# 繪圖
plt.figure()
plt.contour(W0_f, W1_f, posterior_all)
# 這裡因範圍差異較大，通常不畫原 prior，或只畫出中心點
plt.title('Posterior distribution after observing all data points')
plt.scatter(map_w0, map_w1, color='red', marker='x', label='MAP Estimate')
plt.xlabel('w0')
plt.ylabel('w1')
plt.legend()
plt.show()