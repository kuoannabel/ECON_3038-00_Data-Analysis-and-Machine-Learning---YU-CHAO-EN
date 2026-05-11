from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
#What is w1 corresponding to the 9th iteration in Fig 4.3 (小數點以下四位，第五位四捨五入)?
#ans: 1.6399
#What should be filled in _______ if you want to draw the line labeled with 0.9 in Fig 4.3?
#ans: 2.197
# 1. 讀取資料 (Colab 環境請改為 'data_logmap.csv')
# data = np.array(pd.read_csv('data_logmap.csv', header=None))
data = np.array(pd.read_csv('C:\\Users\\kuoan\\OneDrive\\Desktop\\data\\ECON_3038-00_Data-Analysis-and-Machine-Learning---YU-CHAO-EN\\code\\week11_2026_05_04\\data_logmap.csv', header=None))

n = len(data)
t = data[:, 2]
x = np.array(data[:, [0, 1]])
sig2 = 10

# 初始設定
w = np.array([0.0, 0.0]) # 必須使用浮點數
wlst = []

# --- 修正點：必須放入迴圈執行 9 次 ---
for i in range(9):
    # 計算目前的預測值 (Sigmoid)
    y = 1 / (1 + np.exp(-(x @ w))) 
    
    # 1. Jacobian (梯度)
    Jacobian = x.T @ (y - t) + (1/sig2) * w
    
    # 2. Hessian (二階導數矩陣)
    R = np.diag(y * (1 - y)) 
    Hessian = x.T @ R @ x + np.eye(2) / sig2
    
    # 3. 更新權重
    w = w - np.linalg.inv(Hessian) @ Jacobian
    wlst.append(w.copy())
    
    print(f"Iteration {i+1}: w = {w}")

# --- 輸出第 9 次迭代結果 ---w0 = w[0] w1 = w[1]
print(f"\n第 9 次迭代的 w1 (四捨五入): {round(w[1], 4)}")

# --- 繪圖部分 ---
w1 = w[0]; w2 = w[1]
x1 = sp.Symbol('x1'); x2 = sp.Symbol('x2')

# 修正點：決策邊界 ln(p/(1-p)) = w1*x1 + w2*x2 = 0.9 的對應值為 2.197
eq = sp.solve(w1*x1 + w2*x2 - 2.197, x2)[0]  

xx = np.linspace(-5, 5, 101)
yy = [float(eq.subs(x1, i)) for i in xx]

plt.figure(figsize=(8, 6))
plt.plot(xx, yy, 'c', label='Decision Boundary')
plt.scatter(x[t==0][:,0], x[t==0][:,1], marker='o', label='t=0')
plt.scatter(x[t==1][:,0], x[t==1][:,1], marker='s', label='t=1')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.legend()
plt.title(f"Iteration 9: w1={w1:.4f}")
plt.show()