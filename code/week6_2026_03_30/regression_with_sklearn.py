import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. 讀取資料 (請確保路徑正確)
# 建議將檔案放在程式碼同資料夾，使用相對路徑
try:
    data = np.array(pd.read_csv('C:\\Users\\kuoan\\OneDrive\\Desktop\\data\\ECON_3038-00_Data-Analysis-and-Machine-Learning---YU-CHAO-EN\\code\\week6_2026_03_30\\olympicman.csv', header=None))
except FileNotFoundError:
    print("找不到檔案，請確認 c:\\temp\\olympicman.csv 是否存在")
    # 這裡放一點範例數據防止程式崩潰
    data = np.array([[1900, 11.0], [1904, 11.0], [1908, 10.8], [1912, 10.7], [1920, 10.8]]) 

# 2. 切割資料 (前 19 筆為訓練集)
train_data, valid_data = data[:19], data[19:]

# 準備訓練集 X (年份) 與 t (時間)
trainx = train_data[:, 0].reshape(-1, 1)  # reshape(-1, 1) 等同於你的 np.transpose
traint = train_data[:, 1]

# 準備驗證集
validx = valid_data[:, 0].reshape(-1, 1)
validt = valid_data[:, 1]

# 3. 訓練線性回歸模型
model = LinearRegression()
model.fit(trainx, traint)
"""
而在使用 model.fit(trainx, traint) 時,Scikit-learn 在後台自動幫你完成了以下幾件事：

1. 自動處理數學細節
求解權重： 它會根據資料量的大小，自動選擇最有效率的方式（可能是 SVD 分解或正規方程式）來算出最佳的斜率(coef_)與截距(intercept_)。

數值穩定性： 手寫 np.linalg.inv 時,如果矩陣不可逆(Singular Matrix),程式會崩潰；但 LinearRegression() 內部有優化機制，處理這類問題更穩定。

雖然 LinearRegression() 很方便，但理解背後的公式（如 SSE 誤差平方和）能幫你判斷模型到底好不好：

訓練集誤差 (Train SSE)：代表模型對「過去數據」的解釋能力。
驗證集誤差 (Valid SSE)：代表模型對「未來趨勢」的預測能力。

小提醒在 VS Code 中使用 LinearRegression() 時,記得檢查你的數據維度。Scikit-learn 要求特徵 $X$ 必須是 2D 矩陣（即使只有一個特徵），所以你才會看到 reshape(-1, 1) 或是 np.transpose 的操作。
"""
# 4. 計算訓練集與驗證集的誤差 (SSE)
train_pred = model.predict(trainx)
train_sse = np.sum((traint - train_pred)**2)

valid_pred = model.predict(validx)
valid_sse = np.sum((validt - valid_pred)**2)# 這裡的 valid_sse 是驗證集的誤差平方和 (Sum of Squared Errors)，它衡量了模型在驗證集上的預測表現。較小的 valid_sse 表示模型對驗證集的預測更準確。

# 5. 輸出結果 (在 VS Code 必須使用 print)
print(f"模型截距 (Intercept): {model.intercept_}")
print(f"模型斜率 (Coefficient): {model.coef_[0]}")
print(f"訓練集誤差平方和 (Train SSE): {train_sse}")
print(f"驗證集誤差平方和 (Valid SSE): {valid_sse}")
print(f"驗證集誤差的 Log10 值: {np.log10(valid_sse)}")

# 6. 畫圖分析 (這在分析數據時非常有幫助)
plt.figure(figsize=(10, 6))
plt.scatter(trainx, traint, color='blue', label='Training Data')
plt.scatter(validx, validt, color='red', label='Validation Data')
plt.plot(data[:, 0], model.predict(data[:, 0].reshape(-1, 1)), color='green', label='Linear Fit')
plt.xlabel('Year')
plt.ylabel('Time (s)')
plt.title('Olympic Men\'s 100m Times Prediction')
plt.legend()
plt.grid(True)
plt.show() # 這行會在 VS Code 彈出視窗顯示圖表