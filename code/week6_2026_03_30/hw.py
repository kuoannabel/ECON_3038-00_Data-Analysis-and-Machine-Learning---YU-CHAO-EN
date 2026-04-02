from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
"""
1.What is the best exclusion of explanatory variables for the California housing dataset?
2.What is log(validation loss) for the best selection   (小數點以下三位，第四位四捨五入) ?
"""
# 1. 載入與清理資料
df = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv')
df = df.drop(['ocean_proximity'], axis=1)
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())

# 定義特徵與目標
X = df.drop(['median_house_value'], axis=1).values
t = df['median_house_value'].values
feature_names = df.columns[:-1]

# 2. 切分資料 (維持 shuffle=False 確保可重複性)
trainX, validX, traint, validt = train_test_split(X, t, test_size=0.2, shuffle=False)

# -----------------------------
# 🔥 核心：純模型版本的實驗
# -----------------------------
results = []

for i in range(trainX.shape[1]):
    # 移除一個特徵 (使用 np.delete)
    X_train_drop = np.delete(trainX, i, axis=1)#trainX 的第 i 列被刪除後的訓練特徵矩陣
    X_valid_drop = np.delete(validX, i, axis=1)#validX 的第 i 列被刪除後的驗證特徵矩陣

    # 3. 訓練模型 (這就是只用模型的簡潔之處)
    # 不需手動加 ones，LinearRegression 會自動處理 Intercept
    model = LinearRegression()
    model.fit(X_train_drop, traint)#訓練模型，使用刪除特徵後的訓練資料 X_train_drop 和對應的目標變數 traint
    
    # 4. 預測與計算損失
    predict = model.predict(X_valid_drop)#使用刪除特徵後的驗證資料 X_valid_drop 進行預測，得到預測值 predict
    
    # 計算 SSE (Sum of Squared Errors)
    sse = np.sum((validt - predict) ** 2)

    # 2. 取 log10
    log10_sse = np.log10(sse)
    
    # 注意這裡要用 X_valid_drop 而不是 validX
    #by hand calculation of log10_sse
    error = validt - (model.intercept_ + X_valid_drop @ model.coef_)
    log10_sse = np.log10(np.sum(error ** 2))
    # 3. 儲存結果
    results.append(log10_sse)
    print(f"Drop {feature_names[i]:18} → Log10(SSE) = {log10_sse:.4f}")

# 5. 找出最佳排除對象
best_index = np.argmin(results)#找到 results 中最小值的索引，這個索引對應於最佳排除特徵
print("\n" + "="*30)
print(f"Best Excluded Feature: {feature_names[best_index]}")
print(f"Smallest Log(Validation Loss): {results[best_index]:.4f}")
print("="*30)

# 將結果轉換為「損失增加量」（相對於排除掉最不重要特徵後的變化）
plt.figure(figsize=(10, 6))
plt.barh(feature_names, results, color='teal')
plt.xlabel('Average Log Loss (smaller is better)')
plt.ylabel('Dropped Feature')
plt.title('Feature Sensitivity Analysis (Leave-One-Out)')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 標註出最優選擇
plt.gca().get_yticklabels()[best_index].set_color('red')
plt.gca().get_yticklabels()[best_index].set_weight('bold')

plt.show()

# 1. 用「所有特徵」訓練一個最終模型
final_model = LinearRegression()
final_model.fit(trainX, traint)

# 2. 建立一個 DataFrame 讓結果變整齊
importance_df = pd.DataFrame({#pd.DataFrame 是用來創建一個新的 DataFrame，這裡我們將特徵名稱和對應的權重（係數）放在一起，方便後續分析和展示。
    'Feature': feature_names,
    'Coefficient (Weight)': final_model.coef_
})

# 3. 依照權重絕對值排序 (看誰影響力最大)
importance_df['Absolute_Weight'] = importance_df['Coefficient (Weight)'].abs()
importance_df = importance_df.sort_values(by='Absolute_Weight', ascending=False)

print("\n--- 模型學到的特徵權重分析 ---")
print(importance_df[['Feature', 'Coefficient (Weight)']].to_string(index=False))

 
#標準化 (Standardization)
print("\n--- Standardization ---")
from sklearn.preprocessing import StandardScaler

# 1. 初始化標準化器
scaler = StandardScaler()

# 2. 對特徵進行縮放 (注意：目標值 traint 通常不縮放,保持原始單位)
trainX_scaled = scaler.fit_transform(trainX)
validX_scaled = scaler.transform(validX)

# 3. 重新訓練模型
scaled_model = LinearRegression()
scaled_model.fit(trainX_scaled, traint)

# 4. 建立新的權重分析表
importance_scaled = pd.DataFrame({
    'Feature': feature_names,
    'Scaled Coefficient': scaled_model.coef_
})

# 排序看看
importance_scaled['Abs_Weight'] = importance_scaled['Scaled Coefficient'].abs()
print("\n--- 標準化後的特徵權重 (這才是真實影響力) ---")
print(importance_scaled.sort_values(by='Abs_Weight', ascending=False)[['Feature', 'Scaled Coefficient']].to_string(index=False))

# 查看 scaler 記住的平均值與標準差
print("\n--- 標準化單位換算 ---")
for name, m, s in zip(feature_names, scaler.mean_, scaler.scale_):
    print(f"{name:18} | Average: {m:10.2f} | 1 Standard Deviation = {s:10.2f}")
   
   
def predict_house_price(model, scaler, feature_names):
    print("\n--- 🏠 歡迎使用加州房價預測器 ---")
    print("Please enter the following information (if unsure, refer to the average values):")
    
    user_input = []
    # 這裡我們手動模擬一組數據，或讓使用者輸入
    # 範例數據：經度 -122, 緯度 37, 屋齡 30, 房間 2000, 臥室 400, 人口 1000, 戶數 300, 收入 5
    default_values = [-122.0, 37.0, 30.0, 2000.0, 400.0, 1000.0, 300.0, 5.0]
    
    for i, name in enumerate(feature_names):
        val = input(f"Please enter {name} (default {default_values[i]}): ")
        user_input.append(float(val) if val else default_values[i])
    
    # 1. 轉為 2D Array
    input_array = np.array([user_input])
    
    # 2. 必須使用剛才訓練時的 scaler 進行標準化！
    input_scaled = scaler.transform(input_array)
    
    # 3. 預測
    prediction = model.predict(input_scaled)
    
    print(f"\n✨ 預測結果：該房屋估價約為 ${prediction[0]:,.2f}")
    print("--------------------------------")

# 執行預測器
predict_house_price(scaled_model, scaler, feature_names)