from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
###
data = np.array(pd.read_csv('C:\\Users\\kuoan\\OneDrive\\Desktop\\data\\ECON_3038-00_Data-Analysis-and-Machine-Learning---YU-CHAO-EN\\code\\week7_2026_04_13\\olympicman.csv', header=None))


len(data)
data=np.split(data, [19])
train=data[0]; valid=data[1];
trainx=np.transpose([train[:,0], train[:,0]**2]); traint=train[:,1];
validx=np.transpose([valid[:,0], valid[:,0]**2]); validt=valid[:,1];
validt=valid[:,1];
print("trainx.shape:", trainx.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(trainx, traint)#fit 是 LinearRegression 模型的方法，用於訓練模型。它接受兩個參數：trainx 是訓練資料的特徵（年份），traint 是訓練資料的目標變數（奪金時間）。通過調用 fit 方法，模型會學習到特徵與目標變數之間的關係，從而找到最佳的線性回歸方程來預測奪金時間。
 #coef_ 是 LinearRegression 模型的屬性，包含了模型的係數（斜率）。在這裡，它將返回一個包含年份特徵的係數值，表示每增加一年，預測的奪金時間會增加或減少多少。這個值可以用來解釋年份對奪金時間的影響程度。
print("model.coef_斜率:", model.coef_)#每個 feature 的權重,在「單一變數線性回歸」時：是「那一條直線的 slope"
print("model.intercept_截距:", model.intercept_)

rss = sum((traint - (model.intercept_ + trainx@model.coef_))**2 )#explain: # 計算 Residual Sum of Squares (RSS, 殘差平方和)
# model.intercept_ + trainx @ model.coef_
# → 模型預測值 ŷ = β0 + Xβ 
# # traint - (預測值)
# → 殘差 (實際值 - 預測值)
# **2
# → 將殘差平方化，確保所有的殘差都是正值，並且放大較大的殘差，以便更好地反映模型的預測誤差。
# sum(...)
# → 將所有資料的平方殘差加總

print("rss:", rss)
rss_valid = np.log10(sum((validt - (model.intercept_ + validx @ model.coef_))**2 ))#截距+斜率 #@:矩陣乘法運算符，表示將 validx（驗證資料的年份特徵）與 model.coef_（模型的係數）進行矩陣乘法，得到模型對驗證資料的預測值。這個預測值再加上 model.intercept_（模型的截距）就得到了模型對驗證資料的總預測值。接著，validt - (model.intercept_ + validx @ model.coef_) 計算了驗證資料的實際值與預測值之間的殘差，**2 將殘差平方化，最後 sum(...) 將所有資料的平方殘差加總，並且 np.log10(...) 將結果取對數，以便更好地比較不同模型的表現.




###
plt.plot(trainx[:, 0], model.intercept_+trainx@model.coef_, color='blue' )#(x軸, y軸, 標記樣式, 顏色) 
#WWI 間隔八年
#model.intercept_ + trainx @ model.coef_ 是線性回歸模型的預測值，表示根據訓練資料的年份特徵（trainx）和模型的係數（model.coef_）以及截距（model.intercept_）計算出的預測奪金時間。這條線代表了模型對訓練資料的擬合程度，可以用來視覺化模型的表現。
plt.xlabel('Year')
plt.ylabel('predicted Time to Win Gold Medal')
plt.show()
