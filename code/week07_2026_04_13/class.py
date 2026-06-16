from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


###
data = pd.read_csv('C:\\Users\\kuoan\\OneDrive\\Desktop\\data\\ECON_3038-00_Data-Analysis-and-Machine-Learning---YU-CHAO-EN\\code\\week7_2026_04_13\\olympicman.csv', header=None).values
print("Type of data:", type(data))
###
###
data = np.array(pd.read_csv('C:\\Users\\kuoan\\OneDrive\\Desktop\\data\\ECON_3038-00_Data-Analysis-and-Machine-Learning---YU-CHAO-EN\\code\\week7_2026_04_13\\olympicman.csv', header=None))
print("Type of data:", type(data))
###

print("len(data):", len(data))
data = np.split(data, [19])
print("data:", data)


###
train = data[0]
print("\ntrain:", train)
trainx = np.transpose([train[:, 0]])#transpose 是用來轉置矩陣的函數，將行和列互換。在這裡，我們將 train[:, 0]（訓練資料的第一列，即年份）轉置成一個列向量，以便於後續的線性回歸模型訓練。這樣做是因為 scikit-learn 的線性回歸模型需要輸入的特徵矩陣是二維的，而不是一維的。
#print("\ntrainx:", trainx)
print("trainx.shape:", trainx.shape) # (19, 1)
traint = train[:, 1]
print("\ntraint:", traint)
###
###
valid = data[1] 
print("\nvalid:", valid)
validx = np.transpose([valid[:, 0]])#explain: valid[:, 0] is the first column of the valid data, which represents the year. We create a new array by stacking the year and the year squared (valid[:,0]**2) as two columns. The np.transpose function is used to switch the rows and columns, so that we have two features (year and year squared) for each validation example.
validt = valid[:,1]#valid[:,1] is the second column of the valid data, which represents the target variable (the time taken to win the gold medal). We assign this column to validt, which will be used as the target variable for the validation set.
###
###
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
rss_valid = np.log10(sum((validt - (model.intercept_ + validx @ model.coef_))**2 ))#截距+斜率 #@:矩陣乘法運算符，表示將 validx（驗證資料的年份特徵）與 model.coef_（模型的係數）進行矩陣乘法，得到模型對驗證資料的預測值。這個預測值再加上 model.intercept_（模型的截距）就得到了模型對驗證資料的總預測值。接著，validt - (model.intercept_ + validx @ model.coef_) 計算了驗證資料的實際值與預測值之間的殘差，**2 將殘差平方化，最後 sum(...) 將所有資料的平方殘差加總，並且 np.log10(...) 將結果取對數，以便更好地比較不同模型的表現。

###
plt.plot( model.intercept_+trainx@model.coef_,'.',color='blue' 
)
#model.intercept_ + trainx @ model.coef_ 是線性回歸模型的預測值，表示根據訓練資料的年份特徵（trainx）和模型的係數（model.coef_）以及截距（model.intercept_）計算出的預測奪金時間。這條線代表了模型對訓練資料的擬合程度，可以用來視覺化模型的表現。
plt.xlabel('Year')
plt.ylabel('predicted Time to Win Gold Medal')
plt.show()


###
plt.plot(trainx, model.intercept_+trainx@model.coef_,color='blue' )#(x軸, y軸, 標記樣式, 顏色) 
#WWI 間隔八年
#model.intercept_ + trainx @ model.coef_ 是線性回歸模型的預測值，表示根據訓練資料的年份特徵（trainx）和模型的係數（model.coef_）以及截距（model.intercept_）計算出的預測奪金時間。這條線代表了模型對訓練資料的擬合程度，可以用來視覺化模型的表現。
plt.xlabel('Year')
plt.ylabel('predicted Time to Win Gold Medal')
plt.show()

print("rss_valid:", rss_valid)
plt.scatter(train[:, 0], train[:, 1], label='Train Data', color='blue')
plt.scatter(valid[:, 0], valid[:, 1], label='Validation Data', color='orange')
plt.plot(train[:, 0], model.intercept_ + trainx @ model.coef_,label='Fitted Line', color='red')
plt.xlabel('Year')
plt.ylabel('Time to Win Gold Medal')
plt.title('Olympic Gold Medal Time vs Year')
plt.legend()#是 Matplotlib 裡用來**顯示圖例（legend）**的指令
plt.show()
