from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#https://youtu.be/Un5FvZzvRS4?si=-nX6bS1jeCUFX6m4


###
data = pd.read_csv('C:\\Users\\kuoan\\OneDrive\\Desktop\\data\\ECON_3038-00_Data-Analysis-and-Machine-Learning---YU-CHAO-EN\\code\\week7_2026_04_13\\olympicman.csv', header=None).values
print("Type of data:", type(data))
###
###
data = np.array(pd.read_csv('C:\\Users\\kuoan\\OneDrive\\Desktop\\data\\ECON_3038-00_Data-Analysis-and-Machine-Learning---YU-CHAO-EN\\code\\week7_2026_04_13\\olympicman.csv', header=None))

data[:, 0] = (data[:, 0] - 1896) / 4#standardize year, 1896 是第一屆奧運會的年份，將年份減去 1896 可以將年份轉換為從第一屆奧運會開始的相對年份。除以 4 是因為奧運會每四年舉行一次，這樣做可以將年份標準化，使得不同年份之間的差距更容易比較和分析。
print(data[:, 0])#印出標準化後的年份
#
print("Type of data:", type(data))
###

###
print("len(data):", len(data))

data = np.split(data, [19, 23])#將 data 分割成三個部分：前 19 筆作為訓練資料，接下來的 4 筆作為驗證資料，剩下的作為測試資料。這樣做是為了在訓練模型時能夠使用訓練資料進行學習，並使用驗證資料來評估模型的表現，最後使用測試資料來檢驗模型的泛化能力。
print("data:", data)



train = data[0]
print("\ntrain:", train)
trainx = np.transpose([train[:, 0], train[:, 0]**2])#transpose 是用來轉置矩陣的函數，將行和列互換。在這裡，我們將 train[:, 0]（訓練資料的第一列，即年份）轉置成一個列向量，以便於後續的線性回歸模型訓練。這樣做是因為 scikit-learn 的線性回歸模型需要輸入的特徵矩陣是二維的，而不是一維的。
#print("\ntrainx:", trainx)
print("trainx.shape:", trainx.shape) # (19, 2)
traint = train[:, 1]
print("\ntraint:", traint)
###

###
valid = data[1] 
print("\nvalid:", valid)
validx = np.transpose([valid[:, 0], valid[:, 0]**2 ])  
#explain: valid[:, 0] is the first column of the valid data, which represents the year. We create a new array by stacking the year and the year squared (valid[:,0]**2) as two columns. The np.transpose function is used to switch the rows and columns, so that we have two features (year and year squared) for each validation example.
print("\n\nvalidx:", validx)

validt = valid[:,1]#valid[:,1] is the second column of the valid data, which represents the target variable (the time taken to win the gold medal). We assign this column to validt, which will be used as the target variable for the validation set.
###

######
###
from sklearn.linear_model import LinearRegression
###

#Polynomial regression 之所以能變曲線，是因為把 x 映射成 (x, x², x³...)，讓「線性模型在高維空間擬合直線，但在原空間呈現曲線」。
trainloss = []
logvalidloss = []
for i in range(1, 9):
    trainlist = []; 
    validlist = [];
    for j in range(1, i+1):#第一次:一次函數,第二次:二次函數,第三次:三次函數...以此類推 第i次: j次函數(j=1,2,...,i)
        trainlist.append(train[:, 0]**j)
        trainx = np.transpose(trainlist)
        validlist.append(valid[:, 0]**j)#後八筆 驗證資料
        validx = np.transpose(validlist)

    print("### trainx.shape:", trainx.shape,end=" ###") #(19, i)
    #trainx = np.transpose(trainlist)

    model = LinearRegression()
    model.fit(trainx, traint)#fit 是 LinearRegression 模型的方法，用於訓練模型。它接受兩個參數：trainx 是訓練資料的特徵（年份），traint 是訓練資料的目標變數（奪金時間）。通過調用 fit 方法，模型會學習到特徵與目標變數之間的關係，從而找到最佳的線性回歸方程來預測奪金時間。
    #coef_ 是 LinearRegression 模型的屬性，包含了模型的係數（斜率）。在這裡，它將返回一個包含年份特徵的係數值，表示每增加一年，預測的奪金時間會增加或減少多少。這個值可以用來解釋年份對奪金時間的影響程度。
    print("\nmodel.coef_斜率:", model.coef_)#每個 feature 的權重,在「單一變數線性回歸」時：是「那一條直線的 slope"
    print("model.intercept_截距:", model.intercept_,end='\n\n')
    ###

    ###
    rss = sum((traint - (model.intercept_ + trainx@model.coef_))**2 )#explain: # 計算 Residual Sum of Squares (RSS, 殘差平方和)
    # model.intercept_ + trainx @ model.coef_
    # → 模型預測值 ŷ = β0 + Xβ 
    # # traint - (預測值)
    # → 殘差 (實際值 - 預測值)
    # **2
    # → 將殘差平方化，確保所有的殘差都是正值，並且放大較大的殘差，以便更好地反映模型的預測誤差。
    # sum(...)
    # → 將所有資料的平方殘差加總
    trainloss.append(rss)
    print("rss:", rss)
    rss_valid = np.log10(sum((validt - (model.intercept_ + validx @ model.coef_))**2 ))#截距+斜率 #@:矩陣乘法運算符，表示將 validx（驗證資料的年份特徵）與 model.coef_（模型的係數）進行矩陣乘法，得到模型對驗證資料的預測值。這個預測值再加上 model.intercept_（模型的截距）就得到了模型對驗證資料的總預測值。接著，validt - (model.intercept_ + validx @ model.coef_) 計算了驗證資料的實際值與預測值之間的殘差，**2 將殘差平方化，最後 sum(...) 將所有資料的平方殘差加總，並且 np.log10(...) 將結果取對數，以便更好地比較不同模型的表現。
    print("rss_valid:", rss_valid, end='\n\n')
    logvalidloss.append(rss_valid)
    print("###\n\n")
plt.plot(range(1, 9), trainloss, marker='o', label='Train Loss')
plt.xlabel('Polynomial Degree')
plt.ylabel('Train Loss (RSS)')
plt.title('Train Loss vs Polynomial Degree')
plt.legend()
plt.show()
###
plt.plot(range(1, 9), logvalidloss, marker='o', label='Log Validation Loss')

plt.xlabel('Polynomial Degree')
plt.ylabel('Log Validation Loss (log10(RSS))')
plt.title('Log Validation Loss vs Polynomial Degree')
plt.legend()
plt.show()
###




 

