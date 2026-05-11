import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
#https://youtu.be/eRRt92ZISvI?si=lfl6ulMgAsgZLbGv

# df=pd.read_csv('/kaggle/input/datasets/camnugent/california-housing-prices/housing.csv')
df=pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv')
print("df.shape:", df.shape)
print("df.head():", df.head()) 
print("\ndf.isnull().sum():", df.isnull().sum())  # there are missing data in the variable 'total_bedrooms'

df.pop('ocean_proximity')  # delete the non-numerical column
print("\ndf.shape after deleting 'ocean_proximity':", df.shape)

# df=df.drop(['ocean_proximity'], axis=1)
df.loc[:, 'total_bedrooms'] = df.loc[:, 'total_bedrooms'].fillna(value=np.mean(df.loc[:, 'total_bedrooms'])) # fill in the empty cell with the mean of 'total_bedrooms'

print("isnull().sum() after filling missing values:", df.isnull().sum())  # Now there is no missing data

print("\ndf.isnull().sum() after filling missing values:", df.isnull().sum())  # Now there is no missing data



data=np.array(df, dtype=float);  # transfer DataFrame data to an array so that we can use array index
print("\ndata", data)

t=data[:,-1]; X=data[:,:-1];
trainX, validX, traint, validt = train_test_split(X, t, test_size=0.2, shuffle=False, random_state = 234)#last 20% data for validation, first 80% data for training, shuffle=False means the data will not be shuffled before splitting random_state = 234 is used to ensure that the split of the data is reproducible. By setting a specific random state, you can ensure that the same split of the data will occur each time you run the code, which is important for consistency in training and evaluating machine learning models.

print("trainX.shape:", trainX.shape)





print("## feature selection ##")
###
from sklearn.linear_model import LinearRegression
###


trainloss = []
logvalidloss = []
for i in range(8):
    trainlist = []
    validlist = []
    for j in range(8):#第一次:一次函數,第二次:二次函數,第三次:三次函數...以此類推 第i次: j次函數(j=1,2,...,i)
        if j != i:
            trainlist.append(trainX[:,j])
            trainx = np.transpose(trainlist)
        
        
            validlist.append(validX[:, j])#第一次:一次函數,第二次:二次函數,第三次:三次函數...以此類推 第i次: j次函數(j=1,2,...,i)
            validx = np.transpose(validlist)

    print("### trainx.shape:", trainx.shape,end=" ###") #(19, i)
    #trainx = np.transpose(trainlist)

    model = LinearRegression()
    model.fit(trainx, traint)#fit 是 LinearRegression 模型的方法，用於訓練模型。它接受兩個參數：trainx 是訓練資料的特徵（年份），traint 是訓練資料的目標變數（奪金時間）。通過調用 fit 方法，模型會學習到特徵與目標變數之間的關係，從而找到最佳的線性回歸方程來預測奪金時間。
    #coef_ 是 LinearRegression 模型的屬性，包含了模型的係數（斜率）。在這裡，它將返回一個包含年份特徵的係數值，表示每增加一年，預測的奪金時間會增加或減少多少。這個值可以用來解釋年份對奪金時間的影響程度。
    print("\nmodel.coef_ 斜率:", model.coef_)#每個 feature 的權重,在「單一變數線性回歸」時：是「那一條直線的 slope"
    print("\nmodel.intercept_ 截距:", model.intercept_,end='\n\n')
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
    
    
plt.plot(range(8), trainloss, marker='o', label='Train Loss')
plt.xlabel('Polynomial Degree')
plt.ylabel('Train Loss (RSS)')
plt.title('Train Loss vs Polynomial Degree')
plt.legend()
plt.show()
###
plt.plot(range(8), logvalidloss, marker='o', label='Log Validation Loss')

plt.xlabel('Polynomial Degree')
plt.ylabel('Log Validation Loss (log10(RSS))')
plt.title('Log Validation Loss vs Polynomial Degree')
plt.legend()
plt.show()
###

print("trainloss:", trainloss, end='\n\n')
print("logvalidloss:", logvalidloss, end='\n\n')

print("### Train Loss Summary ###")
print("min trainloss:", min(trainloss), end='\n')
idex = trainloss.index(min(trainloss))#找出 trainloss 中最小值的索引位置

print("data point index min trainloss:", idex, end='\n\n')
feature_names = df.columns[:-1]  # 去掉 target
for i in range(8):
    print( i, feature_names[i], "train loss:", trainloss[i], end='\n')


print("\n* removed feature:", feature_names[idex], "\ntrain loss:", trainloss[idex], end='\n\n')
print("### end ###\n\n")

print("### Validation Loss Summary ###")
print("min logvalidloss:", min(logvalidloss), end='\n')
idex_valid = logvalidloss.index(min(logvalidloss))#找出 logvalidloss 中最小值的索引位置
print("data point index min logvalidloss:", idex_valid, end='\n\n')
for i in range(8):
    print( i, feature_names[i], "log validation loss:", logvalidloss[i], end='\n')  
    
print("\n* removed feature:", feature_names[idex_valid], "\nlog validation loss:", logvalidloss[idex_valid],"\nround log validation loss:", np.round(logvalidloss[idex_valid], 3), end='\n\n')#np.round(logvalidloss[idex_valid], 3) 是將 logvalidloss[idex_valid] 的值四捨五入到小數點後三位。這樣做的目的是為了更清晰地展示 log validation loss 的數值，尤其是在比較不同模型的表現時，過多的小數位可能會讓數據看起來過於複雜，而四捨五入可以使結果更易於理解和比較。
print("### end ###\n\n")




 

