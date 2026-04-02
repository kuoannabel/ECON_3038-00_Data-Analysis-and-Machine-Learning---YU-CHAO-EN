import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# df=pd.read_csv('/kaggle/input/datasets/camnugent/california-housing-prices/housing.csv')
df=pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv')
print("Column names:")
print(df.columns)
print("Shape:")
print(df.shape)
df.head()
print("Median house value:")
print(df.loc[:,'median_house_value'])#loc is used to access a group of rows and columns by labels or a boolean array. iloc is used to access a group of rows and columns by integer position.
print("Total bedrooms:")
print(df.iloc[:, -2])
print("Dataset description:")
print(df.describe())#summary statistics of the dataset
print("Missing values:")
print(df.isnull().sum())

df.pop('ocean_proximity')
# df=df.drop(['ocean_proximity'], axis=1)
df.loc[:, 'total_bedrooms'] = df.loc[:, 'total_bedrooms'].fillna(value=np.mean(df.loc[:, 'total_bedrooms']))
df.isnull().sum()

data=np.array(df, dtype=float);
t=data[:,-1]; X=data[:,:-1];
trainX, validX, traint, validt = train_test_split(X, t, test_size=0.2, shuffle=False)
trainX.shape
t[:16512]==traint

#correlation between each feature and the target variable
corre = []
for i in range(trainX.shape[1]):
    print(np.corrcoef(traint, trainX[:, i])[0,1])
    corre.append(np.corrcoef(traint, trainX[:, i])[0,1])

min_index = np.argmin(corre)
print("\n# The feature with the lowest correlation with the target variable is:", df.columns[min_index])
print("# The correlation coefficient is:", corre[min_index])
df.pop('df.columns[min_index]')#removing the feature with the lowest correlation with the target variable
df.loc[:16511,'median_income']==trainX[:,-1]#correlation between median_income and median_house_value

plt.scatter(trainX[:,-1], traint)
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.show()