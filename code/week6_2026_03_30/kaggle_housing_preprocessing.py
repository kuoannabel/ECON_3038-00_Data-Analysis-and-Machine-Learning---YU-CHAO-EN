import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# df=pd.read_csv('/kaggle/input/datasets/camnugent/california-housing-prices/housing.csv')
df=pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv')
df.shape
df.head()
df.isnull().sum()  # there are missing data in the variable 'total_bedrooms'

df.pop('ocean_proximity')  # delete the non-numerical column
# df=df.drop(['ocean_proximity'], axis=1)
df.loc[:, 'total_bedrooms'] = df.loc[:, 'total_bedrooms'].fillna(value=np.mean(df.loc[:, 'total_bedrooms'])) # fill in the empty cell with the mean of 'total_bedrooms'
df.isnull().sum()  # Now there is no missing data

data=np.array(df, dtype=float);  # transfer DataFrame data to an array so that we can use array index
t=data[:,-1]; X=data[:,:-1];
trainX, validX, traint, validt = train_test_split(X, t, test_size=0.2, shuffle=False)
trainX.shape