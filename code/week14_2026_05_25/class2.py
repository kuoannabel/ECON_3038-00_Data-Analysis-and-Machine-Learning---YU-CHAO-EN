import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

X_train=[[1.14, 0.59], [0.03, -0.75], [5.14, 4.59], [4.03, 3.24]]
y_train=np.array([-1, -1, 1, 1])
# X_train=pd.read_csv('c:\\temp\\data_svmhard.csv')
# y_train=np.concatenate(([-1 for i in range(int(len(X_train)/2))], [1 for i in range(int(len(X_train)/2))]), axis=0)

# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# sc.fit(X_train)
# X_train=sc.transform(X_train)
#解釋: 在這裡，我們定義了訓練數據集X_train和對應的標籤y_train。X_train是一個包含四個樣本的二維列表，每個樣本有兩個特徵。y_train是一個一維的NumPy數組，包含每個樣本的類別標籤，其中-1表示第一類，1表示第二類。

from sklearn.svm import SVC, LinearSVC
svm=SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
svm.coef_[0]
svm.intercept_[0]
svm.support_vectors_
-svm.coef_[0,0]/svm.coef_[0,1]
svm1=LinearSVC()
svm1.fit(X_train, y_train)
svm1.intercept_[0]
-svm1.coef_[0,0]/svm1.coef_[0,1]
slopes=np.array([-svm.coef_[0,0]/svm.coef_[0,1], -svm1.coef_[0,0]/svm1.coef_[0,1]])
print(slopes)
