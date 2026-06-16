import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

data=np.array(pd.read_csv('c:\\temp\\data_knnexample.csv'))
len(data)
X=data[:,0:2]; X.shape
t=data[:,2]
plt.scatter(X[:,0][t==0], X[:,1][t==0], marker='o', c='k')
plt.scatter(X[:,0][t==1], X[:,1][t==1], marker='s', s=40, facecolors='none', edgecolors='k')

Xv=np.linspace(-3,6,101);
Yv=np.linspace(-3,6,101);
K=3; classes=[];
for y in Yv:
    for x in Xv:
        dists = np.sum((X - [x, y])**2, axis=1);
        votes = list(t[dists<=sorted(dists)[K-1]])#K-1 because of 0-based indexing
        if votes.count(0)==votes.count(1):
            classes.append(random.choice([0,1]))
        elif votes.count(0)>votes.count(1):
            classes.append(0)
        else:
            classes.append(1)
classes=np.array(classes).reshape(len(Yv),len(Xv))
classes.shape
plt.contour(Xv, Yv, classes, [0.5])
plt.scatter(X[:,0][t==0], X[:,1][t==0], marker='o', c='k')
plt.scatter(X[:,0][t==1], X[:,1][t==1], marker='s', s=40, facecolors='none', edgecolors='k')


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(1)
model.fit(X, t)
Xv=np.linspace(-3,6,101);
Yv=np.linspace(-3,6,101);
K=1; classes2=[];
for y in Yv:
    for x in Xv:
        classes2.append(model.predict([[x, y]]))
classes2=np.array(classes2).reshape(len(Yv),len(Xv))
classes2.shape
plt.contour(Xv, Yv, classes2, [0.5])
plt.contour(Xv, Yv, classes, [0.5])
plt.scatter(X[:,0][t==0], X[:,1][t==0], marker='o', c='k')
plt.scatter(X[:,0][t==1], X[:,1][t==1], marker='s', s=40, facecolors='none', edgecolors='k')

XM, YM=np.meshgrid(Xv, Yv)
XM.reshape(101*101,)
classes.reshape(101*101,)
classes2.reshape(101*101,)
XM[classes!=classes2]