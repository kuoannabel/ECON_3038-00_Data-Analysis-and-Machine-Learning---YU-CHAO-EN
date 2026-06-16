import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

Y=np.array(pd.read_csv('c:\\temp\\data_pca.csv', header=None))
plt.scatter(Y[:,0], Y[:,1])
plt.show()

N=len(Y)
Y.shape
Y[0]

from sklearn.decomposition import PCA

pca=PCA(n_components=7)
pca.fit(Y)
pca.explained_variance_
plt.bar(range(len(pca.explained_variance_)), pca.explained_variance_)
plt.show()
pca.explained_variance_/sum(pca.explained_variance_)
pca.explained_variance_ratio_

X_new=pca.transform(Y)
plt.scatter(X_new[:,0], X_new[:,1], marker='o')
plt.show()

# By eigenvalues
Y=Y-np.mean(Y,axis=0)
C=Y.T@Y/len(Y)
lam, w = np.linalg.eig(C)
plt.bar(range(len(lam)), lam)
plt.show()

X=Y@w[:,:2]
plt.scatter(X[:,0], X[:,1])
plt.show()