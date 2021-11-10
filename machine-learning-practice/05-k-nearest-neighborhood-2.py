'''
https://ko.wikipedia.org/wiki/K-%EC%B5%9C%EA%B7%BC%EC%A0%91_%EC%9D%B4%EC%9B%83_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98
https://scikit-learn.org/stable/modules/neighbors.html
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X = iris.data[:, :3]
y = iris.target

# preprocessing
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3)

print(X)
print(X_train)

# # feature scaling
# scaler = StandardScaler()
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# k = 5
# clf = neighbors.KNeighborsClassifier(k)
# clf_fit = clf.fit(X_train, y_train)

# clf_predict = clf_fit.predict(X_test)
# print(confusion_matrix(y_test, clf_predict))
