'''
https://ko.wikipedia.org/wiki/K-%EC%B5%9C%EA%B7%BC%EC%A0%91_%EC%9D%B4%EC%9B%83_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98
https://scikit-learn.org/stable/modules/neighbors.html
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

k = 5
clf = neighbors.KNeighborsClassifier(k)
clf_fit = clf.fit(X, y)

clf_predict = clf_fit.predict(X)
print(confusion_matrix(y, clf_predict))
