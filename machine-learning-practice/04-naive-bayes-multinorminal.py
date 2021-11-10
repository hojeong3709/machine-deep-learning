'''
docs
https://ko.wikipedia.org/wiki/%EB%82%98%EC%9D%B4%EB%B8%8C_%EB%B2%A0%EC%9D%B4%EC%A6%88_%EB%B6%84%EB%A5%98
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

X = np.random.randint(0, 5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])

clf = MultinomialNB()
clf_fit = clf.fit(X, y)
clf_predict = clf_fit.predict(X)

# slice for dimension not index
print(X[2:3].shape)
# (1, 100)
# print(X[2].shape)
# (100, )

print(clf_fit.predict(X[2:3]))
print(clf_fit.predict_proba(X[2:3]))
print(confusion_matrix(y, clf_predict))

clf2 = MultinomialNB(class_prior=[0.1, 0.5, 0.1, 0.1, 0.1, 0.1])
clf2_fit = clf2.fit(X, y)
clf2_predict = clf2.predict(X)

print(clf2_fit.predict(X[2:3]))
print(clf2_fit.predict_proba(X[2:3]))
print(confusion_matrix(y, clf2_predict))
