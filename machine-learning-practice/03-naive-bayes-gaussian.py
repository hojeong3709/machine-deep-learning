'''
docs
https://ko.wikipedia.org/wiki/%EB%82%98%EC%9D%B4%EB%B8%8C_%EB%B2%A0%EC%9D%B4%EC%A6%88_%EB%B6%84%EB%A5%98
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
'''
'''
https://www.quora.com/What-is-the-difference-between-the-the-Gaussian-Bernoulli-Multinomial-and-the-regular-Naive-Bayes-algorithms
Bernoulli Naive Bayes : 
- It assumes that all our features are binary such that they take only two values. 
- Means 0s can represent “word does not occur in the document” and 1s as "word occurs in the document" .
Multinomial Naive Bayes : 
- Its is used when we have discrete data (e.g. movie ratings ranging 1 and 5 as each rating will have certain frequency to represent).
- In text learning we have the count of each word to predict the class or label.
Gaussian Naive Bayes : 
- Because of the assumption of the normal distribution, Gaussian Naive Bayes is used in cases when all our features are continuous.
- For example in Iris dataset features are sepal width, petal width, sepal length, petal length. 
- So its features can have different values in data set as width and length can vary. 
- We can’t represent features in terms of their occurrences. This means data is continuous. 
- Hence we use Gaussian Naive Bayes here.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
print(dir(iris))
print(iris.data.shape)
# (150, 4)
print(iris.target.shape)
# (150, )

# simple attributes or functions
for name in dir(iris):
    if not callable(getattr(iris, name)):
        print(name, type(getattr(iris, name)))

clf = GaussianNB()
clf_fit = clf.fit(iris.data, iris.target)
y_predict = clf_fit.predict(iris.data)
print(y_predict)
print(confusion_matrix(iris.target, y_predict))
'''
[[50  0  0]
 [ 0 47  3]
 [ 0  3 47]]
'''

y_predict_proba_argmax = np.argmax(clf_fit.predict_proba(iris.data)[[1, 48, 51, 100]], axis=1)
print(y_predict_proba_argmax)

clf2 = GaussianNB(priors=[1/100, 1/100, 98/100])
clf2_fit = clf2.fit(iris.data, iris.target)
y_predict2 = clf2_fit.predict(iris.data)
print(confusion_matrix(iris.target, y_predict2))
'''
[[50  0  0]
 [ 0 33 17]
 [ 0  0 50]]
 '''


