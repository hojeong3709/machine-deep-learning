'''
docs
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
'''

import numpy as np
from sklearn.linear_model import LinearRegression

# Preprocessing
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([6, 8, 9, 11])
# y = np.dot(X, np.array([1, 2])) + 3
# y = X * W + b

print(X.shape)
print(y.shape)

# Model Create and Fit
model = LinearRegression()
model_fit = model.fit(X, y)
# R2 score
print(model_fit.score(X, y))
# coefficient
print(model_fit.coef_)
# bias
print(model_fit.intercept_)

# Predict
print(model_fit.predict(np.array([[3, 5]])))


