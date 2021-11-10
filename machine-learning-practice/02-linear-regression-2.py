'''
docs
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
'''

import numpy as np
import statsmodels.api as sm

# Preprocessing
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([6, 8, 9, 11])
# y = np.dot(X, np.array([1, 2])) + 3
# y = X * W + b

print(X.shape)
print(y.shape)

# Model Create and Fit
X = sm.add_constant(X)
model = sm.OLS(y, X)
model_fit = model.fit()
print(model_fit.summary())
print(model_fit.params)
print(model_fit.pvalues)
# Predict