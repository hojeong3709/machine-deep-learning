import numpy as np
import pandas as pd

s1 = pd.Series(data=[1, 1, 1, 2, 2, 3, 4, 5, 5, 5, np.NaN])
# n1 = np.array([1, 1, 1, 2, 2, 3, 4, 5, 5, 5, np.NaN])
# print(s1)
# print(n1)

print(s1.index)
print(s1.index.values)
# print(s1.values)
# print(s1.shape)
# print(s1.count())
# print(np.sum(s1.isna()))
# print(s1.unique())
# print(s1.value_counts())

# numpy와 다른점
# print(s1.mean())
# print(np.mean(n))

# s2 = pd.Series(data=np.arange(100, 105), index=['a', 'b', 'c', 'd', 'e'])
# print(s2)

# s2['a'] = 200
# s2['k'] = 300

# print(s2)

# print(s2.drop('k'))
# print(s2)

# print(s2.drop('k', inplace=True))
# print(s2)

s3 = pd.Series(data=np.arange(100, 105), index=[1, 2, 3, 4, 5])
print(s3)
print(s3[1:3])

s4 = pd.Series(data=np.arange(100, 105), index=['a', 'b', 'c', 'd', 'e'], dtype=np.int64)
print(s4)
print(s4['a':'c'])