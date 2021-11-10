
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

print(os.getcwd())

train_data = pd.read_csv("data/train.csv", index_col="PassengerId", usecols=["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Embarked"])
# print(train_data.shape)
# print(train_data.head())

# print(train_data["Survived"])
# print(train_data[["Survived", "Name"]])
# print(train_data[0:2])

# print(train_data.loc[0]) # error
# print(train_data.loc[0:5])
# print(train_data.loc[[100, 200, 300]])

# print(train_data.iloc[0])
# print(train_data.iloc[0:5])
# print(train_data.loc[[100, 200, 300]])

# print(train_data.loc[0:5])
# print(train_data.loc[1:5, ["Survived", "Pclass"]])
# print(train_data.loc[[1, 5], ["Survived", "Pclass"]])

# class_condition = train_data["Pclass"] == 1
# age_condition = (train_data["Age"] >= 30) & (train_data["Age"] < 40)
# mask = class_condition & age_condition
# print(train_data[mask])

# # column insert at the end of columns
# train_data["Age_Double"] = train_data["Age"] * 2
# print(train_data)

# # column insert with index
# train_data.insert(1, "Age_Triple", train_data["Age"] * 3)
# print(train_data)

# # column delete
# train_data.drop(["Age_Double", "Age_Triple"], axis=1, inplace=True)
# print(train_data)

# # NaN
# print(np.sum(train_data.isna()))
# print(train_data.dropna())
# print(train_data.dropna(axis=1))
# print(train_data.dropna(subset=["Sex", "Age"]))

# # 생존자 나이평균
# mean1 = train_data[train_data["Survived"] == 1]["Age"].mean()

# # 사망자 나이평균
# mean2 = train_data[train_data["Survived"] == 0]["Age"].mean()

# print(mean1, mean2)
# # print(train_data["Age"].fillna(train_data["Age"].mean()))
# print(train_data[train_data["Survived"] == 1].fillna(mean1))
# print(train_data[train_data["Survived"] == 1].fillna(mean2))

# # 숫자형 데이터 범주화
# print(train_data["Age"])

# 20대/30대/40대 ...
# import math
# def age_categorize(age):
#     if math.isnan(age):
#         return -1
#     return math.floor(age / 10) * 10

# print(train_data["Age"].apply(age_categorize))

# set_index, reset_index
# print(train_data)
# print(train_data.set_index("Pclass"))
# print(train_data.reset_index())
# print(train_data.set_index("Age").groupby(age_categorize).mean())

# # 범주형 데이터 전처리
# print(pd.get_dummies(train_data, columns=["Pclass", "Sex", "Embarked"], drop_first=False))
# # class 한개를 몰라도 추측할 수 있으므로 Feature 개수를 줄이기 위해 drop_first 사용
# print(pd.get_dummies(train_data, columns=["Pclass", "Sex", "Embarked"], drop_first=True))

# groupby
# class_group = train_data.groupby(["Pclass", "Sex"])
# # print(class_group.groups)
# print(class_group.count())
# # print(class_group.max())
# print(class_group.mean())
# print(class_group.mean().loc[(2, "female")])
# print(train_data.set_index(["Pclass", "Sex", "Embarked"]).groupby(level=[0, 2]).mean())
# print(train_data.set_index(["Pclass", "Sex"]).groupby(level=[0, 1]).aggregate([np.mean, np.sum, np.max]))

# transform
# print(train_data.groupby("Pclass").mean())
# print(train_data.groupby("Pclass").transform(np.mean))

# print(train_data.groupby(["Pclass", "Sex"]).mean())
# print(train_data.groupby(["Pclass", "Sex"]).transform(np.mean))


# pivot table
# train_data = train_data.reset_index().drop(columns=["PassengerId", "Pclass", "Sex", "Survived"], axis=1)
# print(train_data.head())
# print(np.sum(train_data.isna()))

# train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())
# train_data = train_data.dropna()
# print(np.sum(train_data.isna()))

# print(train_data.pivot("Name", "Embarked"))
# print(pd.pivot_table(train_data, index="Name", columns="Embarked", aggfunc=np.mean))




