import numpy as np
import pandas as pd


# df1 = pd.DataFrame({"key" : np.arange(10), "value" : np.random.randn(10)})
# df2 = pd.DataFrame({"key2" : np.arange(100, 110), "value2" : np.random.rand(10)})

# df3 = pd.concat([df1, df2], ignore_index=True, axis=1)
# print(df3)

customer = pd.DataFrame({
    'customer_id' : np.arange(6), 
    'name' : ['철수'"", '영희', '길동', '영수', '수민', '동건'], 
    '나이' : [40, 20, 21, 30, 31, 18]})

orders = pd.DataFrame({
    'customer_id' : [1, 1, 2, 2, 2, 3, 3, 1, 4, 9], 
    'item' : ['치약', '칫솔', '이어폰', '헤드셋', '수건', '생수', '수건', '치약', '생수', '케이스'], 
    'quantity' : [1, 2, 1, 1, 3, 2, 2, 3, 2, 1]})

# print(pd.merge(customer, orders, on="customer_id", how="inner"))
# print(pd.merge(customer, orders, on="customer_id", how="left"))
# print(pd.merge(customer, orders, on="customer_id", how="right"))
# print(pd.merge(customer, orders, on="customer_id", how="outer"))

# customer = customer.set_index("customer_id")
# orders = orders.set_index("customer_id")

# print(pd.merge(customer, orders, left_index=True, right_index=True))


print(customer.head())
print(orders.head())

# 가장 많이 팔린 아이템은?
print(pd.merge(customer, orders, on="customer_id", how="inner").groupby('item').sum().sort_values(ascending=False, by='quantity'))

# 영희가 가장 많이 구매한 아이템은?
print(pd.merge(customer, orders, on="customer_id", how="inner").groupby(['name', 'item']).sum())
print(pd.merge(customer, orders, on="customer_id", how="inner").groupby(['name', 'item']).sum().loc["영희", "quantity"])