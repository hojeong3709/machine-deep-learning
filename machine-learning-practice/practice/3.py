#%%
import numpy as np

# Lotto
print(np.random.choice(np.arange(1, 46), size=6, replace=False))


# %%
# monte carlo method

import matplotlib.pyplot as plt

total_cnt = int(1e7)

points = np.random.rand(total_cnt, 2)
print(np.sum(points ** 2, axis=1))

print(4 * np.sum(np.sum(points ** 2, axis=1) < 1) / total_cnt)



# plt.plot(x, y)


# %%
