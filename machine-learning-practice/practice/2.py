#%%
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline

x = np.linspace(0, 10, 11)
y = x ** 2 + x + 2 + np.random.randn(11)

print(x)
print(y)

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("X-Y relation")
plt.grid(True)
plt.xlim(0, 20)
plt.ylim(0, 200)
plt.plot(x, y, color="black", linestyle="--", linewidth=3, marker="^")

# %%


plt.subplot(2, 2, 1)
plt.plot(x, y, "r")


plt.subplot(2, 2, 2)
plt.plot(x, y, "g")


plt.subplot(2, 2, 3)
plt.plot(x, y, "b")


plt.subplot(2, 2, 4)
plt.plot(x, np.exp(x), "r")

# %%

data = np.random.randint(1, 100, size=200)
plt.hist(data, bins=10, alpha=0.3)
plt.xlabel("data")
plt.ylabel("count")
plt.grid(True)
# %%
