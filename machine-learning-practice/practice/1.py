

import numpy as np

# np.random.seed(100)
# print(np.random.rand(10))

# np.random.seed(100)
# print(np.random.rand(10))

# print(np.random.randn(10))
# print(np.random.randint(1, 10, size=(2,3)))


# print(np.array([1, 2, 3, 4, 5]))
# print(np.arange(1, 10, 2))

# print(np.empty((4, 4)))
# print(np.ones((4, 4)))
# print(np.zeros((4, 4)))
# print(np.full((4, 4), 10))

# print(np.eye(4))

# print(np.linspace(1, 10, 5))

# print(np.random.uniform(1., 10., 10))
# print(np.random.normal(size=(3, 4)))
# print(np.random.choice(100, size=(3, 4)))

x = np.arange(10).reshape(2, 5)
print(x)
# print(type(x[1, -1]))

# print(x[:, -1:])

# print(y)
# print(type(y[1, 1, 1]))

# print(y)

# print(y[:, -1:, :])

# ravel, flatten

# copy no
# print(x.ravel())
# print(np.ravel(x)) 

# copy yes
# print(x.flatten())
# print(x.ndim)

print(np.sum(x, axis=1))
print(np.mean(x, axis=1))

# z = np.random.randn(10).reshape(2, 5)
# print(z)
# print(np.any( z > 0 ))
# print(np.all( z > 0 ))

# print(np.where( z > 0, z, 0))

print(np.argmax(x, axis=1))

