import numpy as np

# 단어 수준의 One-Hot Encoding

samples = ["The cat sat on the mat.", "The dog ate my homework."]

token_index = dict()

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

max_length = 10

results = np.zeros(shape=(len(samples),
                          max_length,
                          max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.

print(results)

# 단어 수준의 해싱기법을 이용한 One-Hot

dimensionality = 1000

results = np.zeros(shape=(len(samples),
                          max_length,
                          dimensionality))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.

print(results)