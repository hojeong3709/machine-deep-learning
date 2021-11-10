import numpy as np
import string

# 문자 수준의 One-Hot Encoding

samples = ["The cat sat on the mat.", "The dog ate my homework."]

characters = string.printable
print(characters)
token_index = dict(zip(characters, range(1, len(characters) + 1)))
print(token_index)

max_length = 50

results = np.zeros(shape=(len(samples),
                          max_length,
                          max(token_index.values()) + 1))

results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        print(character)
        index = token_index.get(character)
        results[i, j, index] = 1.


print(results)