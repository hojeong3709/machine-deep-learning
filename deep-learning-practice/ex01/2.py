from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(type(train_data))
print(train_data)
print(train_data.shape)
print(type(train_labels))
print(train_labels)
print(train_labels.shape)

print(type(test_data))
print(test_data.shape)

# word_index = imdb.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])

import exNP as np

def vectorize_sequence(sequences, dimension=10000):
    results = exNP.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

y_train = exNP.asarray(train_labels).astype('float32')
y_test = exNP.asarray(test_labels).astype('float32')

print(x_train[0])
print(type(x_train[0]))
print(x_train[0].shape)

print(y_train)
print(type(y_train))
print(y_train.shape)
print(y_train.ndim)
print(y_train[0])


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=5,
                    batch_size=512,
                    validation_data=(x_val, y_val))

import matplotlib.pyplot as plt
history_dict = history.history

print(history_dict)

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

# loss plot
# plt.plot(epochs, loss, 'bo', label="Training loss")
# plt.plot(epochs, val_loss, 'b', label="Validation loss")
# plt.title("Training and Validation loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
#
# plt.show()

# acc plot
plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label="Training acc")
plt.plot(epochs, val_acc, 'b', label="Validation acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
