import mnist

"""
    using the sequential model
    using to_categorical for one-hot encoding, i.e
            2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    using Adam optimiser
    using softmax at the output layer
    using categorical cross-entropy loss function
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


training_set, testing_set = mnist.data()

train_Images = training_set[0]
train_Lables = training_set[1]

test_Images = testing_set[0]
test_Lables = testing_set[1]

# Build the model
model = Sequential({
    Dense(60, activation='relu', input_shape=(784,)),
    Dense(60, activation='relu'),
    Dense(10, activation='softmax') # output layer
})

# Compile the model
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# Train the model
model.fit(
    train_Images,
    to_categorical(train_Lables),
    epochs = 5,
    batch_size = 32
)

# Evaluate the model.
model.evaluate(
  test_Images,
  to_categorical(test_Lables)
)

# save the model
model.save_weights('model.h5')