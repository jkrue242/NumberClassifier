# SIMPLE NEURAL NETWORK FOR HANDWRITTEN NUMBER CLASSIFICATION
# 8/26/2022 - Joseph Krueger
# tutorial from CodeBasics YouTube

import keras
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# set testing and training sets for x and y
(xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data()

# Scale data so all values are between 0 and 1
xTrain = xTrain / 255
xTest = xTest / 255

# reshaping data to be len(xTrain)x784 instead of 28x28
xTrain_flattened = xTrain.reshape(len(xTrain), 28 * 28)
xTest_flattened = xTest.reshape(len(xTest), 28 * 28)

# how to show an image from the dataset
# plt.matshow(xTrain[0])
# plt.show()

# Create neural net- keras.sequential is a stack of layers in neural net
model = keras.Sequential([
    keras.layers.Dense(10, input_shape = (784, ), activation = 'sigmoid')
])

# compile and add optimizers to train more efficiently
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# fit model to training data
model.fit(xTrain_flattened, yTrain, epochs=5)

# evaluate model with test data
yPredicted = model.evaluate(xTest_flattened, yTest)

# predict xTest
model.predict(xTest_flattened)
