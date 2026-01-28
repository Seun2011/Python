import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam'
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining the model...\n")
model.fit(x_train, y_train, epochs=5)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("\nTest Accuracy:", test_accuracy)

predictions = model.predict(x_test)

index = 2

plt.imshow(x_test[index], cmap='gray')
plt.title(f"Predicted Digit: {np.argmax(predictions[index])}")
plt.axis('off')
plt.show()