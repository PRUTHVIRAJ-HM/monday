
# Deep Neural Network for Classification
# Example: MNIST handwritten digits
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
# 1. Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 2. Normalize the data (0–255 -> 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0
# 3. One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# 4. Build the Deep Neural Network model
model = Sequential([
Flatten(input_shape=(28, 28)), # Flatten 28x28 images into 784 input features
Dense(128, activation='relu'), # Hidden layer 1
Dense(64, activation='relu'), # Hidden layer 2
Dense(10, activation='softmax') # Output layer (10 classes)
])
# 5. Compile the model
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])
# 6. Train the model
print("Training the Deep Neural Network...")
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
# 7. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
