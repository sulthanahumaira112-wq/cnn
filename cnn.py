import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(X_train,y_train),(X_test,y_test)=mnist.load_data()

print(X_train.shape)

plt.imshow(X_train[0],cmap="gray")
plt.title(f"original shape:{y_train[0]}")
plt.show()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)

print(X_train.shape)

datagen=ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
ori=X_train[0]
aug=datagen.random_transform(ori)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(ori.squeeze(),cmap="gray")
plt.title("original")
plt.subplot(1,2,2)
plt.imshow(aug.squeeze(),cmap="gray")
plt.title("augmented")
plt.show()

model=Sequential()
model.add(Conv2D(
    32,
    (3,3),
    activation="relu",
    input_shape=(28,28,1)
))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

batch_size = 32

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    epochs=5,
    steps_per_epoch=len(X_train) // batch_size
)

pred=model.predict(X_test[0:1])
print(pred)
prediction=np.argmax(pred[0])
print(f"actual:{y_test[0]},predicted:{prediction}")
plt.imshow(X_test[0].squeeze(), cmap="gray")
plt.title(f"Actual: {y_test[0]} | Predicted: {prediction}")
plt.show()