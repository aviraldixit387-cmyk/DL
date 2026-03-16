import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

print("Preparing dataset...")

# Image size and batch size
img_size = 224
batch_size = 32

# -----------------------------
# Data Preprocessing
# -----------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

print("Dataset loaded successfully")

# -----------------------------
# CNN Model
# -----------------------------

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(128,activation='relu'))

model.add(layers.Dense(2,activation='softmax'))

# -----------------------------
# Compile Model
# -----------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training started...")

# -----------------------------
# Train Model
# -----------------------------

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# -----------------------------
# Evaluate Model
# -----------------------------

loss, accuracy = model.evaluate(val_generator)

print("Validation Accuracy:", accuracy)

# -----------------------------
# Plot Training Accuracy
# -----------------------------

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.title("Training vs Validation Accuracy")

plt.show()
