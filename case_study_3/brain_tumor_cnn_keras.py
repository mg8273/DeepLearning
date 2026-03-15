# ---------------------------------------------------------
# 1. Mount Google Drive
# ---------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# ---------------------------------------------------------
# 2. Import Libraries
# ---------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# ---------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------
img_size = 224
batch_size = 32

# ---------------------------------------------------------
# 4. Data Preprocessing & Generators
# ---------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    # Optional: Data Augmentation (Variation #3)
    # rotation_range=20,
    # horizontal_flip=True,
    # zoom_range=0.2
)

# Path to the Training folder
train_dir = '/content/drive/MyDrive/archive/Training'

print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("Loading validation data...")
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Show class indices
print("\nClass indices:")
print(train_generator.class_indices)

# ---------------------------------------------------------
# 5. Build CNN Model
# ---------------------------------------------------------

num_classes = 4  # Change to 2 if you want binary classification

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
# Optional: Add Dropout (Variation #2)
# model.add(layers.Dropout(0.5))

model.add(layers.Dense(num_classes, activation='softmax'))

# ---------------------------------------------------------
# 6. Compile and Train Model
# ---------------------------------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nStarting training...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# ---------------------------------------------------------
# 7. Evaluate and Visualize
# ---------------------------------------------------------
loss, accuracy = model.evaluate(val_generator)
print(f"\nValidation Accuracy: {accuracy:.4f}")

# Plot accuracy curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.show()
