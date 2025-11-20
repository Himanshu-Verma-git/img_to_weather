import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Sequential
from sklearn.metrics import classification_report

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# Directories
train_dir = "weather_prediction/weather dataset/train/"
val_dir = "weather_prediction/weather dataset/validation/"
test_dir = "weather_prediction/weather dataset/test/"

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0/255)
valid_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    batch_size=64,
    target_size=(224, 224),
    class_mode="categorical"
)

valid_data = valid_datagen.flow_from_directory(
    val_dir,
    batch_size=64,
    target_size=(224, 224),
    class_mode="categorical"
)

test_data = valid_datagen.flow_from_directory(
    test_dir,
    batch_size=64,
    target_size=(224, 224),
    class_mode="categorical",
    shuffle=False
)

# Modern augmentation block (compatible with TF 2.16+)
data_augmentation = Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="augmentation")

# Model
model = Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    Conv2D(16, 3, activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(32, 3, activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.fit(
    train_data,
    epochs=20,
    validation_data=valid_data,
    callbacks=[early_stopping]
)

# Evaluation
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = test_data.classes

test_loss, test_acc = model.evaluate(test_data)
print("Test accuracy:", test_acc)
print(classification_report(true_labels, predicted_labels))

# Save in the correct format
model.save("weather_prediction/weather.keras")
print("Model saved successfully.")

