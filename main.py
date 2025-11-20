import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0/255)
train_dir = "weather_prediction/weather dataset/train/"
train_data = train_datagen.flow_from_directory(
    train_dir,
    batch_size=64,
    target_size=(224, 224),
    class_mode="categorical"
)

model = tf.keras.models.load_model("weather_prediction/weather.keras")

def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array / 255.0


def predict(img_path):
    processed = load_and_preprocess_image(img_path)
    preds = model.predict(processed)
    idx = np.argmax(preds, axis=1)[0]

    class_labels = list(train_data.class_indices.keys())
    return class_labels[idx]


st.title("Weather Predictor")
img = st.file_uploader("Choose a file")

if img:
    st.image(img, caption="Uploaded Image")
    with open("temp.jpg", "wb") as f:
        f.write(img.read())
    result = predict("temp.jpg")
    st.write(result)

