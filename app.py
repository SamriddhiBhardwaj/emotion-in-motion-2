import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import wget
import os
import plotly.express as px

st.set_page_config(layout="wide", page_title="Emotion in Motion", initial_sidebar_state="expanded")

EMOTIONS = ['ANGRY', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']

st.title("Emotion in Motion")
st.header("Reads your mind :0")


st.sidebar.markdown(
    f"This [demo](https://github.com/SamriddhiBhardwaj/emotion-in-motion) - Emotion in Motion - uses keras for detecting facial emotion. You can add in your images and test the model on them, or choose from the examples! Images from google."
)
st.sidebar.markdown(
    "Made with 💜 by [Samriddhi Bhardwaj](https://github.com/SamriddhiBhardwaj)"
)


if not os.path.exists("models/emotion_detection_model_for_streamlit.h5"):
    with st.spinner("Loading model..."):
        os.system("wget --no-check-certificate -O models/emotion_detection_model_for_streamlit.h5 \"https://www.dropbox.com/s/072b5vf4b33bu1l/emotion_detection_model_for_streamlit.h5\"")

model = tf.keras.models.load_model("models/emotion_detection_model_for_streamlit.h5")

f = st.file_uploader("Upload an Image")

images = { "None":"None", "Angry": "images/angry.jpeg", "Happy": "images/happy.jpeg", "Sad": "images/sad.jpeg", "Neutral": "images/neutral.jpeg","Surprised":"images/surprise.jpeg"}
image_options = list(images.keys())
image_choice = st.selectbox("or select a demo image", image_options)
image_location = images[image_choice]

file_bytes = None

if image_choice == "None":
    if f is not None: 
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
else:
    with open(image_location, 'rb') as image_file:
        if image_file is not None:
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)

if file_bytes is not None:
  # file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  st.image(image, channels="BGR")

  resized = cv2.resize(image, (48, 48), interpolation=cv2.INTER_LANCZOS4)

  gray_1d = np.mean(resized, axis=-1)
  gray = np.zeros_like(resized)
  gray[:,:,0] = gray_1d
  gray[:,:,1] = gray_1d
  gray[:,:,2] = gray_1d

  normalized = gray/255
  
  model_input = np.expand_dims(normalized,0)
  scores = model.predict(model_input).flatten()

  df = pd.DataFrame()
  df["Emotion"] = EMOTIONS
  df["Scores"] = scores
  st.write(px.bar(df, x='Emotion', y='Scores', title="Model scores for each emotion"))

  prediction = EMOTIONS[scores.argmax()]
  st.write(f"Prediction: {prediction}")


