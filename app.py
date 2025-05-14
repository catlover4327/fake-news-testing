
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
model = tf.keras.models.load_model("lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Streamlit UI
st.title("Fake News Detection App")
title = st.text_input("Enter the News Title")
text = st.text_area("Enter the News Content")

if st.button("Predict"):
    input_text = title + " " + text
    sequence = tokenizer.texts_to_sequences([input_text])
    padded = pad_sequences(sequence, maxlen=500)
    prediction = model.predict(padded)[0][0]
    label = "Real News" if prediction >= 0.5 else "Fake News"
    st.write(f"🧠 Prediction: **{label}**")
