import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("lstm_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Set constants
MAX_LENGTH = 500

# Text preprocessing (must match the training preprocessing)
import re

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text

# Streamlit UI
st.title("📰 Fake News Detection with LSTM")
st.write("Enter the **title** and **content** of a news article to check if it's real or fake.")

title = st.text_area("📝 Title")
content = st.text_area("📜 Content")

if st.button("🔍 Predict"):
    if not title.strip() and not content.strip():
        st.warning("Please enter a title and/or content.")
    else:
        full_text = clean_text(title + " " + content)
        seq = tokenizer.texts_to_sequences([full_text])
        padded = pad_sequences(seq, maxlen=MAX_LENGTH)
        prediction = model.predict(padded)[0][0]

        if prediction > 0.5:
            st.success("✅ This news is predicted to be **REAL**.")
        else:
            st.error("🚫 This news is predicted to be **FAKE**.")
