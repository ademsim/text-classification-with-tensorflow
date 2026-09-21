import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Airline Tweet Sentiment (Neural Network)")

MODEL_PATH = Path(__file__).parent / "sentiment_avgpool.keras"
CLASSES = ["negative", "neutral", "positive"]
ICONS = {"negative": "😠", "neutral": "😐", "positive": "😊"}


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict(cleaned):
    logits = model.predict(tf.constant([cleaned], shape=(1, 1)), verbose=0)[0]
    e = np.exp(logits - logits.max())
    return e / e.sum()


EXAMPLES = {
    "Write your own": "",
    "Negative": "@united my flight was delayed for 5 hours and nobody told us anything",
    "Positive": "@delta thanks for the amazing service, the crew was great!",
    "Neutral": "@americanair can you tell me the baggage allowance for international flights?",
    "Negation": "@jetblue not happy with the service today",
}

st.title("Airline Tweet Sentiment")
st.write(
    "A neural network (word embeddings + average pooling, built with TensorFlow/Keras) that reads a tweet "
    "about an airline and predicts if it is **negative**, **neutral**, or **positive**. English text only."
)

choice = st.selectbox("Load an example (optional)", list(EXAMPLES.keys()))
tweet = st.text_area("Tweet", value=EXAMPLES[choice], height=120)

if st.button("Analyze sentiment"):
    cleaned = clean(tweet)
    if len(cleaned) < 3:
        st.warning("Please enter a longer text with a few English words.")
    else:
        probs = predict(cleaned)
        guess = CLASSES[int(probs.argmax())]
        st.success(f"{ICONS[guess]} Predicted sentiment: **{guess}** ({probs.max():.0%})")
        st.bar_chart(pd.Series(probs, index=CLASSES, name="probability"))
        if len(cleaned.split()) < 4:
            st.caption("Very short texts contain few clues, so the prediction may be less reliable.")

st.caption(
    "Trained on about 11,500 US airline tweets from 2015 (macro F1 = 0.75 on the test set, the same as a "
    "TF-IDF + Logistic Regression model). It cannot detect sarcasm, and it only knows airline-related language."
)
