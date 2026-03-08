import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# تحميل الموديل
rnn_model = tf.keras.models.load_model("news_rnn_baseline.keras")
rnn_tokenizer = joblib.load("news_rnn_tokenizer.pkl")

RNN_MAX_LEN = 600

def predict(text):
    seq = rnn_tokenizer.texts_to_sequences([text])
    x = pad_sequences(seq, maxlen=RNN_MAX_LEN, padding="post", truncating="post")

    p_ai = rnn_model.predict(x, verbose=0).ravel()[0]
    p_human = 1.0 - p_ai

    return float(p_human), float(p_ai)


if __name__ == "__main__":
    sample = "This is a test article."
    human, ai = predict(sample)
    print("Human:", human)
    print("AI:", ai)