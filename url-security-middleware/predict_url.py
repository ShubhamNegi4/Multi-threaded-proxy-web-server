# predict_url.py

import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from urllib.parse import urlparse

# 🔃 Load model and pre-processing tools
model = tf.keras.models.load_model(r"saved_models/url_cnn_lstm_model.keras")
with open(r"saved_models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open(r"saved_models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 200  # Must match training
THRESHOLD = 0.80  # Confidence threshold

# ✅ Trusted allowlist of known safe domains


# 🔍 Input URLs for prediction
urls = [
    # Valid benign
    "https://www.google.com",
    "https://github.com",
    "https://university.edu/home?ref=42",
    # Malicious
    "http://free-bitcoin.ru/get-rich-now",
    "https://secure-login.ph1sh.xyz/index.php?id=123",
    "http://malware-download.biz/<script>alert(1)</script>",
    "https://httpbin.org/get",
    # Invalid/non-URL
    "hmy name name",
    "just some random text",
    "1234567890",
    # Edge-case
    "http://192.168.1.1",
    "ftp://example.com/resource",
    "http://clickjack.tk/?q=' OR 1=1 --",
    "http://example.com/%3Csvg/onload=alert(1)%3E"
]
allowlist = [
    "www.google.com",
    "google.com",
    "www.microsoft.com",
    "microsoft.com",
    "github.com",
    "www.github.com",
    "www.wikipedia.org",
    "wikipedia.org",
    "httpbin.org",
    "www.httpbin.org",
    "example.com",
    "www.example.com",
    "stackoverflow.com",
    "www.stackoverflow.com",
    "reddit.com",
    "www.reddit.com",
    "youtube.com",
    "www.youtube.com",
    "facebook.com",
    "www.facebook.com",
    "twitter.com",
    "www.twitter.com",
    "linkedin.com",
    "www.linkedin.com"
]
def predict_url(url: str):
    """Predict class for a URL or string. Returns top-2 classes and confidences, with explanations for not_a_url/edge_case."""
    # Convert HttpUrl object to string if needed
    if hasattr(url, '__str__'):
        url = str(url)

    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    scheme = parsed.scheme.lower()

    # ✅ Check against allowlist (only for valid URLs with netloc)
    if domain in allowlist and domain != "":
        return {
            "prediction": "benign",
            "score": 0,
            "result": 0,
            "explanation": "Trusted domain (allowlisted)."
        }

    # ⚠️ Input validation: check if input is a valid URL
    is_valid_url = bool(domain) and bool(scheme)
    warning = None
    if not is_valid_url:
        warning = "⚠️ Input is not a valid URL. Prediction may not be meaningful."

    # 🔮 Predict using model regardless
    seq = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prediction = model.predict(padded, verbose=0)[0]

    top2_idx = prediction.argsort()[-2:][::-1]
    top1, top2 = top2_idx[0], top2_idx[1]
    class1 = label_encoder.inverse_transform([top1])[0]
    class2 = label_encoder.inverse_transform([top2])[0]
    conf1 = float(prediction[top1])
    conf2 = float(prediction[top2])

    explanation = None
    if class1 == "not_a_url":
        explanation = "Input does not appear to be a URL."
    elif class1 == "edge_case":
        explanation = "Input is a rare/edge-case URL (e.g., IP, FTP, encoded, or partial)."
    elif warning:
        explanation = warning
    else:
        explanation = None

    # Assign score based on prediction
    if class1 == "not_a_url":
        score = 1
    elif class1 == "benign":
        score = 0
    else:
        # Malicious/edge/other: score is proportional to confidence, but always at least 0.1
        score = max(0.1, round(conf1, 2))

    # result: 0 if benign, 1 otherwise
    result_flag = 0 if class1 == "benign" else 1

    return {
        "prediction": class1,
        "score": score,
        "result": result_flag,
        "explanation": explanation
    }

# Demo code for testing
if __name__ == "__main__":
    print()
    for url in urls:
        print(f"🔍 {url}")
        result = predict_url(url)
        if isinstance(result, dict):
            print(f"  Prediction      : {result['prediction'].upper()}")
            print(f"  Score           : {result['score']}")
            print(f"  Result          : {result['result']}")
            if result['explanation']:
                print(f"  Explanation     : {result['explanation']}")
            print()
        else:
            # fallback for any tuple output
            print(f"  Prediction      : {result[0]}")
            print(f"  Confidence      : {result[1]:.2f}\n")
