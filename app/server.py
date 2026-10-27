"""
SentimentSense - Flask backend
Serves the web page and a small JSON API that runs the trained RNN / LSTM models.

Run from anywhere:   python server.py
Then open:           http://127.0.0.1:5000
"""
import os
import re
import threading

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow info logs

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- Settings: must match the notebook used for training -------------------
VOCAB_SIZE = 10000
MAX_LEN = 250

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_FILES = {
    "lstm": "lstm.keras",
    "rnn": "simple_rnn.keras",
}

MAX_TEXT_CHARS = 5000      # longest single review accepted
MAX_BATCH = 2000           # most reviews accepted in one upload
POSITIVE_AT = 0.60         # score >= 0.60 -> Positive
NEGATIVE_AT = 0.40         # score <= 0.40 -> Negative, in between -> Uncertain

app = Flask(__name__, static_folder="static", static_url_path="")

# ---- Load word index and models once, at start-up --------------------------
print("Loading IMDB word index...")
WORD_INDEX = imdb.get_word_index()

MODELS = {}
for key, filename in MODEL_FILES.items():
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        print(f"Loading {key} model from {path}")
        MODELS[key] = tf.keras.models.load_model(path)
        MODELS[key].predict(np.zeros((1, MAX_LEN), dtype="int32"), verbose=0)  # warm-up
    else:
        print(f"WARNING: {path} not found. The '{key}' model is not available.")

predict_lock = threading.Lock()


# ---- Text -> numbers (same rules as the Keras IMDB dataset) ----------------
def encode(text):
    """
    Keras IMDB rules:
      0 = padding, 1 = start of review, 2 = unknown word,
      real words = word_index[word] + 3
    """
    text = re.sub(r"<[^>]+>", " ", text.lower())      # remove HTML tags
    text = re.sub(r"[^a-z0-9' ]", " ", text)          # keep letters, digits, apostrophes
    ids = [1]
    for word in text.split():
        idx = WORD_INDEX.get(word)
        idx = idx + 3 if idx is not None else 2
        ids.append(idx if idx < VOCAB_SIZE else 2)
    return ids


def score_texts(model_key, texts):
    """Return the probability of 'positive' for each text."""
    seqs = [encode(t) for t in texts]
    x = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    with predict_lock:
        probs = MODELS[model_key].predict(x, batch_size=64, verbose=0)
    return probs.ravel().tolist()


def describe(score):
    if score >= POSITIVE_AT:
        label = "Positive"
    elif score <= NEGATIVE_AT:
        label = "Negative"
    else:
        label = "Uncertain"
    return {
        "label": label,
        "score": round(score, 4),                       # probability of positive
        "confidence": round(max(score, 1 - score), 4),
    }


def pick_models(choice):
    """'lstm', 'rnn' or 'both' -> list of model keys that are loaded."""
    keys = list(MODELS) if choice == "both" else [choice]
    missing = [k for k in keys if k not in MODELS]
    if not keys or missing:
        return None, f"Model not available: {', '.join(missing) or choice}. Train and save it first."
    return keys, None


# ---- Routes ----------------------------------------------------------------
@app.get("/")
def home():
    return app.send_static_file("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "models": list(MODELS)})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    choice = data.get("model", "lstm")

    if not text:
        return jsonify({"error": "Please enter a review."}), 400
    if len(text) > MAX_TEXT_CHARS:
        return jsonify({"error": f"Review is too long (max {MAX_TEXT_CHARS} characters)."}), 400

    keys, error = pick_models(choice)
    if error:
        return jsonify({"error": error}), 400

    results = {k: describe(score_texts(k, [text])[0]) for k in keys}
    return jsonify({"results": results})


@app.post("/predict_batch")
def predict_batch():
    data = request.get_json(silent=True) or {}
    texts = [str(t).strip() for t in (data.get("texts") or []) if str(t).strip()]
    choice = data.get("model", "lstm")

    if not texts:
        return jsonify({"error": "No reviews found in the file."}), 400
    if len(texts) > MAX_BATCH:
        return jsonify({"error": f"Too many reviews (max {MAX_BATCH} per upload)."}), 400
    if choice == "both":
        choice = "lstm" if "lstm" in MODELS else choice

    keys, error = pick_models(choice)
    if error:
        return jsonify({"error": error}), 400

    key = keys[0]
    scores = score_texts(key, [t[:MAX_TEXT_CHARS] for t in texts])
    results = [describe(s) for s in scores]

    summary = {"total": len(results), "positive": 0, "negative": 0, "uncertain": 0}
    for r in results:
        summary[r["label"].lower()] += 1

    return jsonify({"model": key, "results": results, "summary": summary})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)