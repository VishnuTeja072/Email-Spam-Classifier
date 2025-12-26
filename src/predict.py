from pathlib import Path
import pickle
from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"
VECT_PATH = ROOT / "models" / "vectorizer.pkl"

def _load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

model = _load_pickle(MODEL_PATH)
vectorizer = _load_pickle(VECT_PATH)

def predict_email(text):
    if model is None or vectorizer is None:
        raise RuntimeError(f"Model or vectorizer not found. Expected {MODEL_PATH} and {VECT_PATH}")
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = int(model.predict(vector)[0])
    # handle models without predict_proba
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(vector).max())
    else:
        prob = 1.0
    return prediction, round(prob * 100, 2)
