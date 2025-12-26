from pathlib import Path
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "spam.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data and normalize columns (common spam dataset uses v1/v2)
df = pd.read_csv(DATA_PATH, encoding="latin-1")
if "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
else:
    # fallback to commonly used names
    df = df.rename(columns={col: col.strip() for col in df.columns})

df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str).apply(clean_text)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model & vectorizer
with open(MODELS_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(MODELS_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved successfully")
