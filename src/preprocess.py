import re

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove non-letters, collapse whitespace."""
    text = str(text).lower()
    # remove urls and emails
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
