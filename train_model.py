import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("emotion_dataset.csv")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

df["clean_text"] = df["text"].apply(clean_text)

# Vectorization
tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words="english",
    max_features=5000
)

X = tfidf.fit_transform(df["clean_text"])
y = df["emotion"]

# Train
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)
model.fit(X, y)

# Save model
pickle.dump(model, open("emotion_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("✅ Model trained successfully")
