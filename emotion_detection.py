import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("emotion_dataset.csv")

# -----------------------------
# 2. Preprocessing Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["emotion"]

# -----------------------------
# 3. Vectorization
# -----------------------------
tfidf = TfidfVectorizer(max_features=3000)
X_vec = tfidf.fit_transform(X)

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# -----------------------------
# 5. Model Training
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# -----------------------------
# 7. Save Model & Vectorizer
# -----------------------------
pickle.dump(model, open("emotion_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("Model & vectorizer saved successfully!")
