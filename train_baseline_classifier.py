import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "TrainingData.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

MODEL_DIR = "D:/RA with Steven Silver/models"
VECTORIZER_PATH = f"{MODEL_DIR}/tfidf_vectorizer.pkl"
MODEL_PATH = f"{MODEL_DIR}/logreg_model.pkl"

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading training data...")
df = pd.read_csv(DATA_PATH)

X = df[TEXT_COLUMN].astype(str)
y = df[LABEL_COLUMN].astype(int)

print("Total samples:", len(df))
print("Label distribution:")
print(y.value_counts())

# -----------------------------
# TRAIN / VALIDATION SPLIT
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))

# -----------------------------
# TF-IDF VECTORIZATION
# -----------------------------
print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=5,
    stop_words="english",
    max_features=200_000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# -----------------------------
# TRAIN CLASSIFIER
# -----------------------------
print("Training Logistic Regression...")
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

clf.fit(X_train_tfidf, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
print("\nValidation Results:")
y_pred = clf.predict(X_val_tfidf)

print(classification_report(y_val, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
print("\nSaving model artifacts...")
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(clf, MODEL_PATH)

print("Saved:")
print(VECTORIZER_PATH)
print(MODEL_PATH)
