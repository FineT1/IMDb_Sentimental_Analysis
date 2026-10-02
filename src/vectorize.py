import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib

# Load preprocessed datasets
train_df = pd.read_csv("data/train_processed.csv")
test_df = pd.read_csv("data/test_processed.csv")

X_train_text = train_df["clean_review"]
X_test_text = test_df["clean_review"]

y_train = train_df["sentiment"]
y_test = test_df["sentiment"]


# --------------------------------------------------
# 1. Binary + 3-gram
# --------------------------------------------------

binary_vectorizer = CountVectorizer(
    binary=True,
    ngram_range=(1, 3)
)

X_train_binary = binary_vectorizer.fit_transform(X_train_text)
X_test_binary = binary_vectorizer.transform(X_test_text)

joblib.dump(binary_vectorizer, "models/binary_3gram_vectorizer.pkl")


# --------------------------------------------------
# 2. Word Count + 3-gram
# --------------------------------------------------

count_vectorizer = CountVectorizer(
    binary=False,
    ngram_range=(1, 3)
)

X_train_count = count_vectorizer.fit_transform(X_train_text)
X_test_count = count_vectorizer.transform(X_test_text)

joblib.dump(count_vectorizer, "models/count_3gram_vectorizer.pkl")


# --------------------------------------------------
# 3. TF-IDF + 3-gram
# --------------------------------------------------

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3)
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

joblib.dump(tfidf_vectorizer, "models/tfidf_3gram_vectorizer.pkl")


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n========== VECTORIZATION COMPLETE ==========")

print("\nBinary + 3-gram:")
print("Training shape:", X_train_binary.shape)
print("Testing shape :", X_test_binary.shape)

print("\nWord Count + 3-gram:")
print("Training shape:", X_train_count.shape)
print("Testing shape :", X_test_count.shape)

print("\nTF-IDF + 3-gram:")
print("Training shape:", X_train_tfidf.shape)
print("Testing shape :", X_test_tfidf.shape)

print("\nVectorizers saved successfully!")