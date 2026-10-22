import os
import time
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --------------------------------------------------
# 0. Setup and Data Loading
# --------------------------------------------------
# Automatically create models directory if missing
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("STARTING FEATURE VECTORIZATION")
print("=" * 60)
start_time = time.time()

print("Loading preprocessed datasets...")
train_df = pd.read_csv("data/train_processed.csv")
test_df = pd.read_csv("data/test_processed.csv")

# Fill missing/NaN values to avoid vectorizer errors
X_train_text = train_df["clean_review"].fillna("")
X_test_text = test_df["clean_review"].fillna("")

print(f"Loaded {len(X_train_text)} training reviews and {len(X_test_text)} testing reviews.")


# --------------------------------------------------
# 1. Binary + 3-gram
# --------------------------------------------------
print("\n[1/3] Generating Binary 3-gram Vectorizer...")
step_start = time.time()

binary_vectorizer = CountVectorizer(
    binary=True,
    ngram_range=(1, 3),
    max_features=50000,
    min_df=2
)

X_train_binary = binary_vectorizer.fit_transform(X_train_text)
X_test_binary = binary_vectorizer.transform(X_test_text)

joblib.dump(binary_vectorizer, "models/binary_3gram_vectorizer.pkl")

print(f"✓ Completed in {time.time() - step_start:.2f}s")
print(f"  Training shape : {X_train_binary.shape}")
print(f"  Testing shape  : {X_test_binary.shape}")


# --------------------------------------------------
# 2. Word Count + 3-gram
# --------------------------------------------------
print("\n[2/3] Generating Word Count 3-gram Vectorizer...")
step_start = time.time()

count_vectorizer = CountVectorizer(
    binary=False,
    ngram_range=(1, 3),
    max_features=50000,
    min_df=2
)

X_train_count = count_vectorizer.fit_transform(X_train_text)
X_test_count = count_vectorizer.transform(X_test_text)

joblib.dump(count_vectorizer, "models/count_3gram_vectorizer.pkl")

print(f"✓ Completed in {time.time() - step_start:.2f}s")
print(f"  Training shape : {X_train_count.shape}")
print(f"  Testing shape  : {X_test_count.shape}")


# --------------------------------------------------
# 3. TF-IDF + 3-gram
# --------------------------------------------------
print("\n[3/3] Generating TF-IDF 3-gram Vectorizer...")
step_start = time.time()

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=50000,
    min_df=2
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

joblib.dump(tfidf_vectorizer, "models/tfidf_3gram_vectorizer.pkl")

print(f"✓ Completed in {time.time() - step_start:.2f}s")
print(f"  Training shape : {X_train_tfidf.shape}")
print(f"  Testing shape  : {X_test_tfidf.shape}")


# --------------------------------------------------
# Display Final Results
# --------------------------------------------------
total_time = time.time() - start_time
print("\n" + "=" * 60)
print(f"VECTORIZATION COMPLETE (Total Time: {total_time:.2f}s)")
print("=" * 60)

print("\nSaved vectorizers in models/:")
print("  - models/binary_3gram_vectorizer.pkl")
print("  - models/count_3gram_vectorizer.pkl")
print("  - models/tfidf_3gram_vectorizer.pkl\n")