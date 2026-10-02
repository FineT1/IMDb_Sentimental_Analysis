import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Load train and test datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print("Applying preprocessing to training data...")
train_df["clean_review"] = train_df["review"].apply(preprocess_text)

print("Applying preprocessing to testing data...")
test_df["clean_review"] = test_df["review"].apply(preprocess_text)

# Save processed datasets
train_df.to_csv("data/train_processed.csv", index=False)
test_df.to_csv("data/test_processed.csv", index=False)

print("\nPreprocessing completed successfully!")

print("\nTraining dataset:")
print(train_df.shape)

print("\nTesting dataset:")
print(test_df.shape)

print("\nExample:")
print("Original:", train_df.loc[0, "review"][:200])
print("Processed:", train_df.loc[0, "clean_review"][:200])