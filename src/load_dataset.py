import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/IMDB Dataset.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Stratified 50/50 split
train_df, test_df = train_test_split(
    df,
    test_size=0.50,
    random_state=42,
    stratify=df["sentiment"]
)

# Reset indexes
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Save the split datasets
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

# Display results
print("\nTraining dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

print("\nTraining sentiment distribution:")
print(train_df["sentiment"].value_counts())

print("\nTesting sentiment distribution:")
print(test_df["sentiment"].value_counts())

print("\nTrain and test datasets saved successfully.")