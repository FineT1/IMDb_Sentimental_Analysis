import os
import gc
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# Create required folders
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# --------------------------------------------------
# Load processed datasets
# --------------------------------------------------

train_df = pd.read_csv("data/train_processed.csv")
test_df = pd.read_csv("data/test_processed.csv")

X_train_text = train_df["clean_review"]
X_test_text = test_df["clean_review"]

y_train = train_df["sentiment"]
y_test = test_df["sentiment"]


# Convert labels to numbers
y_train = y_train.map({"negative": 0, "positive": 1})
y_test = y_test.map({"negative": 0, "positive": 1})


# --------------------------------------------------
# Function to evaluate a model
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, cm


# --------------------------------------------------
# Vectorizers to evaluate
# --------------------------------------------------

vectorizers = {
    "Binary_3gram": "models/binary_3gram_vectorizer.pkl",
    "Count_3gram": "models/count_3gram_vectorizer.pkl",
    "TFIDF_3gram": "models/tfidf_3gram_vectorizer.pkl"
}


results = []


# --------------------------------------------------
# Train models one representation at a time
# --------------------------------------------------

for representation, vectorizer_path in vectorizers.items():

    print("\n" + "=" * 60)
    print(f"PROCESSING: {representation}")
    print("=" * 60)

    # Load vectorizer
    vectorizer = joblib.load(vectorizer_path)

    print("Transforming training data...")
    X_train = vectorizer.transform(X_train_text)

    print("Transforming testing data...")
    X_test = vectorizer.transform(X_test_text)

    print("Training matrix:", X_train.shape)
    print("Testing matrix :", X_test.shape)

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")

    lr_model = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        random_state=42
    )

    lr_model.fit(X_train, y_train)

    lr_accuracy, lr_precision, lr_recall, lr_f1, lr_cm = evaluate_model(
        lr_model,
        X_test,
        y_test
    )

    print("\nLogistic Regression Results")
    print("Accuracy :", lr_accuracy)
    print("Precision:", lr_precision)
    print("Recall   :", lr_recall)
    print("F1 Score :", lr_f1)

    print("Confusion Matrix:")
    print(lr_cm)

    joblib.dump(
        lr_model,
        f"models/logistic_regression_{representation}.pkl"
    )


    # --------------------------------------------------
    # Naive Bayes
    # --------------------------------------------------

    print("\nTraining Naive Bayes...")

    nb_model = MultinomialNB()

    nb_model.fit(X_train, y_train)

    nb_accuracy, nb_precision, nb_recall, nb_f1, nb_cm = evaluate_model(
        nb_model,
        X_test,
        y_test
    )

    print("\nNaive Bayes Results")
    print("Accuracy :", nb_accuracy)
    print("Precision:", nb_precision)
    print("Recall   :", nb_recall)
    print("F1 Score :", nb_f1)

    print("Confusion Matrix:")
    print(nb_cm)

    joblib.dump(
        nb_model,
        f"models/naive_bayes_{representation}.pkl"
    )


    # --------------------------------------------------
    # Store results
    # --------------------------------------------------

    results.append({
        "Representation": representation,
        "Model": "Logistic Regression",
        "Accuracy": lr_accuracy,
        "Precision": lr_precision,
        "Recall": lr_recall,
        "F1 Score": lr_f1
    })

    results.append({
        "Representation": representation,
        "Model": "Naive Bayes",
        "Accuracy": nb_accuracy,
        "Precision": nb_precision,
        "Recall": nb_recall,
        "F1 Score": nb_f1
    })


    # --------------------------------------------------
    # Free memory before next representation
    # --------------------------------------------------

    del X_train
    del X_test
    del vectorizer
    del lr_model
    del nb_model

    gc.collect()


# --------------------------------------------------
# Save final results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/member1_results.csv",
    index=False
)

print("\n" + "=" * 60)
print("ALL MEMBER 1 EXPERIMENTS COMPLETED")
print("=" * 60)

print("\nFinal Results:")
print(results_df.to_string(index=False))

print("\nResults saved to:")
print("results/member1_results.csv")