import os
import gc
import time
import joblib
import pandas as pd
 
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
 
# --------------------------------------------------
# Settings (change here if runs are too slow / too heavy)
# --------------------------------------------------
 
RANDOM_STATE = 42
 
# The 1-3 gram matrices have millions of columns. Random Forest, Boosting
# and the DNN cannot realistically train on that, so for these three models
# we keep only the top-K features ranked by chi-square (fitted on TRAIN only).
# SVM uses the full feature matrix.
K_FEATURES_RF_DNN = 20000
K_FEATURES_BOOST = 10000      # boosting is the slowest; use fewer features
 
SVM_C_VALUES = [0.01, 0.05, 0.25, 0.5, 1]   # from the paper
 
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
 
# Guard against empty reviews after cleaning
train_df["clean_review"] = train_df["clean_review"].fillna("")
test_df["clean_review"] = test_df["clean_review"].fillna("")
 
X_train_text = train_df["clean_review"]
X_test_text = test_df["clean_review"]
 
y_train = train_df["sentiment"].map({"negative": 0, "positive": 1})
y_test = test_df["sentiment"].map({"negative": 0, "positive": 1})
 
print("Train size:", len(train_df), "| Test size:", len(test_df))
 
 
# --------------------------------------------------
# Evaluation helper (same metrics as Member 1)
# --------------------------------------------------
 
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
 
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)               # positive class
    neg_precision = precision_score(y_test, y_pred, pos_label=0)  # negative class
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
 
    return accuracy, precision, neg_precision, recall, f1, cm
 
 
# --------------------------------------------------
# Vectorizers to evaluate (created by vectorize.py)
# --------------------------------------------------
 
vectorizers = {
    "Binary_3gram": "models/binary_3gram_vectorizer.pkl",
    "Count_3gram": "models/count_3gram_vectorizer.pkl",
    "TFIDF_3gram": "models/tfidf_3gram_vectorizer.pkl",
}
 
results = []
RESULTS_PATH = "results/member2_results.csv"
 
 
def save_results():
    """Save after every model so a crash does not lose finished work."""
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
 
 
# --------------------------------------------------
# Main loop: one representation at a time
# --------------------------------------------------
 
for representation, vectorizer_path in vectorizers.items():
 
    print("\n" + "=" * 60)
    print(f"PROCESSING: {representation}")
    print("=" * 60)
 
    vectorizer = joblib.load(vectorizer_path)
 
    print("Transforming training data...")
    X_train = vectorizer.transform(X_train_text)
    print("Transforming testing data...")
    X_test = vectorizer.transform(X_test_text)
 
    print("Full training matrix:", X_train.shape)
    print("Full testing matrix :", X_test.shape)
 
    # Reduced feature sets for the heavy models (selector fitted on TRAIN only)
    print(f"\nSelecting top {K_FEATURES_RF_DNN} features (RF / DNN)...")
    selector_main = SelectKBest(chi2, k=K_FEATURES_RF_DNN).fit(X_train, y_train)
    X_train_main = selector_main.transform(X_train)
    X_test_main = selector_main.transform(X_test)
 
    print(f"Selecting top {K_FEATURES_BOOST} features (Boosting)...")
    selector_boost = SelectKBest(chi2, k=K_FEATURES_BOOST).fit(X_train, y_train)
    X_train_boost = selector_boost.transform(X_train)
    X_test_boost = selector_boost.transform(X_test)
 
    # Each entry: model name -> (model object, X_train, X_test, features used)
    experiments = {
        "SVM": (
            GridSearchCV(
                LinearSVC(
                    penalty="l2",
                    tol=1e-4,
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
                param_grid={"C": SVM_C_VALUES},
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
            ),
            X_train,
            X_test,
            "all",
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=100,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            X_train_main,
            X_test_main,
            K_FEATURES_RF_DNN,
        ),
        "Boosting": (
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                min_samples_split=2,
                random_state=RANDOM_STATE,
            ),
            X_train_boost,
            X_test_boost,
            K_FEATURES_BOOST,
        ),
        "DNN": (
            MLPClassifier(
                hidden_layer_sizes=(30, 30, 20, 10, 10),
                activation="logistic",
                solver="adam",
                alpha=0.0001,
                early_stopping=True,
                max_iter=200,
                random_state=RANDOM_STATE,
            ),
            X_train_main,
            X_test_main,
            K_FEATURES_RF_DNN,
        ),
    }
 
    for model_name, (model, Xtr, Xte, n_features) in experiments.items():
 
        print("\n" + "-" * 60)
        print(f"Training {model_name} on {representation} (features: {n_features})")
        print("-" * 60)
 
        start = time.time()
        model.fit(Xtr, y_train)
        elapsed = time.time() - start
 
        if model_name == "SVM":
            print("Best C (from CV on training data):", model.best_params_["C"])
            best_c = model.best_params_["C"]
            model_to_save = model.best_estimator_
        else:
            best_c = None
            model_to_save = model
 
        acc, prec, neg_prec, rec, f1, cm = evaluate_model(model, Xte, y_test)
 
        print(f"\n{model_name} Results ({elapsed:.1f}s)")
        print("Accuracy           :", round(acc, 4))
        print("Positive Precision :", round(prec, 4))
        print("Negative Precision :", round(neg_prec, 4))
        print("Recall             :", round(rec, 4))
        print("F1 Score           :", round(f1, 4))
        print("Confusion Matrix:")
        print(cm)
 
        file_name = model_name.lower().replace(" ", "_")
        joblib.dump(model_to_save, f"models/{file_name}_{representation}.pkl")
 
        results.append({
            "Representation": representation,
            "Model": model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "Negative Precision": neg_prec,
            "Best C": best_c,
            "Features Used": n_features,
            "Train Time (s)": round(elapsed, 1),
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
        })
 
        save_results()   # checkpoint after every model
 
    # Free memory before next representation
    del X_train, X_test, X_train_main, X_test_main, X_train_boost, X_test_boost
    del vectorizer, selector_main, selector_boost, experiments
    gc.collect()
 
 
# --------------------------------------------------
# Final summary
# --------------------------------------------------
 
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_PATH, index=False)
 
print("\n" + "=" * 60)
print("ALL MEMBER 2 EXPERIMENTS COMPLETED")
print("=" * 60)
 
print("\nFinal Results:")
print(
    results_df[
        ["Representation", "Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    ].to_string(index=False)
)
 
print(f"\nResults saved to: {RESULTS_PATH}")