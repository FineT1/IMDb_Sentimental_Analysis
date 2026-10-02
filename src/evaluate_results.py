import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# Create results folder if it does not exist
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

# --------------------------------------------------
# Load Member 1 results
# --------------------------------------------------

results_df = pd.read_csv("results/member1_results.csv")

print("Loaded results successfully!")
print(results_df)

# --------------------------------------------------
# 1. Accuracy Comparison
# --------------------------------------------------

plt.figure(figsize=(10, 6))

labels = (
    results_df["Representation"]
    + "\n"
    + results_df["Model"].str.replace(" ", "\n")
)

plt.bar(labels, results_df["Accuracy"] * 100)

plt.ylabel("Accuracy (%)")
plt.xlabel("Feature Representation and Model")
plt.title("IMDb Sentiment Classification - Accuracy Comparison")
plt.xticks(rotation=0)
plt.ylim(0, 100)

for i, value in enumerate(results_df["Accuracy"] * 100):
    plt.text(i, value + 0.5, f"{value:.2f}%", ha="center")

plt.tight_layout()

plt.savefig(
    "results/accuracy_comparison.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 2. Precision, Recall and F1 Comparison
# --------------------------------------------------

metrics = ["Precision", "Recall", "F1 Score"]

for metric in metrics:

    plt.figure(figsize=(10, 6))

    values = results_df[metric] * 100

    plt.bar(labels, values)

    plt.ylabel(f"{metric} (%)")
    plt.xlabel("Feature Representation and Model")
    plt.title(f"IMDb Sentiment Classification - {metric} Comparison")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)

    for i, value in enumerate(values):
        plt.text(i, value + 0.5, f"{value:.2f}%", ha="center")

    plt.tight_layout()

    filename = metric.lower().replace(" ", "_")

    plt.savefig(
        f"results/{filename}_comparison.png",
        dpi=300
    )

    plt.show()


print("\nGraphs generated successfully!")

print("\nSaved files:")
print("results/accuracy_comparison.png")
print("results/precision_comparison.png")
print("results/recall_comparison.png")
print("results/f1_score_comparison.png")