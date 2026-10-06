import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
os.makedirs("results", exist_ok=True)
 
# --------------------------------------------------
# Load Member 2 results
# --------------------------------------------------
 
results_df = pd.read_csv("results/member2_results.csv")
 
print("Loaded Member 2 results successfully!")
print(results_df[["Representation", "Model", "Accuracy",
                  "Precision", "Recall", "F1 Score"]].to_string(index=False))
 
 
# --------------------------------------------------
# Helper: grouped bar chart
# x-axis groups = feature representation, bars = models
# --------------------------------------------------
 
def grouped_bar(df, metric, title, save_path):
    representations = list(df["Representation"].unique())
    models = list(df["Model"].unique())
 
    x = np.arange(len(representations))
    width = 0.8 / len(models)
 
    plt.figure(figsize=(12, 6))
 
    for i, model in enumerate(models):
        values = []
        for rep in representations:
            row = df[(df["Representation"] == rep) & (df["Model"] == model)]
            values.append(row[metric].values[0] * 100 if len(row) else 0)
 
        positions = x + (i - (len(models) - 1) / 2) * width
        bars = plt.bar(positions, values, width, label=model)
 
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.5,
                f"{value:.1f}",
                ha="center",
                fontsize=8,
            )
 
    plt.xticks(x, representations)
    plt.ylabel(f"{metric} (%)")
    plt.xlabel("Feature Representation")
    plt.title(title)
    plt.ylim(0, 105)
    plt.legend(title="Model", loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
 
 
# --------------------------------------------------
# 1. Member 2 charts: Accuracy, Precision, Recall, F1
# --------------------------------------------------
 
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
 
for metric in metrics:
    filename = metric.lower().replace(" ", "_")
    grouped_bar(
        results_df,
        metric,
        f"IMDb Sentiment Classification - {metric} Comparison (Member 2 Models)",
        f"results/member2_{filename}_comparison.png",
    )
 
 
# --------------------------------------------------
# 2. Best model summary (printed)
# --------------------------------------------------
 
best = results_df.loc[results_df["Accuracy"].idxmax()]
print("\nBest Member 2 configuration:")
print(f"  Model          : {best['Model']}")
print(f"  Representation : {best['Representation']}")
print(f"  Accuracy       : {best['Accuracy'] * 100:.2f}%")
 
print("\nBest accuracy per model:")
print(
    results_df.loc[results_df.groupby("Model")["Accuracy"].idxmax(),
                   ["Model", "Representation", "Accuracy"]]
    .to_string(index=False)
)
 
 
# --------------------------------------------------
# 3. Combined chart with Member 1 (if available)
# --------------------------------------------------
 
member1_path = "results/member1_results.csv"
 
if os.path.exists(member1_path):
    m1 = pd.read_csv(member1_path)
    common_cols = ["Representation", "Model", "Accuracy",
                   "Precision", "Recall", "F1 Score"]
    combined = pd.concat([m1[common_cols], results_df[common_cols]],
                         ignore_index=True)
 
    combined.to_csv("results/all_models_summary.csv", index=False)
 
    grouped_bar(
        combined,
        "Accuracy",
        "IMDb Sentiment Classification - Accuracy of All Models",
        "results/all_models_accuracy_comparison.png",
    )
    print("\nCombined results saved to results/all_models_summary.csv")
else:
    print("\nresults/member1_results.csv not found - skipping combined chart.")
    print("Ask Member 1 for that file if you want the all-models comparison.")
 
 
print("\nGraphs generated successfully!")
print("\nSaved files:")
print("results/member2_accuracy_comparison.png")
print("results/member2_precision_comparison.png")
print("results/member2_recall_comparison.png")
print("results/member2_f1_score_comparison.png")