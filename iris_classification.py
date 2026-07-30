"""
Iris Flower Classification
Classify iris flowers into Setosa, Versicolor, Virginica using
petal/sepal measurements.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print("Dataset shape:", df.shape)
print(df.head(), "\n")
print(df.describe(), "\n")

# ---------------------------------------------------------------
# 2. Explore visually
# ---------------------------------------------------------------
sns.pairplot(df, hue="species", vars=iris.feature_names, corner=True)
plt.savefig("outputs/pairplot.png", dpi=140, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, col in zip(axes, iris.feature_names):
    sns.histplot(data=df, x=col, hue="species", ax=ax, kde=True, legend=(col == iris.feature_names[0]))
    ax.set_title(col)
plt.tight_layout()
plt.savefig("outputs/histograms.png", dpi=140, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 3. Split into train / test
# ---------------------------------------------------------------
X = df[iris.feature_names]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------
# 4. Preprocess (scale — iris is already clean, but scaling helps KNN)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 5. Train and compare a few classifiers
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

results = []
best_name, best_acc, best_model, best_preds = None, -1, None, None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    results.append({"model": name, "accuracy": round(acc, 4), "precision_macro": round(prec, 4)})
    if acc > best_acc:
        best_name, best_acc, best_model, best_preds = name, acc, model, preds

results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
print("Model comparison:\n", results_df, "\n")
results_df.to_csv("outputs/model_comparison.csv", index=False)

# ---------------------------------------------------------------
# 6. Evaluate the best model in detail
# ---------------------------------------------------------------
print(f"Best model: {best_name} (accuracy={best_acc:.4f})\n")
print("Classification report:\n", classification_report(y_test, best_preds, target_names=iris.target_names))

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=140)
plt.close()

print("\nSaved: outputs/pairplot.png, outputs/histograms.png, outputs/model_comparison.csv, outputs/confusion_matrix.png")
