# Iris Flower Classification

Classifies iris flowers into three species — **Setosa, Versicolor, Virginica** —
based on petal and sepal measurements.

## Dataset
The classic Iris dataset, loaded directly from `scikit-learn`
(`sklearn.datasets.load_iris`) — 150 samples, 4 numeric features, already clean.

## Approach
1. **Explore** — pairplot and per-feature histograms by species (`outputs/pairplot.png`, `outputs/histograms.png`).
2. **Split** — 80/20 train/test split, stratified by species.
3. **Preprocess** — standard-scale the four numeric features.
4. **Train & compare** three classifiers:
   - Logistic Regression
   - K-Nearest Neighbors (k=5)
   - Decision Tree
5. **Evaluate** the best model with accuracy, macro precision, a full
   classification report, and a confusion matrix (`outputs/confusion_matrix.png`).

## Results
| Model | Accuracy | Precision (macro) |
|---|---|---|
| Logistic Regression | 0.933 | 0.933 |
| K-Nearest Neighbors | 0.933 | 0.944 |
| Decision Tree | 0.933 | 0.933 |

All three models perform similarly well on this small, well-separated dataset;
Setosa is perfectly separable, most of the (small) error comes from
Versicolor/Virginica overlap, which is expected and visible in the pairplot.

## Run it yourself
```bash
pip install -r requirements.txt
python iris_classification.py
```
Outputs (plots + comparison table) are written to `outputs/`.

## Skills demonstrated
Numeric EDA, train/test splitting, feature scaling, classification modeling,
model comparison, and evaluation via accuracy / precision / confusion matrix.
