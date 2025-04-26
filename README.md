# rfsynth-ad

Anomaly detection using LightGBM trained to discriminate between real data and synthetic data generated uniformly within feature bounds, inspired by the paper "EXPLAINABLE UNSUPERVISED ANOMALY DETECTION WITH RANDOM FOREST".

## Installation

You can install the package using pip:

```bash
pip install rfsynth-ad
```

(Note: You'll need to upload the package to PyPI first for this command to work globally. For local testing, see below.)

## Usage

Here's a basic example of how to use the `RFSynthAnomalyDetector`:

```python
import pandas as pd
import numpy as np
from rfsynth_ad import RFSynthAnomalyDetector

# Create some sample data with outliers
np.random.seed(42)
n_inliers = 1000
n_outliers = 50

# Inliers
inliers_num = np.random.randn(n_inliers, 2) * 5 + 10
inliers_cat = np.random.choice(['A', 'B'], size=n_inliers, p=[0.8, 0.2])
inliers_int = np.random.randint(1, 10, size=n_inliers)
inliers_bool = np.random.choice([True, False], size=n_inliers, p=[0.6, 0.4])
inliers_nullable_int = np.random.choice([1, 2, 3, pd.NA], size=n_inliers, p=[0.3, 0.3, 0.3, 0.1])

df_inliers = pd.DataFrame(inliers_num, columns=['num_col_1', 'num_col_2'])
df_inliers['cat_col'] = inliers_cat
df_inliers['int_col'] = inliers_int
df_inliers['bool_col'] = inliers_bool
df_inliers['nullable_int_col'] = inliers_nullable_int

# Outliers
outliers_num = np.random.rand(n_outliers, 2) * 50 + 50
outliers_cat = np.random.choice(['C', 'D'], size=n_outliers, p=[0.7, 0.3])
outliers_int = np.random.randint(20, 30, size=n_outliers)
outliers_bool = np.random.choice([pd.NA], size=n_outliers)
outliers_nullable_int = np.random.choice([100, 200, pd.NA], size=n_outliers, p=[0.4, 0.4, 0.2])

df_outliers = pd.DataFrame(outliers_num, columns=['num_col_1', 'num_col_2'])
df_outliers['cat_col'] = outliers_cat
df_outliers['int_col'] = outliers_int
df_outliers['bool_col'] = outliers_bool
df_outliers['nullable_int_col'] = outliers_nullable_int

# Combine data and ensure dtypes
df_combined = pd.concat([df_inliers, df_outliers], ignore_index=True)
df_combined['cat_col'] = df_combined['cat_col'].astype('category')
df_combined['int_col'] = df_combined['int_col'].astype('int64')
df_combined['bool_col'] = df_combined['bool_col'].astype('boolean')
df_combined['nullable_int_col'] = df_combined['nullable_int_col'].astype('Int64')

# Add columns the sampler should ignore
df_combined['datetime_ignore'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(range(len(df_combined)), unit='s')
df_combined['large_object_ignore'] = [f'item_{i}' for i in range(len(df_combined))]
df_combined['all_nan_col'] = np.nan


# Initialize the detector
# Defaults to generating n_real synthetic samples
detector = RFSynthAnomalyDetector(random_state=42, object_unique_threshold=50)

# Fit the detector (on the full dataset, unsupervised)
detector.fit(df_combined)

# Score the data
anomaly_scores = detector.score(df_combined)

print("\nAnomaly Scores (first 10):")
print(anomaly_scores[:10])

print("\nAnomaly Scores (first 10 outliers):")
print(anomaly_scores[n_inliers:n_inliers+10])

# Predict outliers using a threshold (e.g., top 5% scores)
threshold = np.percentile(anomaly_scores, 95)
predictions = detector.predict(df_combined, threshold=threshold)

print(f"\nThreshold for prediction: {threshold:.4f}")
print("Predictions (first 10):")
print(predictions[:10])
print("Predictions (first 10 outliers):")
print(predictions[n_inliers:n_inliers+10])

# Evaluate if true labels are available
# (In a real unsupervised scenario, you wouldn't have y_true)
try:
    from sklearn.metrics import roc_auc_score, classification_report
    y_true = np.array([0] * n_inliers + [1] * n_outliers)
    auc = roc_auc_score(y_true, anomaly_scores)
    print(f"\nAUC-ROC Score: {auc:.4f}")
    print("\nClassification Report (using percentile threshold):")
    print(classification_report(y_true, predictions))
except ImportError:
    print("\nScikit-learn not fully installed, skipping evaluation metrics.")

```

## API

*   `RFSynthAnomalyDetector(n_synthetic_samples=None, object_unique_threshold=50, lgbm_params=None, random_state=None)`: Initializes the detector.
    *   `n_synthetic_samples`: Number of synthetic samples to generate (default=number of real samples).
    *   `object_unique_threshold`: Threshold for ignoring object columns.
    *   `lgbm_params`: Dictionary of parameters for `lightgbm.LGBMClassifier`.
    *   `random_state`: Seed for reproducibility.
*   `fit(X)`: Fits the detector to the DataFrame `X`.
*   `score(X)`: Computes anomaly scores for DataFrame `X`.
*   `predict(X, threshold)`: Predicts anomalies based on scores and a threshold.

## Development

To work on the code locally and install it in editable mode:

```bash
git clone <your-repo-url>
cd rfsynth-ad
pip install -e .
```

## License

MIT License
```