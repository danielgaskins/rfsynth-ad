# rfsynth_ad/__init__.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted # Helper for checking if fit was called

from .detector import SyntheticSampler, LGBMAnomalyDetector # Import internal components
from ._metadata import * # Import metadata for versioning and author info

import warnings
import pandas.api.types as ptypes # Needed for internal checks when passing info

class RFSynthAnomalyDetector(BaseEstimator, TransformerMixin):
    """
    Anomaly detector using an LGBM classifier trained to discriminate
    real data from synthetic data generated uniformly within bounds.

    This detector learns the distribution of the real data by training
    a classifier to distinguish real samples from synthetic samples drawn
    from the feature bounds. Anomaly scores are based on the probability
    of a sample being classified as synthetic.

    Parameters
    ----------
    n_synthetic_samples : int or None, default=None
        The number of synthetic samples to generate during fit.
        If None, defaults to the number of samples in the training DataFrame.
        If 0 or less, the LGBM model is not trained, and scoring/prediction
        will return zeros or raise an error if a threshold is required.
    contamination : float or None, default=None
        The proportion of outliers in the training data set. This is used
        to determine the threshold for the `predict` method when no explicit
        threshold is provided. If set to a value between 0. and 0.5,
        the threshold is determined automatically by the quantile of the
        scores on the training data. If None, an explicit `threshold`
        must be provided to the `predict` method.
    object_unique_threshold : int, default=50
        If an object column has more unique values than this threshold,
        it is ignored during synthetic sampling. Otherwise, it's treated as
        categorical.
    lgbm_params : dict or None, default=None
        Parameters to pass to the underlying LGBMClassifier.
        Defaults to basic recommended parameters if None.
    random_state : int or None, default=None
        Random state for reproducibility.
    """
    def __init__(self, n_synthetic_samples=None, contamination=None, object_unique_threshold=50, lgbm_params=None, random_state=None):
        self.n_synthetic_samples = n_synthetic_samples
        self.contamination = contamination
        self.object_unique_threshold = object_unique_threshold
        self.lgbm_params = lgbm_params
        self.random_state = random_state

        # Validate contamination parameter
        if self.contamination is not None:
             if not isinstance(self.contamination, (int, float)):
                  raise TypeError("Contamination must be a float or None.")
             if not (0. <= self.contamination <= 0.5): # Often contamination is assumed < 0.5
                 warnings.warn("Contamination is usually set between 0 and 0.5. Value might be unexpected.")
             if not (0. <= self.contamination <= 1.):
                  raise ValueError("Contamination must be between 0 and 1.")


        # Internal components (instantiated during fit)
        self._sampler = SyntheticSampler(object_unique_threshold=self.object_unique_threshold)
        self._lgbm_detector = None # Will be LGBMAnomalyDetector instance or None
        self._columns_order = None # Store original column order
        self._lgbm_features = None # Store list of features used by LGBM
        self._contamination_threshold = None # Store threshold based on contamination


    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the anomaly detector to the training data.

        This method learns the data distribution using the synthetic sampler
        and trains an LGBM classifier to discriminate real from synthetic samples
        using the relevant features. If `contamination` is set, it also
        calculates the prediction threshold based on training scores.

        Parameters
        ----------
        X : pd.DataFrame
            The training data.
        y : Ignored
            Placeholder for sklearn compatibility.

        Returns
        -------
        self : object
            The fitted detector instance.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if X.empty:
             raise ValueError("Input DataFrame X is empty. Cannot fit the detector.")

        self._columns_order = list(X.columns)
        self._contamination_threshold = None # Reset threshold on fit

        # 1. Learn data boundaries using the sampler
        print("Fitting synthetic sampler...")
        self._sampler.fit(X)

        # Identify columns to use for LGBM (all original columns EXCEPT ignored ones)
        self._lgbm_features = [col for col in self._columns_order if col not in self._sampler._ignored_columns]

        if not self._lgbm_features:
             # Clear any previous LGBM detector
             self._lgbm_detector = None
             # Mark as fitted even if no features could be used, subsequent calls need this state
             # But raise ValueError as fitting was unsuccessful in training a model.
             # Let's refine: if no features, fitting fails cleanly.
             raise ValueError("No columns remaining after ignoring unsupported types. Cannot train LGBM.")

        # 2. Determine number of synthetic samples and generate
        n_real = len(X)
        n_synthetic = self.n_synthetic_samples if self.n_synthetic_samples is not None else n_real

        # Instantiate LGBM detector only if we have synthetic samples to train it
        if n_synthetic > 0:
            print(f"Generating {n_synthetic} synthetic samples for {len(self._lgbm_features)} relevant columns...")
            X_synth_all = self._sampler.generate(n_synthetic)

            # 3. Prepare data for the internal LGBM detector - only relevant features
            X_real_relevant = X[self._lgbm_features].copy()
            X_synth_relevant = X_synth_all[self._lgbm_features].copy()

            # Pass the sampler's learned unique values for relevant columns
            sampler_relevant_unique_values = {
                col: vals for col, vals in self._sampler._unique_values.items() if col in self._lgbm_features
            }

            lgbm_params = {}
            if self.lgbm_params:
                 lgbm_params.update(self.lgbm_params) # Override defaults

            self._lgbm_detector = LGBMAnomalyDetector(
                lgbm_params=lgbm_params,
                _sampler_learned_unique_values=sampler_relevant_unique_values,
                random_state=self.random_state
            )

            # Fit the internal detector using the filtered data
            self._lgbm_detector._fit(X_real_relevant, X_synth_relevant)

            # 4. If contamination is set, calculate the threshold using training data scores
            if self.contamination is not None:
                 if self._lgbm_detector is not None and self._lgbm_detector._is_fitted:
                      print(f"Calculating contamination threshold ({self.contamination=})...")
                      # Get scores for the *real* training data subset
                      training_scores = self._lgbm_detector.score(X_real_relevant)

                      # The threshold is the value at the (1 - contamination) percentile
                      # np.percentile handles 0 and 1 correctly.
                      # For contamination=0, threshold is max score (only points with score == max_score are anomalies)
                      # For contamination=1, threshold is min score (all points are anomalies)
                      self._contamination_threshold = np.percentile(training_scores, (1 - self.contamination) * 100)
                      print(f"Calculated contamination threshold: {self._contamination_threshold:.4f}")
                 else:
                      warnings.warn("LGBM model was not fitted, cannot calculate contamination threshold.")
                      self._contamination_threshold = None # Ensure it's None if LGBM fit failed
        else:
            # n_synthetic <= 0 case
             warnings.warn("Number of synthetic samples is 0 or less. Skipping LGBM training.")
             self._lgbm_detector = None # Ensure LGBM detector is None

        # Mark the detector as fitted even if no LGBM model was trained,
        # to distinguish from the unfitted state.
        self._is_fitted = True
        print("Anomaly detector fit complete.")
        return self


    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes anomaly scores for the input data.

        Scores are typically higher for anomalies. The score is the probability
        predicted by the trained classifier that a data point is synthetic.

        Parameters
        ----------
        X : pd.DataFrame
            The data points to score.

        Returns
        -------
        np.ndarray
            Anomaly scores for each data point.
        """
        # Use sklearn's check_is_fitted
        check_is_fitted(self, ['_is_fitted', '_sampler'])
        # If _is_fitted is True, _lgbm_features and _columns_order should be set.

        # If LGBM was not trained
        if self._lgbm_detector is None:
             warnings.warn("LGBM model was not trained. Returning zeros for scores.")
             return np.zeros(len(X), dtype=np.float64)

        if not isinstance(X, pd.DataFrame):
             raise TypeError("Input X must be a pandas DataFrame.")

        # Select only the relevant columns that the LGBM model was trained on
        if not self._lgbm_features:
             # This state should ideally not be reached if _lgbm_detector is not None
             raise RuntimeError("Relevant features were not identified during fitting.")

        # Ensure scoring data has the relevant columns, raising error if missing
        try:
             # Use the internally stored LGBM feature names to select and order columns
             X_relevant = X[self._lgbm_detector._lgbm_features].copy()
        except KeyError as e:
             missing_cols = set(self._lgbm_detector._lgbm_features) - set(X.columns)
             raise ValueError(f"Scoring data is missing expected columns that the model was trained on: {missing_cols}") from e


        # Use the internal LGBM detector to score
        # Its score method is responsible for handling dtypes and column order internally
        anomaly_scores = self._lgbm_detector.score(X_relevant)

        return anomaly_scores

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        Predicts anomalies (1) or inliers (0) for the input data.

        Uses a provided threshold or the contamination-based threshold
        calculated during fit if `contamination` was set.

        Parameters
        ----------
        X : pd.DataFrame
            The data points to predict on.
        threshold : float or None, default=None
            The anomaly score threshold. Points with scores greater than or
            equal to the threshold are predicted as anomalies (1).
            If None, the contamination-based threshold calculated during fit
            is used (if `contamination` was set).

        Returns
        -------
        np.ndarray
            Binary predictions (1 for anomaly, 0 for inlier).
        """
        # Use sklearn's check_is_fitted
        check_is_fitted(self, ['_is_fitted', '_sampler'])
        # If _is_fitted is True, _lgbm_features and _columns_order should be set.

        # If LGBM was not trained
        if self._lgbm_detector is None:
             warnings.warn("LGBM model was not trained. Returning zeros for predictions.")
             return np.zeros(len(X), dtype=int)

        # Determine the threshold to use
        if threshold is not None:
            prediction_threshold = threshold
        elif self.contamination is not None:
            if self._contamination_threshold is None:
                 # This state indicates LGBM was likely trained but threshold calculation failed
                 raise RuntimeError("Contamination threshold was not calculated during fit. Ensure LGBM training was successful.")
            prediction_threshold = self._contamination_threshold
        else:
            # No threshold provided and contamination was not set
            raise ValueError("No threshold provided and contamination level was not set during initialization. Cannot predict.")


        scores = self.score(X)
        return (scores >= prediction_threshold).astype(int)

    # Add sklearn compatibility methods
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        # Expose user-facing parameters
        params = {
            'n_synthetic_samples': self.n_synthetic_samples,
            'contamination': self.contamination,
            'object_unique_threshold': self.object_unique_threshold,
            'lgbm_params': self.lgbm_params,
            'random_state': self.random_state,
        }
        # Note: Getting params from the internal LGBM detector is tricky due to
        # parameters like _sampler_learned_unique_values which aren't user-facing.
        # We'll only expose the user-facing parameters here.
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        # This basic implementation allows setting top-level parameters
        # For more complex scenarios (like setting internal LGBM params),
        # you'd need to handle nested parameter setting.
        if not params:
            return self

        valid_params = self.get_params(deep=False).keys() # Get user-facing params
        for key, value in params.items():
            if key in valid_params:
                # Add validation for contamination here if being set after init
                if key == 'contamination' and value is not None:
                     if not isinstance(value, (int, float)):
                          raise TypeError("Contamination must be a float or None.")
                     if not (0. <= value <= 1.):
                          raise ValueError("Contamination must be between 0 and 1.")
                     if not (0. <= value <= 0.5): # Optional: warning again
                          warnings.warn("Contamination is usually set between 0 and 0.5. Value might be unexpected.")

                setattr(self, key, value)
            # Ignoring unknown parameters is standard sklearn behavior
        return self