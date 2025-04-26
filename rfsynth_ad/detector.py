# rfsynth_ad/detector.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from collections import defaultdict
import pandas.api.types as ptypes # Import pandas type checking functions

# Define types to ignore conceptually - checks will use ptypes
# IGNORED_TYPES_CONCEPTUAL = (pd.core.dtypes.common.is_datetime64_any_dtype, pd.core.dtypes.common.is_timedelta64_dtype)


class SyntheticSampler:
    """
    Generates synthetic data from a pandas DataFrame by sampling uniformly
    within the boundaries of the observed data for each column type.
    (Internal class used by RFSynthAnomalyDetector)
    """
    def __init__(self, object_unique_threshold=50):
        """
        Initializes the SyntheticSampler.

        Args:
            object_unique_threshold (int): If an object column has more unique
                                           values than this threshold, it is
                                           ignored. Otherwise, it's treated as
                                           categorical.
        """
        self.object_unique_threshold = object_unique_threshold
        self._min_max_bounds = {}
        self._unique_values = {} # Stores unique values for categorical/bool/object-as-cat, includes NA/None
        self._column_dtypes = {}
        self._ignored_columns = []
        self._columns_order = [] # Store original column order

    def fit(self, X: pd.DataFrame):
        """
        Learns the boundaries (min/max or unique values) for each column
        in the input DataFrame.

        Args:
            X (pd.DataFrame): The real data to learn from.

        Returns:
            self: The fitted sampler instance.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        # Empty check is done in the main RFSynthAnomalyDetector.fit

        self._min_max_bounds = {}
        self._unique_values = {}
        self._column_dtypes = {}
        self._ignored_columns = []
        self._columns_order = list(X.columns) # Store original order

        for col in X.columns:
            dtype = X[col].dtype
            self._column_dtypes[col] = dtype

            # Use pandas type checking functions exclusively for robustness
            if ptypes.is_datetime64_any_dtype(dtype) or ptypes.is_timedelta64_dtype(dtype):
                 self._ignored_columns.append(col)
                 warnings.warn(f"Ignoring column '{col}' with type {dtype} as datetime/timedelta are not supported for synthetic sampling.")
                 continue

            # Handle boolean types FIRST, as is_numeric_dtype is True for bool
            # Treat bool as categorical from sampling perspective (True/False/NA are distinct values)
            elif ptypes.is_bool_dtype(dtype):
                 series = X.loc[:, col]
                 all_unique_vals = series.unique() # Include pd.NA if present
                 # Check if the column contains only NA/NaN values
                 if all_unique_vals.size == 0 or (all_unique_vals.size == 1 and pd.isna(all_unique_vals[0])):
                      warnings.warn(f"Column '{col}' is boolean but contains no non-NA/NaN values. Ignoring.")
                      self._ignored_columns.append(col)
                      continue
                 self._unique_values[col] = all_unique_vals # Sample from all unique values found (True, False, pd.NA)


            # Handle categorical types (including pandas CategoricalDtype) AFTER bool
            elif isinstance(dtype, pd.CategoricalDtype):
                 series = X.loc[:, col]
                 # Sample from all unique values found, including NaN/None category if present
                 all_unique_vals = series.unique()
                 # Check if the column contains only NA/NaN values
                 if all_unique_vals.size == 0 or (all_unique_vals.size == 1 and pd.isna(all_unique_vals[0])):
                     warnings.warn(f"Column '{col}' is categorical but contains no non-NA/NaN values. Ignoring.")
                     self._ignored_columns.append(col)
                     continue
                 self._unique_values[col] = all_unique_vals # Store unique values including NaN/None


            # Handle object types - check unique value threshold AFTER bool and category
            elif ptypes.is_object_dtype(dtype):
                series = X.loc[:, col]
                all_unique_vals = series.unique() # Sample from all unique values found, including None/NaN
                # Check if the column contains only None/NaN values
                if all_unique_vals.size == 0 or (all_unique_vals.size == 1 and pd.isna(all_unique_vals[0])):
                     warnings.warn(f"Column '{col}' is object but contains no non-NA/NaN values. Ignoring.")
                     self._ignored_columns.append(col)
                     continue
                # Check unique value threshold for non-NA values
                unique_non_null_vals = series.dropna().unique()
                if len(unique_non_null_vals) > self.object_unique_threshold:
                     self._ignored_columns.append(col)
                     warnings.warn(f"Ignoring object column '{col}' with {len(unique_non_null_vals)} non-NA unique values (>{self.object_unique_threshold}) as it may not be truly categorical.")
                     continue

                self._unique_values[col] = all_unique_vals # Sample from all unique values found (including None/NaN)


            # Handle numeric types LAST, after types that are technically numeric but we want to treat differently
            # Use pandas is_numeric_dtype which handles both numpy and pandas numeric types
            elif ptypes.is_numeric_dtype(dtype):
                series = X.loc[:, col]
                series_clean = series.dropna() # Drop NaNs for bounds calculation

                if series_clean.empty:
                    warnings.warn(f"Column '{col}' is numeric but contains only NaNs. Ignoring.")
                    self._ignored_columns.append(col)
                    continue

                min_val = series_clean.min()
                max_val = series_clean.max()
                # Handle cases where min == max (all non-NaN values are the same) or non-finite bounds
                if pd.isna(min_val) or pd.isna(max_val) or not np.isfinite(min_val) or not np.isfinite(max_val):
                     warnings.warn(f"Column '{col}' min/max resulted in non-finite or NaN/NA values after dropna. Ignoring.")
                     self._ignored_columns.append(col)
                     continue
                self._min_max_bounds[col] = (min_val, max_val)


            else:
                # Any other type we haven't explicitly handled AND wasn't ignored above
                self._ignored_columns.append(col)
                warnings.warn(f"Ignoring column '{col}' with unhandled type {dtype} for synthetic sampling.")

        return self

    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generates a DataFrame of synthetic data.

        Args:
            n_samples (int): The number of synthetic samples to generate.

        Returns:
            pd.DataFrame: The generated synthetic data (for all original columns).
        """
        if n_samples <= 0:
            # Return empty DataFrame with original column names, respecting original dtypes for structure
            # Need to use .astype on an empty frame to get extension dtypes correctly
            empty_df = pd.DataFrame(columns=self._columns_order)
            # Ensure dtypes are set correctly for extension types on the empty frame
            for col in self._columns_order:
                 dtype = self._column_dtypes.get(col)
                 if dtype is not None:
                      try:
                           empty_df[col] = empty_df[col].astype(dtype)
                      except Exception:
                           # Some dtypes like object might raise error on empty cast, ignore
                           pass
            return empty_df


        # Check if fit was called meaningfully
        if not self._column_dtypes:
             warnings.warn("Sampler has not been fitted or learned no columns. Returning empty DataFrame.")
             return pd.DataFrame(columns=self._columns_order if self._columns_order else [])


        synthetic_data = {}

        # Iterate through original columns to maintain order and structure
        for col in self._columns_order:
            dtype = self._column_dtypes.get(col) # Use .get for robustness

            # Columns that were ignored during fit
            # For ignored columns, create a Series of nulls matching the original dtype
            if col in self._ignored_columns:
                 fill_value = pd.NA if ptypes.is_extension_array_dtype(dtype) else np.nan
                 # Create a Series filled with appropriate nulls and the original dtype
                 synthetic_data[col] = pd.Series([fill_value] * n_samples, dtype=dtype)
                 continue

            # Generate numeric samples (from bounds)
            if col in self._min_max_bounds:
                min_val, max_val = self._min_max_bounds[col]
                if min_val == max_val:
                    # Single value case
                    synthetic_data[col] = pd.Series([min_val] * n_samples, dtype=dtype)
                else:
                    # Uniform sampling between bounds using numpy
                    samples = np.random.uniform(min_val, max_val, n_samples)
                    # Convert to original numeric type using pandas Series.astype
                    # This correctly handles casting to numpy dtypes and pandas nullable dtypes
                    if ptypes.is_integer_dtype(dtype): # Handles both numpy int and pandas Int64, UInt64
                         samples = samples.round() # Round floats to nearest integer
                         synthetic_data[col] = pd.Series(samples).astype(dtype)
                    elif ptypes.is_float_dtype(dtype): # Handles both numpy float and pandas Float64
                         synthetic_data[col] = pd.Series(samples).astype(dtype)
                    else: # Fallback, should be covered by is_numeric_dtype in fit
                         # This branch should ideally not be reached with the corrected fit logic
                         warnings.warn(f"Unexpected numeric dtype {dtype} for column '{col}' during generation. Attempting direct conversion.")
                         try:
                              synthetic_data[col] = pd.Series(samples).astype(dtype)
                         except Exception as e:
                              warnings.warn(f"Could not cast synthetic numeric samples to dtype {dtype} for column '{col}': {e}")
                              fill_value = pd.NA if ptypes.is_extension_array_dtype(dtype) else np.nan
                              synthetic_data[col] = pd.Series([fill_value] * n_samples, dtype=dtype)


            # Generate categorical/object/boolean samples (from unique values)
            elif col in self._unique_values:
                unique_vals = self._unique_values[col] # This includes pd.NA/None if present
                # Ensure list conversion handles potential pd.NA/None values correctly
                sample_pool = unique_vals.tolist() if isinstance(unique_vals, (np.ndarray, pd.Index)) else list(unique_vals)

                # Check if sample_pool contains only NA/None - means original had only NA/None, should be ignored
                if all(pd.isna(x) for x in sample_pool):
                     # This case should be caught in fit and ignored, but safety in generate
                     warnings.warn(f"Column '{col}' has only NA/None unique values learned. Treating as ignored during generation.")
                     fill_value = pd.NA if ptypes.is_extension_array_dtype(dtype) else np.nan
                     synthetic_data[col] = pd.Series([fill_value] * n_samples, dtype=dtype)
                elif unique_vals.size == 0: # Also safety check for empty unique values
                     fill_value = pd.NA if ptypes.is_extension_array_dtype(dtype) else np.nan
                     synthetic_data[col] = pd.Series([fill_value] * n_samples, dtype=dtype)
                else:
                     # Sample uniformly from the unique values (including pd.NA/None)
                     samples = np.random.choice(sample_pool, size=n_samples, replace=True)
                     # Create as a pandas Series, specifying the original dtype
                     # This is crucial for CategoricalDtype, BooleanDtype, and correctly handling object with None
                     synthetic_data[col] = pd.Series(samples, dtype=dtype)

            else:
                # Column exists in _column_dtypes but wasn't in _min_max_bounds, _unique_values, or _ignored_columns.
                # This indicates an unhandled dtype that wasn't explicitly ignored.
                # This case should ideally not be reached with the refined fit logic.
                # Treat as ignored as a final fallback.
                warnings.warn(f"Column '{col}' with dtype {dtype} was not handled by sampler logic but was not ignored. Treating as ignored.")
                fill_value = pd.NA if ptypes.is_extension_array_dtype(dtype) else np.nan
                synthetic_data[col] = pd.Series([fill_value] * n_samples, dtype=dtype)


        # Create DataFrame from the dictionary. Since we created Series with specific dtypes
        # in the dict, the DataFrame constructor should respect them.
        synth_df = pd.DataFrame(synthetic_data)

        # Reindex to ensure correct column order.
        # Dtype consistency is handled by creating Series with specific dtypes above.
        if not synth_df.columns.equals(self._columns_order):
             # This warning indicates an issue in how synthetic_data dict was populated
             warnings.warn("Synthetic DataFrame columns mismatch after internal creation step. Reindexing.")
             synth_df = synth_df.reindex(columns=self._columns_order)

        return synth_df


class LGBMAnomalyDetector(BaseEstimator, TransformerMixin):
    """
    Anomaly detector using an LGBM classifier trained to discriminate
    real data from synthetic data generated uniformly within bounds.
    (Internal class used by RFSynthAnomalyDetector)
    """
    def __init__(self, lgbm_params=None, _sampler_learned_unique_values=None, random_state=None):
        """
        Initializes the internal LGBMAnomalyDetector.

        Parameters
        ----------
        lgbm_params : dict or None, default=None
            Parameters to pass to the underlying LGBMClassifier.
        _sampler_learned_unique_values : dict, default=None
            A dictionary mapping column names to their learned unique values
            from the sampler. Used to identify categorical features.
        random_state : int or None, default=None
            Random state for reproducibility.
        """
        # Note: n_synthetic_ratio is handled by the outer RFSynthAnomalyDetector now
        self.lgbm_params = lgbm_params if lgbm_params is not None else {}
        self.random_state = random_state
        self._sampler_learned_unique_values = _sampler_learned_unique_values if _sampler_learned_unique_values is not None else {}

        self._model = None
        self._is_fitted = False
        self._lgbm_features = None # Store the names of features the LGBM model was trained on
        self._lgbm_categorical_features = None # Store the subset of _lgbm_features that are categorical

    def _fit(self, X_real: pd.DataFrame, X_synthetic: pd.DataFrame):
        """
        Fits the internal LGBM classifier.

        Parameters
        ----------
        X_real : pd.DataFrame
            The real data (subset of relevant columns).
        X_synthetic : pd.DataFrame
            The synthetic data (subset of relevant columns).

        Returns
        -------
        self : object
            The fitted detector instance.
        """
        if X_real.empty:
            warnings.warn("Real data is empty. Cannot train LGBM model.")
            return self # Fitted without model

        if X_synthetic.empty:
             warnings.warn("Synthetic data is empty. Cannot train LGBM model.")
             return self # Fitted without model

        # Ensure columns match and store feature names
        if not X_real.columns.equals(X_synthetic.columns):
             raise ValueError("Columns of real and synthetic data subsets do not match.")

        self._lgbm_features = list(X_real.columns)

        # Combine real and synthetic data for training
        X_combined = pd.concat([X_real, X_synthetic], ignore_index=True)
        y_combined = np.array([1] * len(X_real) + [0] * len(X_synthetic)) # 1 for real, 0 for synthetic


        # Identify categorical features for LGBM in the combined data
        self._lgbm_categorical_features = []
        for col in X_combined.columns: # Iterate over the relevant columns
             dtype = X_combined[col].dtype

             # If the column was handled as unique values by the sampler (categorical, bool, object-as-categorical)
             # AND it's not entirely null in the combined data, AND it's not a numeric type in combined data,
             # THEN it should be treated as a categorical feature in LGBM.
             # Check against the learned unique values passed from the sampler.
             if col in self._sampler_learned_unique_values and not X_combined[col].isnull().all() and not ptypes.is_numeric_dtype(dtype):
                  # Convert to category dtype if it's not already
                  if not isinstance(dtype, pd.CategoricalDtype):
                       try:
                            # Convert to category dtype. Using the unique values from the sampler ensures
                            # all possible categories (including NA if it was present) are included.
                            # Get the categories directly from the sampler's learned values.
                            # Handle pd.NA/None explicitly in categories.
                            categories = list(self._sampler_learned_unique_values[col])
                            # Ensure None/pd.NA are handled correctly in categories list if they exist
                            categories = [cat if pd.notna(cat) else None for cat in categories] # Represent NA as None in categories list
                            if None in categories:
                                 categories.remove(None)
                                 # Create a CategoricalDtype that supports NA
                                 cat_dtype = pd.CategoricalDtype(categories=categories, ordered=False)
                                 X_combined[col] = X_combined[col].astype(cat_dtype)
                            else:
                                 # Standard category dtype
                                 X_combined[col] = X_combined[col].astype('category')

                            # Verify it's now categorical before adding
                            if isinstance(X_combined[col].dtype, pd.CategoricalDtype):
                                 self._lgbm_categorical_features.append(col)
                            else:
                                warnings.warn(f"Attempted to convert column '{col}' to category but failed silently.")


                       except Exception as e:
                            warnings.warn(f"Could not convert column '{col}' to category dtype for LGBM: {e}")
                  else:
                       # It's already category dtype
                       self._lgbm_categorical_features.append(col)


        # Filter out categorical features that ended up being all null in the combined data
        self._lgbm_categorical_features = [
             col for col in self._lgbm_categorical_features
             if col in X_combined.columns and not X_combined[col].isnull().all()
        ]

        # Remove duplicates
        self._lgbm_categorical_features = list(dict.fromkeys(self._lgbm_categorical_features))


        # Train LGBM classifier
        print("Training LGBM classifier...")

        self._model = lgb.LGBMClassifier(**self.lgbm_params)

        # Pass the list of relevant columns and categorical features
        # LightGBM will build its internal representation from X_combined
        self._model.fit(X_combined, y_combined,
                        categorical_feature=self._lgbm_categorical_features if self._lgbm_categorical_features else 'auto',
                        # feature_name=self._lgbm_features # LGBM automatically uses column names from DataFrame
                        )


        self._is_fitted = True
        print("LGBM training complete.")
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes anomaly scores for the input data using the fitted LGBM model.

        Parameters
        ----------
        X : pd.DataFrame
            The data points to score (subset of relevant columns).

        Returns
        -------
        np.ndarray
            Anomaly scores (probability of being synthetic).
        """
        if not self._is_fitted or self._model is None:
             # This state should ideally be handled by the outer RFSynthAnomalyDetector
             raise RuntimeError("Internal LGBM detector is not fitted.")

        if not isinstance(X, pd.DataFrame):
             raise TypeError("Input X must be a pandas DataFrame.")

        # Ensure scoring data has same columns as training data subset
        if not X.columns.equals(self._lgbm_features):
            warnings.warn("Scoring data columns do not match LGBM training columns. Attempting reindexing.")
            # Reindex to match training columns, filling with NaN/NA
            # This might add columns back that were filtered out by the outer detector,
            # but the model prediction should ignore them based on its internal feature_name_
            try:
                 X = X.reindex(columns=self._lgbm_features)
            except Exception as e:
                 raise ValueError(f"Could not reindex scoring data to match LGBM training columns: {e}") from e


        # Ensure categorical columns in scoring data have the correct dtype
        # Use the stored list of categorical features used during training
        for col in self._lgbm_categorical_features:
             # Only attempt conversion if column exists and has non-null values
             if col in X.columns and not X[col].isnull().all():
                 # Check the current dtype in the scoring data
                 if not isinstance(X[col].dtype, pd.CategoricalDtype):
                      try:
                           # Convert to category. Need to ensure categories align.
                           # The safest way is often to convert to the exact dtype object from the training data.
                           # But we don't store the full combined dtype.
                           # A simpler approach: just cast to category. LGBM is often forgiving.
                            X[col] = X[col].astype('category')
                      except Exception as e:
                            warnings.warn(f"Could not convert scoring column '{col}' to category dtype for LGBM: {e}")


        # Predict probability of being synthetic (class 0)
        # predict_proba returns shape (n_samples, 2) -> [prob_class_0, prob_class_1]
        # LGBM model expects columns in the order it was trained on. Using model.feature_name_ handles this.
        try:
             X_reordered = X[self._model.feature_name_]
        except Exception as e:
             warnings.warn(f"Could not reorder scoring columns using model feature names: {e}. Proceeding with current column order.")
             X_reordered = X # Fallback


        anomaly_scores = self._model.predict_proba(X_reordered)[:, 0] # Probability of being class 0 (synthetic)

        return anomaly_scores

    # No predict method here; it's handled by the outer class calling score and applying threshold