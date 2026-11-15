"""Supervised model that estimates a restaurant's rating from catalogue features.

Target: the numeric rating of rated listings. Features: cuisine tags, city, price, listed delivery
time, veg hint, safety label and price tier. The restaurant NAME is not a feature.

Why: ~16% of listings are unrated or new. The ranking used a flat 3.5 for them, which is well
below the catalogue average (about 3.9). The model gives each unrated listing an *estimate* that
reflects its cuisine, city and price level. It is used only inside the ranking and is always
labelled as an estimate in the app. It is never shown as a real rating.

Honest expectation: ratings are noisy and the catalogue has no review counts, so the model can
only explain a small share of the variation (R^2 around 0.1). It is evaluated on a split that keeps
all outlets of one restaurant name on the same side, so chains cannot leak between train and test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

FEATURES = ["cuisine_text", "city", "safety_label", "price_tier", "log_price", "delivery", "veg"]
MIN_TRAIN_ROWS = 2000
OLD_FALLBACK_RATING = 3.5  # what the ranking used for unrated listings before the model


def _split_tags(text):
    return text.split("|")


def _to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else x


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs from a cleaned catalogue frame (no rating information is used)."""
    return pd.DataFrame({
        "cuisine_text": df["cuisine_list"].apply(lambda tags: "|".join(tags)),
        "city": df["city"].astype(str),
        "safety_label": df["safety_label"].astype(str),
        "price_tier": df["price_tier"].astype(str),
        "log_price": np.log1p(df["price_numeric"].astype(float)),
        "delivery": df["delivery_time_mins"].astype(float),
        "veg": df["is_pure_veg"].astype(int),
    }, index=df.index)[FEATURES]


def _preprocessor():
    return ColumnTransformer([
        ("cuisine", CountVectorizer(tokenizer=_split_tags, token_pattern=None, binary=True, min_df=5), "cuisine_text"),
        ("category", OneHotEncoder(handle_unknown="ignore", min_frequency=30), ["city", "safety_label", "price_tier"]),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scale", StandardScaler())]), ["log_price", "delivery", "veg"]),
    ])


def _ridge():
    return Pipeline([("prep", _preprocessor()), ("model", Ridge(alpha=3.0))])


def _boosting(seed):
    return Pipeline([("prep", _preprocessor()),
                     ("dense", FunctionTransformer(_to_dense, accept_sparse=True)),
                     ("model", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=24,
                                                             l2_regularization=1.0, random_state=seed))])


class RatingPredictor:
    def __init__(self, seed=42):
        self.seed = seed
        self.model = None
        self.importance_ = pd.DataFrame()

    @staticmethod
    def _rated(df):
        rated = df[df["rating_numeric"].notna()]
        if len(rated) < MIN_TRAIN_ROWS:
            raise ValueError(f"Need at least {MIN_TRAIN_ROWS} rated listings to train, found {len(rated)}.")
        return rated

    def fit(self, df):
        """Train the final model on every rated listing."""
        rated = self._rated(df)
        self.model = _boosting(self.seed).fit(build_features(rated), rated["rating_numeric"])
        return self

    def predict(self, df):
        """Estimated rating for any listings, clipped to the valid 2-5 range."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        return np.clip(self.model.predict(build_features(df)), 2.0, 5.0)

    def evaluate(self, df, test_size=0.2):
        """Hold-out comparison (split grouped by restaurant name). Also fills ``importance_``."""
        rated = self._rated(df).reset_index(drop=True)
        X, y = build_features(rated), rated["rating_numeric"]
        groups = rated["Restaurant Name"].str.lower().str.strip()
        train, test = next(GroupShuffleSplit(1, test_size=test_size, random_state=self.seed).split(X, y, groups))
        candidates = {
            f"Constant {OLD_FALLBACK_RATING} (old fallback)": DummyRegressor(strategy="constant", constant=OLD_FALLBACK_RATING),
            "Predict the average": DummyRegressor(strategy="mean"),
            "Ridge regression": _ridge(),
            "Gradient boosting (used)": _boosting(self.seed),
        }
        rows, fitted = [], {}
        for name, model in candidates.items():
            model.fit(X.iloc[train], y.iloc[train])
            pred = model.predict(X.iloc[test])
            fitted[name] = model
            rows.append({"Model": name,
                         "MAE": round(mean_absolute_error(y.iloc[test], pred), 3),
                         "RMSE": round(float(np.sqrt(mean_squared_error(y.iloc[test], pred))), 3),
                         "R2": round(r2_score(y.iloc[test], pred), 3)})
        importance = permutation_importance(fitted["Gradient boosting (used)"], X.iloc[test], y.iloc[test],
                                            scoring="neg_mean_absolute_error", n_repeats=3, random_state=self.seed)
        labels = {"cuisine_text": "Cuisine tags", "city": "City", "safety_label": "Safety label", "price_tier": "Price tier",
                  "log_price": "Price", "delivery": "Delivery time", "veg": "Veg hint"}
        self.importance_ = (pd.DataFrame({"Feature": [labels[c] for c in X.columns],
                                          "MAE increase when shuffled": importance.importances_mean.round(4)})
                            .sort_values("MAE increase when shuffled", ascending=False).reset_index(drop=True))
        self.test_size_ = len(test)
        self.train_size_ = len(train)
        return pd.DataFrame(rows)
