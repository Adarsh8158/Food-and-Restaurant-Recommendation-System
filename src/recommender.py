"""Sparse TF-IDF recommendation engine with transparent multi-factor ranking."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, WEIGHTS, DEFAULT_TOP_K
from src.ml_model import MIN_TRAIN_ROWS, RatingPredictor
from src.preprocessing import load_raw_data, clean_dataset


class RestaurantRecommender:
    def __init__(self, data_path=None, dataframe=None, use_rating_model=True):
        if dataframe is not None:
            self.df = dataframe.copy().reset_index(drop=True)
        else:
            path = data_path or RAW_DATA_PATH
            if str(path).endswith("cleaned_restaurants.csv") and PROCESSED_DATA_PATH.exists():
                loaded = pd.read_csv(path)
                if "cuisine_list" in loaded:
                    loaded["cuisine_list"] = loaded["cuisine_list"].fillna("").apply(lambda x: [v.strip() for v in re.split(r"[;|]", str(x)) if v.strip()])
                    self.df = loaded
                else:
                    self.df = clean_dataset(loaded)
            else:
                self.df = clean_dataset(load_raw_data(path))
        if "content_soup" not in self.df:
            raise ValueError("Dataframe must be preprocessed with clean_dataset first.")
        self.cities = sorted(self.df["city"].dropna().unique().tolist())
        self.cuisines = sorted({c for row in self.df["cuisine_list"] for c in row})
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["content_soup"].fillna(""))
        self._add_rating_estimates(use_rating_model)

    def _add_rating_estimates(self, use_rating_model):
        """Model-estimated rating for unrated listings (NaN for rated ones, or if too little data)."""
        self.rating_model = None
        self.df["predicted_rating"] = np.nan
        unrated = self.df["rating_numeric"].isna()
        if use_rating_model and unrated.any() and self.df["rating_numeric"].notna().sum() >= MIN_TRAIN_ROWS:
            self.rating_model = RatingPredictor().fit(self.df)
            self.df.loc[unrated, "predicted_rating"] = self.rating_model.predict(self.df[unrated])

    @staticmethod
    def _normalise_weights(weights):
        values = {k: max(0.0, float(v)) for k, v in (weights or WEIGHTS).items() if k in WEIGHTS}
        total = sum(values.values())
        return {k: v / total for k, v in values.items()} if total else WEIGHTS.copy()

    def _base_mask(self, city=None, budget_max=None, min_rating=0, include_unrated=True,
                   dietary_preference="Any", available_only=False):
        mask = pd.Series(True, index=self.df.index)
        if city:
            mask &= self.df["city"].eq(city)
        if budget_max is not None:
            mask &= self.df["price_numeric"].notna() & self.df["price_numeric"].le(float(budget_max))
        if min_rating and min_rating > 0:
            mask &= self.df["rating_numeric"].ge(float(min_rating)) | (include_unrated & self.df["rating_numeric"].isna())
        if dietary_preference == "Veg":
            mask &= self.df["is_pure_veg"].astype(bool)
        elif dietary_preference == "Non-Veg":
            mask &= ~self.df["is_pure_veg"].astype(bool)
        if available_only:
            mask &= self.df["delivery_status"].eq("Delivery time listed")
        return mask

    def _query_similarity(self, cuisines=None, city=None):
        terms = list(cuisines or [])
        query = " ".join(terms + ([city] if city else []))
        if not query.strip():
            return np.zeros(len(self.df), dtype=float)
        vector = self.vectorizer.transform([query.lower()])
        return cosine_similarity(vector, self.tfidf_matrix).ravel()

    def _score_candidates(self, indices, content_scores, budget_max=None, weights=None):
        frame = self.df.loc[indices].copy()
        frame["_catalog_index"] = frame.index
        w = self._normalise_weights(weights)
        # Unrated listings use the model's estimate (or a flat 3.5 if no model is available).
        rating = frame["rating_numeric"].fillna(frame["predicted_rating"]).fillna(3.5).clip(2, 5).sub(2).div(3).to_numpy()
        if budget_max and budget_max > 0:
            prices = frame["price_numeric"].to_numpy(dtype=float)
            price_score = np.where(np.isfinite(prices), np.clip(1 - np.maximum(prices - budget_max * 0.75, 0) / budget_max, 0, 1), 0.5)
        else:
            median = self.df["price_numeric"].median()
            prices = frame["price_numeric"].to_numpy(dtype=float)
            price_score = np.where(np.isfinite(prices), 1 - np.minimum(prices / max(median * 2, 1), 1), 0.5)
        delivery = frame["delivery_time_mins"].to_numpy(dtype=float)
        delivery_score = np.where(np.isfinite(delivery), np.clip(1 - (delivery - 20) / 60, 0, 1), 0.45)
        safety_text = frame["safety_label"].fillna("").str.lower()
        safety_score = np.where(safety_text.str.contains("max safety", regex=False), 1.0,
                                np.where(safety_text.str.contains("who", regex=False), 0.8, 0.5))
        content = np.asarray(content_scores, dtype=float)
        frame["content_score"] = content
        frame["rating_score"] = rating
        frame["price_score"] = price_score
        frame["delivery_score"] = delivery_score
        frame["safety_label_score"] = safety_score
        frame["ranking_score"] = (
            w.get("content_similarity", 0) * content + w.get("rating", 0) * rating +
            w.get("price_compatibility", 0) * price_score + w.get("delivery_speed", 0) * delivery_score +
            w.get("safety_label", 0) * safety_score
        )
        frame["match_score"] = (frame["ranking_score"].clip(0, 1) * 100).round(1)
        frame["explanation"] = frame.apply(self._explain, axis=1)
        return frame

    @staticmethod
    def _explain(row):
        reasons = []
        if row["content_score"] >= 0.25:
            reasons.append("Cuisine and profile match")
        if pd.notna(row["rating_numeric"]):
            reasons.append(f"Rated {row['rating_numeric']:.1f}/5")
        elif pd.notna(row.get("predicted_rating")):
            reasons.append(f"Unrated, model estimate {row['predicted_rating']:.1f}/5")
        if pd.notna(row["price_numeric"]):
            reasons.append(f"₹{row['price_numeric']:.0f} average price for one")
        if pd.notna(row["delivery_time_mins"]):
            reasons.append(f"{row['delivery_time_mins']:.0f} min listed delivery time")
        return " · ".join(reasons[:4]) if reasons else "Matches the selected catalogue filters"

    def _diverse_top_k(self, scored, top_k, diversity=0.12):
        if scored.empty:
            return scored
        # Maximal marginal relevance over the best candidates. The pairwise similarity
        # matrix is computed once (vectorised) instead of one call per candidate pair.
        candidates = scored.sort_values("ranking_score", ascending=False).head(300 + top_k)
        idx = candidates.index.to_numpy()
        relevance = candidates["ranking_score"].to_numpy(dtype=float)
        sim = cosine_similarity(self.tfidf_matrix[idx])
        selected = [0]
        redundancy = sim[0].copy()
        available = np.ones(len(idx), dtype=bool)
        available[0] = False
        while available.any() and len(selected) < top_k:
            value = (1 - diversity) * relevance - diversity * redundancy
            value[~available] = -np.inf
            chosen = int(np.argmax(value))
            selected.append(chosen)
            available[chosen] = False
            redundancy = np.maximum(redundancy, sim[chosen])
        return scored.loc[idx[selected]].copy().reset_index(drop=True)

    def recommend(self, city=None, cuisines=None, budget_max=None, min_rating=0,
                  include_new=True, dietary_preference="Any", available_only=False,
                  top_k=DEFAULT_TOP_K, custom_weights=None, diversity=0.12, dedupe_chains=True):
        mask = self._base_mask(city, budget_max, min_rating, include_new, dietary_preference, available_only)
        indices = self.df.index[mask].to_numpy()
        if len(indices) == 0:
            return self.df.iloc[0:0].copy(), {"status": "no_match", "candidates_found": 0,
                "message": "No restaurants match all selected filters. Try widening your budget or rating range."}
        similarity = self._query_similarity(cuisines, city)
        scored = self._score_candidates(indices, similarity[indices], budget_max, custom_weights)
        # If no cuisine was selected, avoid pretending the content score is preference relevance.
        if not cuisines:
            scored["content_score"] = 0.5
            w = self._normalise_weights(custom_weights)
            scored["ranking_score"] = (
                w.get("content_similarity", 0) * 0.5 + w.get("rating", 0) * scored["rating_score"] +
                w.get("price_compatibility", 0) * scored["price_score"] +
                w.get("delivery_speed", 0) * scored["delivery_score"] +
                w.get("safety_label", 0) * scored["safety_label_score"]
            )
            scored["match_score"] = (scored["ranking_score"].clip(0, 1) * 100).round(1)
        if dedupe_chains:
            # Chains (e.g. Pizza Hut) list many outlets with the same name; keep the best-scoring one.
            scored = (scored.sort_values("ranking_score", ascending=False)
                      .drop_duplicates(subset=["Restaurant Name"], keep="first"))
        results = self._diverse_top_k(scored, max(1, int(top_k)), diversity)
        return results, {"status": "ok", "candidates_found": len(indices), "message": "Recommendations generated from catalogue attributes."}

    def get_similar_restaurants(self, restaurant_name, city=None, top_k=8, source_city=None, source_index=None):
        matches = self.df[self.df["Restaurant Name"].str.casefold().eq(str(restaurant_name).casefold())]
        if source_city:
            matches = matches[matches["city"].eq(source_city)]
        if source_index is not None and source_index in self.df.index:
            source_idx = int(source_index)
        elif not matches.empty:
            source_idx = int(matches.index[0])
        else:
            return self.df.iloc[0:0].assign(similarity_score=[])
        scores = cosine_similarity(self.tfidf_matrix[source_idx], self.tfidf_matrix).ravel()
        mask = self.df.index != source_idx
        if city:
            mask &= self.df["city"].eq(city)
        indices = self.df.index[mask].to_numpy()
        if len(indices) == 0:
            return self.df.iloc[0:0].assign(similarity_score=[])
        out = self.df.loc[indices].copy()
        out["similarity_score"] = (scores[indices] * 100).round(1)
        return out.sort_values("similarity_score", ascending=False).head(top_k).reset_index(drop=True)

    def search_restaurants(self, query, city=None, limit=30):
        mask = self.df["Restaurant Name"].str.contains(str(query), case=False, regex=False, na=False)
        if city:
            mask &= self.df["city"].eq(city)
        return self.df.loc[mask].head(limit).copy()
