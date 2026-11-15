"""Transparent offline checks for catalogue-based recommendation quality.

IMPORTANT: the catalogue has no user interactions, so nothing here measures real user
satisfaction. These are hand-authored filter scenarios. "Cuisine Match" only counts how many
recommended restaurants share a cuisine tag with the query, which the content-based methods
optimise directly, so it is a sanity check, not proof of quality. The metrics that separate the
methods are the other ones: average rating, average price and intra-list diversity.

All methods use the same eligibility filters (city, budget, minimum rating) and the same
chain de-duplication (one listing per restaurant name), so the comparison is fair.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

SCENARIOS = [
    {"city": "Mumbai", "cuisines": ["Italian", "Pizza"], "budget_max": 300, "min_rating": 3.5},
    {"city": "Lucknow", "cuisines": ["North Indian", "Mughlai"], "budget_max": 250, "min_rating": 3.5},
    {"city": "Bangalore", "cuisines": ["South Indian", "Chinese"], "budget_max": 250, "min_rating": 3.5},
    {"city": "NCR", "cuisines": ["Biryani", "North Indian"], "budget_max": 300, "min_rating": 3.5},
    {"city": "Pune", "cuisines": ["Cafe", "Italian"], "budget_max": 400, "min_rating": 3.5},
    {"city": "Hyderabad", "cuisines": ["Biryani", "Mughlai"], "budget_max": 250, "min_rating": 3.5},
    {"city": "Kolkata", "cuisines": ["Chinese", "Rolls"], "budget_max": 200, "min_rating": 3.5},
    {"city": "Chennai", "cuisines": ["South Indian", "Beverages"], "budget_max": 200, "min_rating": 3.5},
    {"city": "Jaipur", "cuisines": ["Desserts", "Ice Cream"], "budget_max": 200, "min_rating": 3.5},
    {"city": "Ahmedabad", "cuisines": ["Fast Food", "Street Food"], "budget_max": 200, "min_rating": 3.5},
]
MODELS = ["Random (eligible)", "Top-rated (cuisine filter)", "Content-based", "FoodWise hybrid"]


class RecommendationEvaluator:
    def __init__(self, recommender, seed=42):
        self.rec = recommender
        self.seed = seed
        self.last_detail = pd.DataFrame()

    @staticmethod
    def _cuisine_match(frame, scenario, k):
        if frame.empty:
            return 0.0
        wanted = {c.casefold() for c in scenario["cuisines"]}
        hit = frame.head(k)["cuisine_list"].apply(lambda row: bool(wanted & {x.casefold() for x in row}))
        return float(hit.mean())

    def _diversity(self, frame):
        if len(frame) < 2:
            return 0.0
        idx = frame["_catalog_index"].to_numpy() if "_catalog_index" in frame.columns else frame.index.to_numpy()
        sim = cosine_similarity(self.rec.tfidf_matrix[idx])
        upper = sim[np.triu_indices_from(sim, k=1)]
        return float(1 - upper.mean()) if len(upper) else 0.0

    def _row(self, model, scenario, frame, k):
        frame = frame.head(k)
        return {
            "Model": model,
            "Scenario": scenario["city"],
            f"Cuisine Match@{k}": round(self._cuisine_match(frame, scenario, k), 3),
            "Avg Rating": round(float(frame["rating_numeric"].mean()), 3) if len(frame) else np.nan,
            "Avg Price (Rs)": round(float(frame["price_numeric"].mean()), 1) if len(frame) else np.nan,
            "Diversity": round(self._diversity(frame), 3),
            "Items Returned": len(frame),
        }

    def run_benchmark(self, k=5):
        rng = np.random.default_rng(self.seed)
        rows, seen = [], {m: set() for m in MODELS}
        for sc in SCENARIOS:
            mask = self.rec._base_mask(sc["city"], sc["budget_max"], sc["min_rating"], True, "Any", False)
            eligible = self.rec.df.loc[mask]
            if eligible.empty:
                continue
            eligible = eligible.drop_duplicates(subset=["Restaurant Name"], keep="first")
            wanted = {c.casefold() for c in sc["cuisines"]}
            on_cuisine = eligible[eligible["cuisine_list"].apply(lambda r: bool(wanted & {x.casefold() for x in r}))]

            random_pick = eligible.iloc[rng.permutation(len(eligible))[:k]]
            top_rated = on_cuisine.sort_values("rating_numeric", ascending=False, kind="mergesort", na_position="last").head(k)
            sims = self.rec._query_similarity(sc["cuisines"], sc["city"])
            content = eligible.assign(_sim=sims[eligible.index.to_numpy()]).sort_values("_sim", ascending=False, kind="mergesort").head(k)
            hybrid, _ = self.rec.recommend(**sc, top_k=k)

            for name, frame in zip(MODELS, [random_pick, top_rated, content, hybrid]):
                rows.append(self._row(name, sc, frame, k))
                seen[name].update(zip(frame["Restaurant Name"].astype(str), frame["city"].astype(str)))
        detail = pd.DataFrame(rows)
        self.last_detail = detail
        if detail.empty:
            return detail
        summary = detail.groupby("Model", sort=False).agg(
            **{f"Mean Cuisine Match@{k}": (f"Cuisine Match@{k}", "mean"),
               "Mean Rating": ("Avg Rating", "mean"),
               "Mean Price (Rs)": ("Avg Price (Rs)", "mean"),
               "Mean Diversity": ("Diversity", "mean")}
        ).round(3).reset_index()
        summary["Unique Items"] = summary["Model"].map(lambda m: len(seen[m]))
        return summary
