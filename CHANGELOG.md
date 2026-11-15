# Changelog

## FoodWise 2.1

- Supervised rating-prediction model (`src/ml_model.py`, gradient boosting) used to estimate ratings of unrated listings; evaluated with a split grouped by restaurant name. New notebook 04 and an Insights page section. Estimates are labelled in the app.

- Dish photos on cards from Wikipedia / Wikimedia Commons (name rules, then cuisine tags, several candidates, cache, emoji fallback, on/off switch).
- New dashboard UI (sidebar stats, top search, banner, filter bar, card grid) with live filtering.
- Fixed veg detection: "Non Veg" / "Veg & Non Veg" names are no longer flagged as pure veg.
- Chain outlets are de-duplicated in recommendations (`dedupe_chains=True`).
- Vectorised the diversity step (about 30 s to about 0.5 s on an unfiltered search).
- New benchmark: random and cuisine-filtered top-rated baselines, 10 scenarios, rating / price / diversity metrics.
- 42 unit tests; notebooks re-run with outputs; fixed a broken column name in notebook 02.
- Streamlit minimum version raised to 1.40.

## FoodWise 2.0

- Rebuilt data validation, parsing and feature preparation.
- Replaced dense/pairwise-heavy discovery paths with sparse TF-IDF representations and targeted cosine comparisons.
- Added multi-factor explainable ranking and diversity-aware result selection.
- Added Discover, Similar Restaurants, Explore Catalogue, Compare, Saved Shortlist, Insights & Model Lab, and About views.
- Added CSV exports, dataset visualizations, benchmark execution and unit tests.
- Removed the duplicate root-level raw dataset copy; `data/raw/zomato_dataset.csv` is the single source of truth.
- Preserved the original exploratory notebooks.
