import unittest

import numpy as np
import pandas as pd

from src.evaluation import RecommendationEvaluator
from src.ml_model import FEATURES, RatingPredictor, build_features
from src.food_images import WikiImageResolver, all_known_titles, candidate_titles
from src.preprocessing import (clean_dataset, extract_price, extract_rating, is_pure_veg_name,
                               load_raw_data, parse_delivery)
from src.recommender import RestaurantRecommender


def row(name, rating, cuisine, price, delivery, safety, city):
    return {"Restaurant Name": name, "Rating": rating, "Cuisine": cuisine, "Average Price": price,
            "Average Delivery Time": delivery, "Safety Measure": safety, "Location": city}


SAMPLE = pd.DataFrame([
    row("Mumbai Pizza House", "4.5", "Pizza, Italian", "₹250 for one", "25 min", "Follows all Max Safety measures", "Mumbai"),
    row("Mumbai Dosa Corner", "4.1", "South Indian", "₹120 for one", "30 min", "WHO protocol", "Mumbai"),
    row("New Cafe", "New", "Cafe, Bakery", "₹180 for one", "Not accepting orders", "Not specified", "Pune"),
    row("Pure Veg Thali", "4.0", "North Indian", "₹150 for one", "20 min", "WHO protocol", "Mumbai"),
    row("Abid Non Veg", "3.9", "North Indian, Mughlai", "₹200 for one", "35 min", "WHO protocol", "Mumbai"),
    row("Chain Pizza", "4.2", "Pizza, Fast Food", "₹200 for one", "28 min", "WHO protocol", "Mumbai"),
    row("Chain Pizza", "4.4", "Pizza, Fast Food", "₹200 for one", "22 min", "WHO protocol", "Mumbai"),
    row("Chain Pizza", "3.8", "Pizza, Fast Food", "₹200 for one", "40 min", "WHO protocol", "Mumbai"),
    row("Closed Place", "4.0", "Chinese", "₹100 for one", "Permanently closed", "WHO protocol", "Mumbai"),
    row("No Cuisine", "4.0", "", "₹100 for one", "20 min", "WHO protocol", "Mumbai"),
])


class ParsingTests(unittest.TestCase):
    def test_price(self):
        self.assertEqual(extract_price("₹250 for one"), 250)
        self.assertEqual(extract_price("₹1,200 for one"), 1200)
        self.assertTrue(np.isnan(extract_price("for one")))
        self.assertTrue(np.isnan(extract_price(None)))

    def test_rating(self):
        self.assertEqual(extract_rating("4.3"), 4.3)
        for bad in ["-", "New", "", None, "7.5"]:
            self.assertTrue(np.isnan(extract_rating(bad)), bad)

    def test_delivery(self):
        self.assertEqual(parse_delivery("25 min"), (25.0, "Delivery time listed"))
        self.assertEqual(parse_delivery("Currently not accepting orders")[1], "Not Accepting Orders")
        self.assertEqual(parse_delivery("Opens at 5pm")[1], "Unknown")
        self.assertEqual(parse_delivery("Temporarily closed")[1], "Temporarily Closed")

    def test_veg_heuristic(self):
        for name in ["Pure Veg Meals", "Anand Vegetarian", "Veg Sutra", "Jain Sweets"]:
            self.assertTrue(is_pure_veg_name(name), name)
        for name in ["Abid Non Veg", "Tasty Veg Non Veg", "Salman Special Veg & Non Veg", "Gauri Non-Veg",
                     "Shri Nonveg Restaurant", "Eat No Veg", "Burger King", None]:
            self.assertFalse(is_pure_veg_name(name), name)


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.cleaned = clean_dataset(SAMPLE)

    def test_core_fields(self):
        r = self.cleaned.iloc[0]
        self.assertEqual(r["city"], "Mumbai")
        self.assertEqual(r["price_numeric"], 250)
        self.assertEqual(r["delivery_time_mins"], 25)
        self.assertEqual(r["cuisine_list"], ["Pizza", "Italian"])
        self.assertEqual(r["price_tier"], "Mid-range")

    def test_unusable_rows_removed(self):
        names = self.cleaned["Restaurant Name"].tolist()
        self.assertNotIn("Closed Place", names)
        self.assertNotIn("No Cuisine", names)

    def test_unrated_is_flagged(self):
        r = self.cleaned[self.cleaned["Restaurant Name"] == "New Cafe"].iloc[0]
        self.assertTrue(np.isnan(r["rating_numeric"]))
        self.assertEqual(r["rating_status"], "Unrated / New")

    def test_veg_flags(self):
        flags = dict(zip(self.cleaned["Restaurant Name"], self.cleaned["is_pure_veg"]))
        self.assertTrue(flags["Pure Veg Thali"])
        self.assertFalse(flags["Abid Non Veg"])

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError):
            clean_dataset(SAMPLE.drop(columns=["Cuisine"]))

    def test_raw_file_loads(self):
        df = load_raw_data()
        self.assertGreater(len(df), 40000)


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.rec = RestaurantRecommender(dataframe=clean_dataset(SAMPLE))

    def test_city_and_budget(self):
        res, meta = self.rec.recommend(city="Mumbai", cuisines=["Pizza"], budget_max=200, min_rating=0, top_k=5)
        self.assertEqual(meta["status"], "ok")
        self.assertTrue((res["city"] == "Mumbai").all())
        self.assertTrue((res["price_numeric"] <= 200).all())

    def test_min_rating(self):
        res, _ = self.rec.recommend(city="Mumbai", min_rating=4.2, include_new=False, top_k=10)
        self.assertTrue((res["rating_numeric"] >= 4.2).all())

    def test_no_match_is_graceful(self):
        res, meta = self.rec.recommend(city="Mumbai", budget_max=10, min_rating=4.9)
        self.assertTrue(res.empty)
        self.assertEqual(meta["status"], "no_match")

    def test_veg_and_non_veg_filters(self):
        veg, _ = self.rec.recommend(city="Mumbai", dietary_preference="Veg", min_rating=0)
        non, _ = self.rec.recommend(city="Mumbai", dietary_preference="Non-Veg", min_rating=0)
        self.assertEqual(veg["Restaurant Name"].tolist(), ["Pure Veg Thali"])
        self.assertNotIn("Pure Veg Thali", non["Restaurant Name"].tolist())
        self.assertIn("Abid Non Veg", non["Restaurant Name"].tolist())

    def test_chain_dedupe(self):
        res, _ = self.rec.recommend(city="Mumbai", cuisines=["Pizza"], min_rating=0, top_k=10)
        self.assertEqual(res["Restaurant Name"].tolist().count("Chain Pizza"), 1)
        res2, _ = self.rec.recommend(city="Mumbai", cuisines=["Pizza"], min_rating=0, top_k=10, dedupe_chains=False)
        self.assertEqual(res2["Restaurant Name"].tolist().count("Chain Pizza"), 3)

    def test_top_k_and_score_range(self):
        res, _ = self.rec.recommend(city="Mumbai", min_rating=0, top_k=2)
        self.assertEqual(len(res), 2)
        self.assertTrue(res["match_score"].between(0, 100).all())

    def test_cuisine_match_ranks_first(self):
        res, _ = self.rec.recommend(city="Mumbai", cuisines=["South Indian"], min_rating=0, top_k=3)
        self.assertEqual(res.iloc[0]["Restaurant Name"], "Mumbai Dosa Corner")

    def test_weights_are_normalised(self):
        w = RestaurantRecommender._normalise_weights({"content_similarity": 2, "rating": 2})
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertEqual(RestaurantRecommender._normalise_weights({"rating": 0})["rating"], 0.25)  # falls back to defaults

    def test_similar_excludes_source(self):
        res = self.rec.get_similar_restaurants("Mumbai Pizza House", city="Mumbai", top_k=3)
        self.assertNotIn("Mumbai Pizza House", res["Restaurant Name"].tolist())

    def test_similar_unknown_name(self):
        self.assertTrue(self.rec.get_similar_restaurants("Does Not Exist").empty)

    def test_search(self):
        self.assertEqual(len(self.rec.search_restaurants("pizza")), 4)  # 1 house + 3 chain rows (after cleaning)


class FoodImageTests(unittest.TestCase):
    def test_name_beats_cuisine(self):
        titles = candidate_titles("Hyderabad Biryani House", ["Fast Food"])
        self.assertTrue(set(titles) <= {"Biryani", "Hyderabadi biryani"})
        self.assertIn("Biryani", titles)

    def test_specific_tag_beats_generic_tag(self):
        titles = candidate_titles("Some Place", ["Fast Food", "Beverages", "Burger"])
        self.assertIn("Hamburger", titles)

    def test_unknown_cuisine_has_no_photo_plan(self):
        self.assertEqual(candidate_titles("Some Place", ["Zzz"]), [])
        self.assertEqual(candidate_titles(None, []), [])

    def test_choice_is_stable_per_restaurant_and_varies_across_restaurants(self):
        a = candidate_titles("Place A", ["North Indian"])
        self.assertEqual(a, candidate_titles("Place A", ["North Indian"]))
        firsts = {candidate_titles(f"Place {i}", ["North Indian"])[0] for i in range(40)}
        self.assertGreater(len(firsts), 1)

    def test_every_cuisine_tag_in_catalogue_is_known_or_generic(self):
        from src.food_images import CUISINE_MAP
        rec = RestaurantRecommender()
        counts = pd.Series([t for row in rec.df["cuisine_list"] for t in row]).str.lower().value_counts()
        unmapped_share = counts[~counts.index.isin(CUISINE_MAP)].sum() / counts.sum()
        self.assertLess(unmapped_share, 0.01)

    def test_resolver_uses_first_candidate_with_a_photo(self):
        pages = {"A": "", "B": "https://img/b.jpg", "C": "https://img/c.jpg"}
        r = WikiImageResolver(cache_path=None, fetch=lambda t: pages[t])
        self.assertEqual(r.resolve(["A", "B", "C"]), "https://img/b.jpg")
        self.assertIsNone(r.resolve(["A"]))

    def test_resolver_caches_results(self):
        calls = []
        r = WikiImageResolver(cache_path=None, fetch=lambda t: calls.append(t) or "https://img/x.jpg")
        r.lookup("X"); r.lookup("X")
        self.assertEqual(calls, ["X"])

    def test_resolver_never_raises_and_pauses_when_offline(self):
        calls = []
        def broken(title):
            calls.append(title)
            raise OSError("offline")
        r = WikiImageResolver(cache_path=None, fetch=broken, breaker_after=3)
        for t in ["A", "B", "C", "D", "E"]:
            self.assertIsNone(r.lookup(t))
        self.assertEqual(len(calls), 3)  # after 3 failures it stops trying for a while
        self.assertEqual(r.cache, {})    # network failures are not cached as "no photo"

    def test_disk_cache_round_trip(self):
        import json, tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "cache.json"
            r = WikiImageResolver(cache_path=path, fetch=lambda t: "https://img/p.jpg")
            r.prefetch(["Pizza", "Biryani"])
            self.assertEqual(set(json.loads(path.read_text(encoding="utf-8"))), {"Pizza", "Biryani"})
            r2 = WikiImageResolver(cache_path=path, fetch=lambda t: self.fail("should use cache"))
            self.assertEqual(r2.lookup("Pizza"), "https://img/p.jpg")

    def test_title_list_is_not_empty(self):
        self.assertGreater(len(all_known_titles()), 100)


class RatingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = RestaurantRecommender()  # real catalogue, trains the rating model
        cls.predictor = RatingPredictor()
        cls.metrics = cls.predictor.evaluate(cls.rec.df).set_index("Model")

    def test_features_do_not_leak_rating_or_name(self):
        feats = build_features(self.rec.df)
        self.assertEqual(list(feats.columns), FEATURES)
        for banned in ["rating_numeric", "predicted_rating", "Restaurant Name", "Rating"]:
            self.assertNotIn(banned, feats.columns)

    def test_too_little_data_is_refused(self):
        with self.assertRaises(ValueError):
            RatingPredictor().fit(clean_dataset(SAMPLE))

    def test_model_beats_baselines_on_held_out_restaurants(self):
        mae = self.metrics["MAE"]
        used = mae["Gradient boosting (used)"]
        self.assertLess(used, mae["Predict the average"])
        self.assertLess(used, mae["Constant 3.5 (old fallback)"] - 0.15)  # far better than the old flat guess
        self.assertGreater(self.metrics.loc["Gradient boosting (used)", "R2"], 0.05)

    def test_evaluation_is_deterministic_and_reports_importance(self):
        again = RatingPredictor().evaluate(self.rec.df).set_index("Model")
        pd.testing.assert_frame_equal(again, self.metrics)
        self.assertEqual(len(self.predictor.importance_), len(FEATURES))
        self.assertEqual(self.predictor.importance_.iloc[0]["Feature"], "Cuisine tags")

    def test_estimates_only_for_unrated_and_in_valid_range(self):
        df = self.rec.df
        unrated = df["rating_numeric"].isna()
        self.assertTrue(df.loc[unrated, "predicted_rating"].between(2, 5).all())
        self.assertTrue(df.loc[unrated, "predicted_rating"].notna().all())
        self.assertTrue(df.loc[~unrated, "predicted_rating"].isna().all())

    def test_ranking_uses_estimate_for_unrated_listings(self):
        df = self.rec.df
        idx = df.index[df["rating_numeric"].isna()][:5]
        scored = self.rec._score_candidates(idx, np.zeros(len(idx)))
        expected = (df.loc[idx, "predicted_rating"].to_numpy() - 2) / 3
        np.testing.assert_allclose(scored["rating_score"].to_numpy(), expected, atol=1e-9)

    def test_small_catalogue_skips_the_model(self):
        small = RestaurantRecommender(dataframe=clean_dataset(SAMPLE))
        self.assertIsNone(small.rating_model)
        self.assertTrue(small.df["predicted_rating"].isna().all())

    def test_model_can_be_turned_off(self):
        off = RestaurantRecommender(use_rating_model=False)
        self.assertIsNone(off.rating_model)
        self.assertTrue(off.df["predicted_rating"].isna().all())


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = RestaurantRecommender()  # real catalogue

    def test_benchmark_shape_and_determinism(self):
        ev = RecommendationEvaluator(self.rec)
        a = ev.run_benchmark(k=5)
        b = RecommendationEvaluator(self.rec).run_benchmark(k=5)
        self.assertEqual(len(a), 4)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(a["Mean Diversity"].between(0, 1).all())

    def test_hybrid_returns_k_unique_items(self):
        res, _ = self.rec.recommend(city="Mumbai", cuisines=["Pizza"], top_k=10)
        self.assertEqual(len(res), 10)
        self.assertEqual(res["Restaurant Name"].nunique(), 10)

    def test_real_data_veg_filter_has_no_non_veg_names(self):
        res, _ = self.rec.recommend(dietary_preference="Veg", top_k=30)
        self.assertFalse(res["Restaurant Name"].str.contains(r"non[\s\-]*veg", case=False).any())


if __name__ == "__main__":
    unittest.main()
