"""End-to-end smoke checks for the FoodWise 2.1 catalogue and recommender."""
from pathlib import Path
from src.recommender import RestaurantRecommender
from src.evaluation import RecommendationEvaluator


def run_all_tests():
    root = Path(__file__).resolve().parent
    rec = RestaurantRecommender(data_path=root / "data" / "raw" / "zomato_dataset.csv")
    print(f"Loaded {len(rec.df):,} listings across {len(rec.cities)} cities and {len(rec.cuisines)} cuisine tags.")
    checks = []
    # Use cities and cuisine tags that are present in the actual dataset.
    city = "Mumbai" if "Mumbai" in rec.cities else rec.cities[0]
    city_rows = rec.df[rec.df["city"].eq(city)]
    cuisine = city_rows.iloc[0]["cuisine_list"][0] if not city_rows.empty else rec.cuisines[0]
    results, meta = rec.recommend(city=city, cuisines=[cuisine], top_k=5, min_rating=0)
    assert not results.empty and meta["status"] == "ok"
    checks.append("recommendations return candidates")
    assert (results["city"] == city).all()
    checks.append("city constraint is respected")
    source_name = results.iloc[0]["Restaurant Name"]
    similar = rec.get_similar_restaurants(source_name, city=city, top_k=3)
    assert source_name not in similar["Restaurant Name"].tolist()
    checks.append("similarity excludes the source listing")
    empty, empty_meta = rec.recommend(city=city, budget_max=1, min_rating=5)
    assert empty.empty and empty_meta["status"] == "no_match"
    checks.append("empty filters are handled")
    benchmark = RecommendationEvaluator(rec).run_benchmark(k=5)
    assert len(benchmark) == 4
    checks.append("offline benchmark executes")
    veg, _ = rec.recommend(dietary_preference="Veg", top_k=30)
    assert not veg["Restaurant Name"].str.contains(r"non[\s\-]*veg", case=False).any()
    checks.append("veg filter excludes non-veg names")
    chain, _ = rec.recommend(city=city, top_k=20)
    assert chain["Restaurant Name"].is_unique
    checks.append("chain duplicates are removed")
    print("\nFoodWise smoke checks:")
    for item in checks:
        print(f"  PASS  {item}")
    print("\nBenchmark summary:")
    print(benchmark.to_string(index=False))
    print(f"\n{len(checks)}/{len(checks)} smoke checks passed.")

if __name__ == "__main__":
    run_all_tests()
