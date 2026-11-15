"""Central configuration for FoodWise 2.1."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "zomato_dataset.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_restaurants.csv"

# Ranking weights are normalized at runtime. Safety is deliberately low-weight:
# the source field is a dataset label, not an independently verified inspection.
WEIGHTS = {
    "content_similarity": 0.48,
    "rating": 0.25,
    "price_compatibility": 0.17,
    "delivery_speed": 0.07,
    "safety_label": 0.03,
}
DEFAULT_TOP_K = 12
CITY_ALIASES = {"Delhi NCR": "NCR", "Delhi": "NCR", "New Delhi": "NCR"}
EXCLUDED_ENTITIES = {"APOLLO PHARMACIES LIMITED", "ADLOGGS TECHNOLOGY PRIVATE LIMITED", "ZTest Aligarh"}
REQUIRED_COLUMNS = [
    "Restaurant Name", "Rating", "Cuisine", "Average Price",
    "Average Delivery Time", "Safety Measure", "Location"
]
