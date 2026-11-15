"""Validated, repeatable preprocessing for the supplied restaurant catalogue."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, CITY_ALIASES, EXCLUDED_ENTITIES, REQUIRED_COLUMNS


def load_raw_data(filepath=None) -> pd.DataFrame:
    path = Path(filepath) if filepath else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    return df


def extract_price(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d[\d,]*", str(value).replace("₹", ""))
    return float(match.group(0).replace(",", "")) if match else np.nan


def extract_rating(value):
    if pd.isna(value) or str(value).strip().lower() in {"", "-", "new", "opening soon", "nan"}:
        return np.nan
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return np.nan
    rating = float(match.group())
    return rating if 0 <= rating <= 5 else np.nan


def parse_delivery(value):
    text = "" if pd.isna(value) else str(value).strip()
    lower = text.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|mins|minutes)?", lower)
    minutes = float(match.group(1)) if match and any(x in lower for x in ["min", "minute"]) else np.nan
    if "permanently closed" in lower:
        status = "Permanently Closed"
    elif "temporarily closed" in lower:
        status = "Temporarily Closed"
    elif "not accepting" in lower:
        status = "Not Accepting Orders"
    elif pd.notna(minutes):
        status = "Delivery time listed"
    else:
        status = "Unknown"
    return minutes, status


VEG_PATTERN = re.compile(r"\b(?:pure\s*veg|veg(?:etarian)?|vaishno|jain|swaminarayan)\b", re.IGNORECASE)
# Names that mention veg but also serve non-veg ("Non Veg", "Non-Veg", "Nonveg", "No Veg", "Veg & Non Veg").
NON_VEG_PATTERN = re.compile(r"\b(?:non|no)[\s\-]*veg(?:etarian)?\b", re.IGNORECASE)


def is_pure_veg_name(name) -> bool:
    """Name-based hint only: True when the name says veg and does not mention non-veg."""
    text = "" if pd.isna(name) else str(name)
    return bool(VEG_PATTERN.search(text)) and not NON_VEG_PATTERN.search(text)


def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.drop_duplicates().copy()
    df["Restaurant Name"] = df["Restaurant Name"].fillna("").astype(str).str.strip()
    df = df[df["Restaurant Name"].ne("")]
    df = df[~df["Restaurant Name"].str.upper().isin(EXCLUDED_ENTITIES)]
    df = df[~df["Average Delivery Time"].fillna("").astype(str).str.contains("permanently closed", case=False, regex=False)]
    df["Cuisine"] = df["Cuisine"].fillna("").astype(str).str.strip()
    df = df[df["Cuisine"].ne("")]
    df["city"] = df["Location"].fillna("Unknown").astype(str).str.strip().replace(CITY_ALIASES)
    df["rating_numeric"] = df["Rating"].apply(extract_rating)
    df["rating_status"] = np.where(df["rating_numeric"].notna(), "Rated", "Unrated / New")
    df["price_numeric"] = df["Average Price"].apply(extract_price)
    df["price_tier"] = pd.cut(df["price_numeric"], [-np.inf, 150, 300, np.inf], labels=["Budget", "Mid-range", "Premium"]).astype("object").fillna("Unknown")
    delivery = df["Average Delivery Time"].apply(parse_delivery)
    df["delivery_time_mins"] = delivery.map(lambda x: x[0])
    df["delivery_status"] = delivery.map(lambda x: x[1])
    df["safety_label"] = df["Safety Measure"].fillna("Not specified").astype(str).str.strip()
    df["cuisine_list"] = df["Cuisine"].apply(lambda x: [v.strip() for v in x.split(",") if v.strip()])
    df["cuisine_clean"] = df["cuisine_list"].apply(lambda x: ", ".join(x))
    # This is only a name-based heuristic, not a verified dietary guarantee.
    df["is_pure_veg"] = df["Restaurant Name"].apply(is_pure_veg_name)
    df["content_soup"] = df.apply(
        lambda r: (r["cuisine_clean"] + " ") * 2 + f" {r['city']} {r['price_tier']} " + ("pure vegetarian veg " if r["is_pure_veg"] else ""),
        axis=1,
    ).str.lower().str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return df.reset_index(drop=True)


def run_pipeline(input_path=None, output_path=None):
    cleaned = clean_dataset(load_raw_data(input_path))
    out = Path(output_path) if output_path else PROCESSED_DATA_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    save = cleaned.copy()
    save["cuisine_list"] = save["cuisine_list"].apply(lambda values: "|".join(values))
    save.to_csv(out, index=False)
    print(f"Saved {len(cleaned):,} cleaned restaurants to {out}")
    return cleaned

if __name__ == "__main__":
    run_pipeline()
