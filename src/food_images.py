"""Dish photos for restaurant cards.

The catalogue has no photos, and a photo of a *different* restaurant must never be shown as if it
were the restaurant's own. So FoodWise shows an illustrative photo of the *dish* instead:

1. The restaurant name is checked first ("Hyderabad Biryani House" -> biryani).
2. Otherwise the cuisine tags are used (specific tags before generic ones like "Fast Food").
3. Each dish has a few candidate Wikipedia articles. The lead image of the first one that has a
   photo is used (Wikipedia / Wikimedia Commons, open licences).
4. If no photo can be found (offline, no image), the card falls back to emoji art.

Lookups use the public Wikipedia REST summary API and are cached in memory and in
``data/food_images.json`` (run ``python scripts/prefetch_food_images.py`` once to fill the cache).
"""
from __future__ import annotations

import json
import re
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from src.config import BASE_DIR

CACHE_PATH = BASE_DIR / "data" / "food_images.json"
API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
USER_AGENT = "FoodWiseApp/2.1 (student portfolio project; https://github.com/adarsh8158)"

# Restaurant-name keywords -> candidate Wikipedia article titles. First match wins.
NAME_RULES = [
    (r"biryani|biriyani|briyani", ["Biryani", "Hyderabadi biryani"]),
    (r"pizza|pizzeria|domino|la pino", ["Pizza", "Neapolitan pizza"]),
    (r"burger|mcdonald", ["Hamburger", "Cheeseburger"]),
    (r"\bmomos?\b", ["Momo (food)"]),
    (r"vada\s*pav", ["Vada pav"]),
    (r"pav\s*bhaji", ["Pav bhaji"]),
    (r"dosa|dosai|dhosa", ["Dosa", "Masala dosa"]),
    (r"\bidli|idly", ["Idli"]),
    (r"chaat|chat\s*bhandar|golgappa|pani\s*puri|puchka", ["Chaat", "Pani puri"]),
    (r"samosa|kachori", ["Samosa", "Kachori"]),
    (r"parath[ae]|parantha", ["Paratha"]),
    (r"chole|bhature|kulche", ["Chole bhature"]),
    (r"thali", ["Thali"]),
    (r"kebab|kabab|tikka|tandoor", ["Kebab", "Tandoori chicken", "Chicken tikka"]),
    (r"shawarma|shwarma", ["Shawarma"]),
    (r"\brolls?\b|kathi", ["Kati roll"]),
    (r"ice\s*cream|icecream|kulfi|gelato|baskin", ["Ice cream", "Kulfi"]),
    (r"cake|pastry|patisserie|bakery|bakers", ["Cake", "Pastry", "Croissant"]),
    (r"sweet|mithai|misthan|mishthan|halwai|halwa", ["Mithai", "Gulab jamun", "Jalebi"]),
    (r"rasgulla|rasmalai|mishti|sandesh", ["Rasgulla", "Mishti doi"]),
    (r"juice|shake|smoothie|lassi", ["Juice", "Milkshake", "Lassi"]),
    (r"\b(?:chai|tea)\b", ["Masala chai"]),
    (r"coffee|starbucks|barista", ["Coffee", "Cappuccino"]),
    (r"\bpaan\b", ["Paan"]),
    (r"sandwich|subway", ["Sandwich", "Submarine sandwich"]),
    (r"waffle", ["Waffle"]),
    (r"pancake|crepe", ["Pancake"]),
    (r"sushi", ["Sushi"]),
    (r"salad", ["Salad"]),
    (r"paneer", ["Paneer tikka"]),
    (r"litti|chokha", ["Litti chokha"]),
    (r"dhokla|khandvi|farsan", ["Dhokla"]),
    (r"noodle|chowmein|chow\s*mein|wok", ["Chow mein", "Hakka noodles"]),
    (r"pasta", ["Pasta", "Spaghetti"]),
    (r"fish|seafood|prawn|crab", ["Fish curry", "Seafood"]),
    (r"kfc|fried\s*chicken|chicken", ["Fried chicken", "Butter chicken"]),
    (r"dhaba", ["Dal makhani", "Butter chicken"]),
]

# Cuisine tag (lower case) -> candidate Wikipedia article titles.
CUISINE_MAP = {
    "north indian": ["Butter chicken", "Dal makhani", "Paneer tikka", "Naan"],
    "fast food": ["French fries", "Hamburger"],
    "chinese": ["Chow mein", "Fried rice", "Dim sum"],
    "beverages": ["Lemonade", "Juice"],
    "desserts": ["Gulab jamun", "Cake", "Dessert"],
    "street food": ["Pani puri", "Chaat", "Vada pav"],
    "biryani": ["Biryani", "Hyderabadi biryani"],
    "pizza": ["Pizza", "Neapolitan pizza"],
    "south indian": ["Dosa", "Masala dosa", "Idli", "Medu vada"],
    "sandwich": ["Sandwich", "Grilled cheese"],
    "shake": ["Milkshake"],
    "burger": ["Hamburger", "Cheeseburger"],
    "bakery": ["Croissant", "Bread", "Cake"],
    "mughlai": ["Korma", "Mughlai paratha"],
    "rolls": ["Kati roll", "Spring roll"],
    "momos": ["Momo (food)"],
    "ice cream": ["Ice cream", "Kulfi"],
    "sichuan": ["Kung Pao chicken", "Mapo doufu"],
    "kebab": ["Kebab", "Seekh kebab", "Shish kebab"],
    "pasta": ["Pasta", "Spaghetti"],
    "seafood": ["Fish curry", "Seafood"],
    "mithai": ["Mithai", "Gulab jamun", "Jalebi", "Rasgulla"],
    "continental": ["Steak", "Pasta"],
    "italian": ["Lasagne", "Pasta", "Pizza"],
    "coffee": ["Coffee", "Cappuccino"],
    "wraps": ["Burrito", "Shawarma"],
    "salad": ["Salad"],
    "cafe": ["Cappuccino", "Croissant"],
    "juices": ["Juice", "Smoothie"],
    "tea": ["Masala chai", "Tea"],
    "healthy food": ["Salad", "Smoothie"],
    "maharashtrian": ["Vada pav", "Misal pav", "Pav bhaji"],
    "asian": ["Fried rice", "Ramen"],
    "waffle": ["Waffle"],
    "bbq": ["Barbecue", "Spare ribs"],
    "shawarma": ["Shawarma"],
    "mexican": ["Taco", "Burrito", "Nachos"],
    "gujarati": ["Dhokla", "Thali"],
    "rajasthani": ["Dal baati churma", "Thali"],
    "arabian": ["Hummus", "Shawarma"],
    "bengali": ["Rasgulla", "Fish curry"],
    "thai": ["Pad thai", "Tom yum"],
    "lebanese": ["Hummus", "Falafel"],
    "andhra": ["Hyderabadi biryani", "Pesarattu"],
    "kerala": ["Appam", "Puttu"],
    "oriental": ["Dim sum", "Fried rice"],
    "hyderabadi": ["Hyderabadi biryani", "Haleem"],
    "american": ["Hamburger", "Hot dog", "French fries"],
    "chettinad": ["Chicken 65", "Dosa"],
    "pancake": ["Pancake"],
    "goan": ["Vindaloo", "Fish curry"],
    "japanese": ["Sushi", "Ramen"],
    "bar food": ["Chicken wings", "Nachos"],
    "sushi": ["Sushi"],
    "bihari": ["Litti chokha", "Thekua"],
    "mandi": ["Mandi (food)", "Kabsa"],
    "finger food": ["French fries", "Samosa"],
    "lucknowi": ["Galouti kebab", "Biryani"],
    "awadhi": ["Galouti kebab", "Biryani"],
    "paan": ["Paan"],
    "roast chicken": ["Roast chicken"],
    "mishti": ["Rasgulla", "Mishti doi"],
    "assamese": ["Thali", "Fish curry"],
    "modern indian": ["Thali"],
    "tibetan": ["Momo (food)", "Thukpa"],
    "mangalorean": ["Neer dosa", "Fish curry"],
    "tamil": ["Dosa", "Idli", "Pongal (dish)"],
    "korean": ["Bibimbap", "Kimchi"],
    "hot dogs": ["Hot dog"],
    "odia": ["Pakhala", "Thali"],
    "cake": ["Cake", "Cheesecake"],
    "malwani": ["Fish curry"],
    "european": ["Steak", "Pasta"],
    "afghan": ["Kabuli pulao", "Kebab"],
    "afghani": ["Kabuli pulao", "Kebab"],
    "north eastern": ["Momo (food)", "Thukpa"],
    "mediterranean": ["Hummus", "Greek salad"],
    "bubble tea": ["Bubble tea"],
    "middle eastern": ["Falafel", "Hummus"],
    "panini": ["Panini (sandwich)", "Sandwich"],
    "burmese": ["Mohinga", "Fried rice"],
    "kathiyawadi": ["Thali", "Dhokla"],
    "charcoal chicken": ["Tandoori chicken", "Roast chicken"],
    "grilled chicken": ["Tandoori chicken", "Roast chicken"],
    "steak": ["Steak"],
    "turkish": ["Doner kebab", "Baklava"],
    "sindhi": ["Dal pakwan"],
    "french": ["Croissant", "Crêpe"],
    "frozen yogurt": ["Frozen yogurt"],
    "cantonese": ["Dim sum"],
    "kashmiri": ["Rogan josh"],
    "tex-mex": ["Nachos", "Burrito"],
    "malaysian": ["Nasi lemak"],
    "mongolian": ["Mongolian beef", "Hot pot"],
    "iranian": ["Chelow kabab", "Kabab koobideh"],
    "parsi": ["Dhansak"],
    "portuguese": ["Pastel de nata"],
    "drinks only": ["Cocktail", "Juice"],
    "greek": ["Greek salad", "Gyros"],
    "spanish": ["Paella"],
    "german": ["Bratwurst"],
    "vietnamese": ["Phở"],
    "belgian": ["Belgian waffle"],
    "indian": ["Thali"],
}
# Tags that say little about the dish; specific tags are preferred over these.
GENERIC_TAGS = {"fast food", "beverages", "desserts", "street food", "healthy food", "finger food",
                "cuisine varies", "bar food", "drinks only", "continental", "asian", "oriental",
                "modern indian", "european", "north indian", "chinese"}

_NAME_RULES = [(re.compile(p, re.IGNORECASE), c) for p, c in NAME_RULES]


def _rotate(candidates, seed_text):
    """Start from a stable, name-based position so restaurants of one cuisine get varied photos."""
    start = zlib.crc32(seed_text.encode("utf-8")) % len(candidates)
    return list(candidates[start:]) + list(candidates[:start])


def candidate_titles(name, cuisine_list):
    """Return ordered Wikipedia article titles to try for a restaurant, or [] if nothing fits."""
    name = "" if name is None else str(name)
    for pattern, candidates in _NAME_RULES:
        if pattern.search(name):
            return _rotate(candidates, name)
    tags = [str(t).strip().lower() for t in (cuisine_list or [])]
    specific = [t for t in tags if t not in GENERIC_TAGS]
    for tag in specific + [t for t in tags if t in GENERIC_TAGS]:
        if tag in CUISINE_MAP:
            return _rotate(CUISINE_MAP[tag], name)
    return []


def all_known_titles():
    titles = {t for _, c in NAME_RULES for t in c}
    titles |= {t for c in CUISINE_MAP.values() for t in c}
    return sorted(titles)


class WikiImageResolver:
    """Looks up the lead photo of a Wikipedia article. Never raises; returns None on any failure."""

    def __init__(self, cache_path=CACHE_PATH, timeout=4.0, fetch=None, breaker_after=3, breaker_pause=300):
        self.cache_path = Path(cache_path) if cache_path else None
        self.timeout = timeout
        self._fetch = fetch or self._fetch_summary
        self._lock = threading.Lock()
        self._failures = 0
        self._breaker_after = breaker_after
        self._breaker_pause = breaker_pause
        self._paused_until = 0.0
        self.cache = {}
        if self.cache_path and self.cache_path.exists():
            try:
                loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
                self.cache = {k: v for k, v in loaded.items() if isinstance(v, str)}
            except (OSError, ValueError):
                self.cache = {}

    # -- network ---------------------------------------------------------------------------
    def _fetch_summary(self, title):
        """Return a thumbnail URL, "" if the article has no photo, raise OSError on network problems."""
        url = API.format(title=quote(title.replace(" ", "_"), safe="()_,'-"))
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                return ""
            raise OSError(f"HTTP {exc.code}") from exc
        if data.get("type") == "disambiguation":
            return ""
        return (data.get("thumbnail") or {}).get("source", "") or ""

    # -- lookup ----------------------------------------------------------------------------
    def lookup(self, title):
        """Photo URL for one article title, or None. Results (including "no photo") are cached."""
        with self._lock:
            if title in self.cache:
                return self.cache[title] or None
            if time.time() < self._paused_until:
                return None
        try:
            url = self._fetch(title)
        except Exception:  # network down, timeout, bad JSON: never break the app
            with self._lock:
                self._failures += 1
                if self._failures >= self._breaker_after:
                    self._paused_until = time.time() + self._breaker_pause
                    self._failures = 0
            return None
        with self._lock:
            self._failures = 0
            self.cache[title] = url or ""
        return url or None

    def resolve(self, titles):
        """First candidate title that has a photo, or None."""
        for title in titles:
            url = self.lookup(title)
            if url:
                return url
        return None

    def prefetch(self, titles, workers=8):
        """Look up many titles in parallel (cache hits are free)."""
        todo = [t for t in dict.fromkeys(titles) if t not in self.cache]
        if todo:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(self.lookup, todo))
            self.save()

    def save(self):
        if not self.cache_path or not self.cache:
            return  # nothing learned yet (for example offline): do not write an empty file
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = json.dumps(dict(sorted(self.cache.items())), indent=1, ensure_ascii=False)
            self.cache_path.write_text(payload, encoding="utf-8")
        except OSError:
            pass  # read-only deployment folder: the in-memory cache still works
