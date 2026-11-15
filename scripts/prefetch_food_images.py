"""Fill data/food_images.json with dish photos from Wikipedia (run once, needs internet).

    python scripts/prefetch_food_images.py

Commit the generated data/food_images.json so deployed copies of the app do not need to look
the photos up again.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.food_images import WikiImageResolver, all_known_titles  # noqa: E402


def main():
    resolver = WikiImageResolver(timeout=10)
    titles = all_known_titles()
    print(f"Looking up {len(titles)} dish articles...")
    resolver.prefetch(titles, workers=6)
    found = [t for t in titles if resolver.cache.get(t)]
    missing = [t for t in titles if not resolver.cache.get(t)]
    print(f"Photos found: {len(found)}/{len(titles)}")
    if missing:
        print("No photo (the app will try the next candidate or use emoji art):")
        for title in missing:
            print("  -", title)
    print(f"Saved to {resolver.cache_path}")


if __name__ == "__main__":
    main()
