"""FoodWise 2.1 — Royal Indian Spice & Feast Edition."""
from pathlib import Path
import sys
import html
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urllib.parse import quote
from src.ml_model import RatingPredictor
from src.food_images import WikiImageResolver, candidate_titles
from src.recommender import RestaurantRecommender
from src.evaluation import RecommendationEvaluator
from src.utils import format_price

st.set_page_config(page_title="FoodWise | Discover Royal Flavors", page_icon="🍛", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@700;800&family=Manrope:wght@600;700;800&display=swap');

:root {
    --fw-maroon: #3E0B10;
    --fw-maroon-light: #5C131B;
    --fw-gold: #F59E0B;
    --fw-gold-light: #FDE68A;
    --fw-crimson: #B71C1C;
    --fw-ivory: #FFFDF7;
    --fw-ink: #2D1214;
    --fw-muted: #7C5E60;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--fw-ink); }
.stApp { background: var(--fw-ivory); }

/* Sidebar Styling - Deep Royal Maroon & Gold */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #3E0B10 0%, #260508 100%);
    border-right: 2px solid #D97706;
}
section[data-testid="stSidebar"] * { color: #FFF8E7 !important; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px 12px;
    border-radius: 10px;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(245, 158, 11, 0.15);
}

/* Visible Catalogue Cards in Sidebar with Gold Borders */
section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: rgba(255, 248, 231, 0.08) !important;
    border: 1.5px solid #F59E0B !important;
    border-radius: 14px !important;
    padding: 12px 16px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.28) !important;
}
section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] * {
    color: #FDE68A !important;
    font-weight: 700 !important;
    font-size: 0.80rem !important;
    text-transform: uppercase;
    letter-spacing: 0.9px;
}
section[data-testid="stSidebar"] div[data-testid="stMetricValue"] * {
    color: #FFFFFF !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.65rem !important;
}

/* Royal Spice Hero Banner */
.hero {
    background: linear-gradient(120deg, #4A0E14 0%, #8E1B24 55%, #B71C1C 100%);
    border: 2px solid #F59E0B;
    padding: 36px 40px;
    border-radius: 22px;
    color: white;
    margin: 4px 0 24px;
    box-shadow: 0 10px 25px rgba(74, 14, 20, 0.18);
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.35rem;
    line-height: 1.15;
    margin: 0 0 10px;
    color: #FFFDF7;
}
.hero p {
    color: #FDE68A;
    margin: 0;
    font-size: 1.02rem;
    max-width: 680px;
}
.eyebrow {
    color: #FBBF24;
    text-transform: uppercase;
    letter-spacing: 2.2px;
    font-weight: 800;
    font-size: 0.74rem;
    margin-bottom: 10px;
}

/* Restaurant Cards & Badges */
.restaurant-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.16rem;
    font-weight: 800;
    color: #3E0B10;
}
.match {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    color: #92400E;
    border: 1px solid #F59E0B;
    border-radius: 30px;
    padding: 5px 12px;
    font-size: 0.8rem;
    font-weight: 800;
    white-space: nowrap;
}
.diet-badge-veg {
    display: inline-block;
    background: #ECFDF5;
    color: #065F46;
    border: 1px solid #10B981;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 800;
    margin-left: 8px;
    vertical-align: middle;
}
.diet-badge-nonveg {
    display: inline-block;
    background: #FEF2F2;
    color: #991B1B;
    border: 1px solid #EF4444;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 800;
    margin-left: 8px;
    vertical-align: middle;
}
.muted { color: #7C5E60; font-size: 0.88rem; }
.reason {
    background: #FFFBEB;
    border-left: 4px solid #D97706;
    border-radius: 0 10px 10px 0;
    padding: 9px 14px;
    color: #5C3A21;
    font-size: 0.85rem;
    margin-top: 10px;
}

/* Containers & Metrics */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1.5px solid #FDE68A;
    padding: 14px 16px;
    border-radius: 16px;
    box-shadow: 0 2px 10px rgba(183, 28, 28, 0.04);
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1.5px solid #F3E5AB !important;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(62, 11, 16, 0.04);
}

/* Primary Buttons */
.stButton button[kind="primary"] {
    background: linear-gradient(90deg, #B71C1C 0%, #D97706 100%);
    border: none;
    color: white !important;
    font-weight: 800;
    border-radius: 10px;
    padding: 0.55rem 1.2rem;
    box-shadow: 0 4px 12px rgba(183, 28, 28, 0.25);
}
.stButton button[kind="primary"]:hover {
    background: linear-gradient(90deg, #8E1B24 0%, #B45309 100%);
}
.small-note { font-size: 0.78rem; color: #7C5E60; }

/* ---------- Dashboard layout (mockup style) ---------- */
.block-container { padding-top: 1.6rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }

/* Sidebar logo + stat cards */
.fw-logo { font-family: 'Playfair Display', serif; font-size: 1.9rem; font-weight: 800; color: #F59E0B !important; margin: 4px 0 2px; }
.fw-logo span { color: #F59E0B !important; }
.fw-side-title { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.15rem; margin: 18px 0 10px; color: #FFF8E7 !important; }
.stat-card {
    border: 1.5px solid #F59E0B; border-radius: 16px; padding: 12px 10px; margin-bottom: 12px;
    text-align: center; background: rgba(255, 248, 231, 0.05); box-shadow: 0 4px 14px rgba(0,0,0,.28);
}
.stat-label { color: #FFF8E7 !important; font-size: 0.88rem; font-weight: 500; }
.stat-value { color: #FBBF24 !important; font-family: 'Manrope', sans-serif; font-size: 1.7rem; font-weight: 800; line-height: 1.25; }

/* Top bar */
.topbar-title { font-family: 'Manrope', sans-serif; font-size: 1.25rem; font-weight: 800; color: #1F1F1F; padding-top: 6px; }
.avatar { width: 42px; height: 42px; border-radius: 50%; background: #3E0B10; color: #FDE68A; display: flex; align-items: center; justify-content: center; font-size: 1.25rem; margin-left: auto; }
.st-key-topsearch input { border-radius: 10px !important; border: 1.5px solid #3E0B10 !important; background: #fff !important; }
.st-key-topsearch div[data-baseweb="input"] { border-radius: 10px; border: 1.5px solid #3E0B10; background: #fff; }

/* Banner carousel (pure CSS) */
.fw-banner {
    position: relative; height: 190px; border-radius: 12px; overflow: hidden; margin: 6px 0 0;
    background: linear-gradient(100deg, #3E0B10 0%, #5C131B 50%, #3E0B10 100%); border: 2px solid #F59E0B;
}
.fw-banner:after {
    content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 10px;
    background: repeating-linear-gradient(90deg, #B71C1C 0 10px, #F59E0B 10px 14px, #B71C1C 14px 24px);
}
.fw-slide { position: absolute; inset: 0; display: flex; align-items: center; justify-content: space-between; padding: 0 34px; opacity: 0; animation: fwslide 16s infinite; }
.fw-slide .plate { width: 150px; height: 150px; border-radius: 50%; background: radial-gradient(circle, #FFF8E7 0 55%, #F5D58A 56% 70%, #D9A441 71%); display: flex; align-items: center; justify-content: center; font-size: 4.6rem; box-shadow: 0 6px 18px rgba(0,0,0,.45); }
.fw-slide .txt { text-align: center; color: #fff; }
.fw-slide .txt .k { color: #FDE68A; font-size: 1.25rem; letter-spacing: .5px; }
.fw-slide .txt .t { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.7rem; line-height: 1.25; margin-top: 6px; }
@keyframes fwslide { 0% {opacity: 0} 4% {opacity: 1} 25% {opacity: 1} 29% {opacity: 0} 100% {opacity: 0} }
@keyframes fwdot { 0% {background: #3E0B10; width: 26px} 25% {background: #3E0B10; width: 26px} 29% {background: #C9B8A8; width: 8px} 100% {background: #C9B8A8; width: 8px} }
.fw-dots { display: flex; justify-content: center; gap: 6px; margin: 12px 0 14px; }
.fw-dots i { display: block; height: 7px; border-radius: 6px; background: #C9B8A8; width: 8px; animation: fwdot 16s infinite; }

/* Yellow filter bar */
.st-key-filterbar { background: #FBBF24; border-radius: 12px; padding: 14px 16px 6px; margin-bottom: 8px; }
.st-key-filterbar div[data-baseweb="select"] > div { background: #fff; border-radius: 8px; border: 1.5px solid #3E0B10; min-height: 40px; }
.st-key-filterbar label p { font-weight: 700; color: #3E0B10; }
.st-key-filterbar div[data-testid="stCheckbox"] label[data-selected="false"] > div:first-of-type { background-color: #9CA3AF !important; }
.st-key-tg_veg label[data-selected="true"] > div:first-of-type { background-color: #15803D !important; }
.st-key-tg_nonveg label[data-selected="true"] > div:first-of-type { background-color: #B71C1C !important; }
.st-key-filterbar div[data-testid="stCheckbox"] label p { font-weight: 700; color: #3E0B10; }

.section-title { font-family: 'Manrope', sans-serif; font-size: 1.1rem; font-weight: 800; color: #1F1F1F; margin: 14px 0 10px; }

/* Restaurant cards */
.fw-art { position: relative; height: 112px; border-radius: 10px; display: flex; align-items: center; justify-content: center; gap: 8px; font-size: 2.6rem; overflow: hidden; margin-bottom: 10px; }
.fw-art .badge-match { position: absolute; top: 0; right: 0; background: #FBBF24; color: #3E0B10; font-weight: 800; font-size: .78rem; padding: 4px 10px; border-bottom-left-radius: 10px; }
.fw-art .vegdot { position: absolute; top: 8px; left: 8px; width: 16px; height: 16px; border-radius: 3px; background: #fff; border: 2px solid #15803D; display: flex; align-items: center; justify-content: center; }
.fw-art .vegdot:after { content: ""; width: 7px; height: 7px; border-radius: 50%; background: #15803D; }
.fw-art.photo { box-shadow: inset 0 -28px 24px -18px rgba(0,0,0,.35); }
.fw-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.fw-name { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.05rem; color: #2D1214; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fw-rate { background: #15803D; color: #fff; font-weight: 800; font-size: .78rem; border-radius: 6px; padding: 3px 8px; white-space: nowrap; }
.fw-rate.none { background: #9CA3AF; }
.fw-meta { display: flex; flex-wrap: nowrap; gap: 4px 12px; white-space: nowrap; overflow: hidden; color: #5B4B4C; font-size: .82rem; margin: 6px 0; }
.fw-desc { color: #3b3b3b; font-size: .83rem; line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; height: 2.7em; }
.fw-strip { height: 6px; border-radius: 3px; margin-top: 10px; background: repeating-linear-gradient(90deg, #B71C1C 0 8px, #F59E0B 8px 11px, #B71C1C 11px 19px); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Preparing the Royal Spice recommendation engine…")
def load_engine():
    return RestaurantRecommender()

try:
    rec = load_engine()
except Exception as exc:
    st.error(f"FoodWise could not load its catalogue: {exc}")
    st.info("Check that data/raw/zomato_dataset.csv exists and matches the documented seven-column schema.")
    st.stop()

if "favorites" not in st.session_state:
    st.session_state.favorites = []

with st.sidebar:
    st.markdown('<div class="fw-logo">🍴 FoodWise</div>', unsafe_allow_html=True)
    page = st.radio(
        "Workspace",
        ["Discover", "Similar restaurants", "Explore catalogue", "Compare", "Saved shortlist", "Insights & model lab", "About FoodWise"],
        label_visibility="collapsed"
    )
    st.markdown('<div class="fw-side-title">Catalog Stats</div>', unsafe_allow_html=True)
    for _label, _value in [
        ("Active Cities", f"{len(rec.cities):,}"),
        ("Partner Restaurants", f"{len(rec.df):,}"),
        ("Cuisine Varieties", f"{len(rec.cuisines):,}"),
        ("Rated Listings", f"{rec.df['rating_numeric'].notna().mean():.0%}"),
    ]:
        st.markdown(f'<div class="stat-card"><div class="stat-label">{_label}</div><div class="stat-value">{_value}</div></div>', unsafe_allow_html=True)
    st.caption("Listing data is from the supplied catalogue, not live availability.")


def hero(kicker, title, subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">✨ {kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


ART = [  # (keyword, emoji, gradient)
    ("biryani", "🍛", ("#F59E0B", "#B45309")), ("pizza", "🍕", ("#EF4444", "#991B1B")),
    ("burger", "🍔", ("#F97316", "#9A3412")), ("chinese", "🥡", ("#DC2626", "#7F1D1D")),
    ("momo", "🥟", ("#F59E0B", "#92400E")), ("roll", "🌯", ("#EA580C", "#7C2D12")),
    ("south indian", "🥞", ("#FBBF24", "#B45309")), ("north indian", "🍛", ("#D97706", "#7C2D12")),
    ("mughlai", "🍢", ("#B91C1C", "#7F1D1D")), ("kebab", "🍢", ("#B91C1C", "#7F1D1D")),
    ("street food", "🥙", ("#F59E0B", "#B91C1C")), ("sandwich", "🥪", ("#CA8A04", "#78350F")),
    ("pasta", "🍝", ("#EF4444", "#7C2D12")), ("italian", "🍝", ("#EF4444", "#7C2D12")),
    ("ice cream", "🍨", ("#F472B6", "#9D174D")), ("dessert", "🍰", ("#F472B6", "#9D174D")),
    ("bakery", "🥐", ("#D97706", "#78350F")), ("cafe", "☕", ("#A16207", "#451A03")),
    ("mithai", "🍬", ("#F59E0B", "#9A3412")), ("beverage", "🥤", ("#10B981", "#065F46")),
    ("juice", "🥤", ("#10B981", "#065F46")), ("tea", "🍵", ("#65A30D", "#365314")),
    ("seafood", "🍤", ("#0EA5E9", "#0C4A6E")), ("thali", "🍱", ("#F59E0B", "#7C2D12")),
]

def card_art(cuisine_text):
    text = str(cuisine_text).lower()
    emojis, grad = [], ("#B45309", "#3E0B10")
    for key, emoji, g in ART:
        if key in text and emoji not in emojis:
            if not emojis:
                grad = g
            emojis.append(emoji)
        if len(emojis) == 3:
            break
    return (emojis or ["🍽️"]), grad


@st.cache_resource
def get_image_resolver():
    return WikiImageResolver()


def cuisine_tags(row):
    tags = row.get("cuisine_list")
    if isinstance(tags, (list, tuple)):
        return list(tags)
    return [t.strip() for t in str(row.get("Cuisine", "")).split(",") if t.strip()]


def restaurant_card(row, key_prefix="card", img_url=None):
    name = html.escape(str(row.get("Restaurant Name", "Restaurant")))
    city = html.escape(str(row.get("city", "")))
    cuisines_raw = str(row.get("Cuisine", "Cuisine not listed"))
    emojis, (g1, g2) = card_art(cuisines_raw)
    rating_val = row.get("rating_numeric")
    est_val = row.get("predicted_rating")
    if pd.notna(rating_val):
        rate_html = f'<span class="fw-rate">★ {float(rating_val):.1f} / 5</span>'
    elif pd.notna(est_val):
        rate_html = (f'<span class="fw-rate none" title="Unrated. Estimated by the rating model from cuisine, city and price; not a real rating.">'
                     f'Unrated · est. {float(est_val):.1f}</span>')
    else:
        rate_html = '<span class="fw-rate none">Unrated</span>'
    price_val, mins = row.get("price_numeric"), row.get("delivery_time_mins")
    price = f"₹{float(price_val):,.0f}" if pd.notna(price_val) else "Price n/a"
    delivery = f"{float(mins):.0f} min" if pd.notna(mins) else "Time n/a"
    score = row.get("match_score", row.get("similarity_score", None))
    badge = ""
    if score is not None and pd.notna(score):
        badge = f'<div class="badge-match">{float(score):.0f}% {"Match" if "match_score" in row else "Similar"}</div>'
    vegdot = '<div class="vegdot"></div>' if bool(row.get("is_pure_veg", False)) else ""
    reason = html.escape(str(row.get("explanation", "") or ""))
    if img_url:
        safe_url = quote(img_url, safe=":/%?=&_-.,()~")
        art = (f'<div class="fw-art photo" style="background-image:url(\'{safe_url}\'),linear-gradient(135deg,{g1},{g2});'
               f'background-size:cover;background-position:center">{vegdot}{badge}</div>')
    else:
        art = f'<div class="fw-art" style="background:linear-gradient(135deg,{g1},{g2})">{vegdot}{badge}{"".join(emojis)}</div>'
    html_card = (
        art +
        f'<div class="fw-head"><div class="fw-name" title="{name}">{name}</div>{rate_html}</div>'
        f'<div class="fw-meta"><span style="overflow:hidden;text-overflow:ellipsis">📍 {city}</span><span>⏱ {delivery}</span><span>💰 {price}</span></div>'
        f'<div class="fw-desc" title="{reason}">{html.escape(cuisines_raw)}</div>'
        f'<div class="fw-strip"></div>'
    )
    with st.container(border=True):
        st.markdown(html_card, unsafe_allow_html=True)
        fav_id = f"{row.get('Restaurant Name')}|{row.get('city')}"
        saved = fav_id in st.session_state.favorites
        if st.button("♥ Saved" if saved else "♡ Save to shortlist", key=f"{key_prefix}_{fav_id}", use_container_width=True):
            if saved:
                st.session_state.favorites.remove(fav_id)
            else:
                st.session_state.favorites.append(fav_id)
            st.rerun()


def render_grid(frame, key_prefix, cols=3):
    rows = [r for _, r in frame.iterrows()]
    urls = [None] * len(rows)
    if st.session_state.get("show_photos", True) and rows:
        resolver = get_image_resolver()
        plans = [candidate_titles(r.get("Restaurant Name"), cuisine_tags(r)) for r in rows]
        resolver.prefetch([p[0] for p in plans if p])  # look up the first-choice photos in parallel
        urls = [resolver.resolve(p) if p else None for p in plans]
    for start in range(0, len(rows), cols):
        columns = st.columns(cols)
        for offset, col in enumerate(columns):
            if start + offset < len(rows):
                with col:
                    restaurant_card(rows[start + offset], f"{key_prefix}_{start + offset}", urls[start + offset])
    if any(urls):
        st.caption("Dish photos are illustrative (Wikipedia / Wikimedia Commons, open licences), chosen from the restaurant name and cuisine. They are not photos of the restaurant itself.")


def to_csv_download(frame, filename, label="Download CSV"):
    if frame is not None and not frame.empty:
        safe = frame.drop(columns=["cuisine_list", "content_soup"], errors="ignore").copy()
        st.download_button(label, safe.to_csv(index=False).encode("utf-8-sig"), filename, "text/csv")


BUDGETS = {"Any budget": None, "Up to ₹100": 100, "Up to ₹150": 150, "Up to ₹300": 300, "Up to ₹500": 500, "Up to ₹800": 800}
SLIDES = [
    ("🍛", "🍚", "Discover Royal Flavors", "Street Food & Royal Spice<br>Market Secrets Await"),
    ("🥘", "🍢", "Biryani, Kebabs & Curries", "Slow-Cooked Classics<br>From Every Kitchen"),
    ("🍨", "🥐", "Sweet Endings", "Mithai, Bakery & Dessert<br>Favourites Near You"),
    ("🍕", "🥡", "Something for Every Mood", f"{len(rec.df):,} Restaurants<br>Across {len(rec.cities)} Cities"),
]


@st.cache_data(show_spinner=False)
def run_recommend(_rec, **kwargs):
    return _rec.recommend(**kwargs)


if page == "Discover":
    t1, t2, t3 = st.columns([4, 3, 0.6], vertical_alignment="center")
    t1.markdown('<div class="topbar-title">Dashboard</div>', unsafe_allow_html=True)
    with t2:
        with st.container(key="topsearch"):
            search_q = st.text_input("Search restaurants", placeholder="🔍  Search restaurants by name", label_visibility="collapsed")
    t3.markdown('<div class="avatar">👤</div>', unsafe_allow_html=True)

    slides_html = "".join(
        f'<div class="fw-slide" style="animation-delay:{i * 4}s"><div class="plate">{l}</div>'
        f'<div class="txt"><div class="k">{k}</div><div class="t">{t}</div></div><div class="plate">{r}</div></div>'
        for i, (l, r, k, t) in enumerate(SLIDES)
    )
    dots_html = "".join(f'<i style="animation-delay:{i * 4}s"></i>' for i in range(len(SLIDES)))
    st.markdown(f'<div class="fw-banner">{slides_html}</div><div class="fw-dots">{dots_html}</div>', unsafe_allow_html=True)

    with st.container(key="filterbar"):
        f1, f2, f3, f4, f5 = st.columns([1.3, 1.3, 1.1, 0.75, 0.9], vertical_alignment="center")
        city = f1.selectbox("City", rec.cities, index=None, placeholder="City", label_visibility="collapsed")
        cuisine = f2.selectbox("Cuisine", rec.cuisines, index=None, placeholder="Cuisine", label_visibility="collapsed")
        budget_label = f3.selectbox("Budget", list(BUDGETS), index=None, placeholder="Budget", label_visibility="collapsed")
        veg_on = f4.toggle("Veg", value=True, key="tg_veg")
        nonveg_on = f5.toggle("Non-Veg", value=True, key="tg_nonveg")
    budget = BUDGETS.get(budget_label)
    dietary_pref = "Veg" if (veg_on and not nonveg_on) else "Non-Veg" if (nonveg_on and not veg_on) else "Any"

    with st.expander("More filters"):
        m1, m2, m3 = st.columns(3)
        min_rating = m1.slider("Minimum rating", 0.0, 5.0, 3.5, 0.5, help="Unrated listings can be included separately.")
        top_k = m2.slider("Results count", 6, 30, 12, 3)
        diversity = m3.slider("Menu variety", 0.0, 0.4, 0.12, 0.04, help="Higher values reduce near-duplicate cuisine profiles.")
        m4, m5, m6 = st.columns(3)
        include_unrated = m4.checkbox("Include unrated / new listings", True)
        delivery_only = m5.checkbox("Delivery time recorded only", False)
        m6.checkbox("Show dish photos", True, key="show_photos", help="Needs internet. Turn off to use emoji art.")
    st.caption("Veg filter uses a name-based heuristic (e.g. 'Pure Veg' in the name), not verified dietary data.")

    if search_q.strip():
        found = rec.search_restaurants(search_q, city=city, limit=12)
        st.markdown(f'<div class="section-title">Search results for “{html.escape(search_q.strip())}”</div>', unsafe_allow_html=True)
        if found.empty:
            st.info("No restaurant names matched. Try a shorter search term.")
        else:
            render_grid(found, "search")
    else:
        results, meta = run_recommend(
            rec, city=city, cuisines=[cuisine] if cuisine else [], budget_max=budget, min_rating=min_rating,
            include_new=include_unrated, dietary_preference=dietary_pref, available_only=delivery_only,
            top_k=top_k, diversity=diversity,
        )
        head, exp = st.columns([3, 1], vertical_alignment="center")
        head.markdown('<div class="section-title">Recommendation Cards</div>', unsafe_allow_html=True)
        if results.empty:
            st.warning(meta["message"])
        else:
            with exp:
                to_csv_download(results, "foodwise_recommendations.csv", "⬇ Export results")
            st.caption(f"Top {len(results)} of {meta['candidates_found']:,} eligible restaurants. Match % is a ranking score, not a probability.")
            render_grid(results, "recommend")

elif page == "Similar restaurants":
    hero("FIND YOUR NEXT FAVOURITE", "More places like the ones you love.", "Choose a restaurant to discover catalogue entries with similar cuisine and spice profiles.")
    query = st.text_input("Search restaurant name", placeholder="Start typing a restaurant name…")
    matches = rec.search_restaurants(query, limit=60) if query.strip() else rec.df.head(0)
    if not matches.empty:
        labels = [f"{r['Restaurant Name']} — {r['city']}" for _, r in matches.iterrows()]
        chosen = st.selectbox("Choose a restaurant", labels)
        source = matches.iloc[labels.index(chosen)]
        same_city = st.checkbox(f"Keep results in {source['city']}", True)
        count = st.slider("Number of similar restaurants", 3, 15, 8)
        if st.button("Find similar places", type="primary"):
            similar = rec.get_similar_restaurants(source["Restaurant Name"], source["city"] if same_city else None, count, source_city=source["city"], source_index=int(source.name))
            st.session_state.similar_results = similar
        if "similar_results" in st.session_state:
            similar = st.session_state.similar_results
            st.subheader(f"Similar to {source['Restaurant Name']}")
            render_grid(similar, "similar")
            to_csv_download(similar, "foodwise_similar_restaurants.csv")
    elif query.strip():
        st.info("No restaurant names matched. Try a shorter search term.")
    else:
        st.info("Search for a restaurant to start exploring similar places.")

elif page == "Explore catalogue":
    hero("THE RESTAURANT DIRECTORY", "Explore the full royal catalogue.", "Search and filter the supplied listings by city, cuisine, and dietary preference.")
    c1, c2, c3, c4 = st.columns([1.2, 0.9, 0.9, 0.8])
    q = c1.text_input("Restaurant name contains", placeholder="e.g. Biryani, Thali, Cafe")
    city = c2.selectbox("City", ["All cities"] + rec.cities, key="catalog_city")
    cuisine = c3.selectbox("Cuisine", ["All cuisines"] + rec.cuisines, key="catalog_cuisine")
    diet_cat = c4.selectbox("Diet", ["Any", "Veg", "Non-Veg"], key="catalog_diet")

    data = rec.df.copy()
    if q:
        data = data[data["Restaurant Name"].str.contains(q, case=False, regex=False, na=False)]
    if city != "All cities":
        data = data[data["city"].eq(city)]
    if cuisine != "All cuisines":
        data = data[data["cuisine_list"].apply(lambda x: cuisine in x)]
    if diet_cat == "Veg":
        data = data[data["is_pure_veg"].astype(bool)]
    elif diet_cat == "Non-Veg":
        data = data[~data["is_pure_veg"].astype(bool)]

    st.write(f"**{len(data):,}** matching listings")
    view_cols = ["Restaurant Name", "city", "Cuisine", "rating_numeric", "price_numeric", "delivery_time_mins", "is_pure_veg", "safety_label"]
    st.dataframe(
        data[view_cols].rename(columns={
            "city": "City",
            "rating_numeric": "Rating",
            "price_numeric": "Price (₹)",
            "delivery_time_mins": "Delivery (min)",
            "is_pure_veg": "Pure Veg (Name)",
            "safety_label": "Source safety label"
        }),
        use_container_width=True,
        hide_index=True
    )
    to_csv_download(data, "foodwise_catalogue_export.csv", "⬇ Export filtered catalogue")

elif page == "Compare":
    hero("MAKE A CLEARER CHOICE", "Compare restaurants side by side.", "Select up to four catalogue listings and compare their available attributes.")
    compare_query = st.text_input("Find restaurants to compare", placeholder="Search by restaurant name")
    candidates = rec.search_restaurants(compare_query, limit=150) if compare_query.strip() else rec.df.head(0)
    labels = [f"{r['Restaurant Name']} — {r['city']} (row {i})" for i, r in candidates.iterrows()]
    chosen = st.multiselect("Choose up to four restaurants", labels, max_selections=4, placeholder="Search above, then select listings")
    if chosen:
        indices = [int(x.rsplit("row ", 1)[1].rstrip(")")) for x in chosen]
        comp = rec.df.loc[indices].copy()
        comp = comp.set_index(comp["Restaurant Name"] + " · " + comp["city"])
        fields = {
            "City": "city",
            "Cuisine": "Cuisine",
            "Rating": "rating_numeric",
            "Price per person (₹)": "price_numeric",
            "Delivery time (min)": "delivery_time_mins",
            "Source safety label": "safety_label",
            "Name-based veg indicator": "is_pure_veg"
        }
        table = pd.DataFrame({label: comp[col] for label, col in fields.items()}).T
        st.dataframe(table, use_container_width=True)
        st.caption("Missing values mean the source catalogue does not provide that attribute. Safety and vegetarian indicators are not independent verification.")

elif page == "Saved shortlist":
    hero("YOUR PERSONAL FEAST LIST", "Restaurants worth another look.", "Save listings while exploring and bring them together here. Your shortlist is kept for this app session.")
    if not st.session_state.favorites:
        st.info("No saved restaurants yet. Use the ♡ Save to Shortlist control on any restaurant card to add one here.")
    else:
        saved_rows = []
        for item in st.session_state.favorites:
            name, city = item.rsplit("|", 1)
            match = rec.df[(rec.df["Restaurant Name"] == name) & (rec.df["city"] == city)]
            if not match.empty:
                saved_rows.append(match.iloc[0])
        if saved_rows:
            saved = pd.DataFrame(saved_rows)
            st.write(f"**{len(saved)}** saved listings")
            render_grid(saved, "saved")
            to_csv_download(saved, "foodwise_saved_shortlist.csv")
        else:
            st.info("Saved listings are no longer present in the current catalogue.")

elif page == "Insights & model lab":
    hero("BEHIND THE RECOMMENDATIONS", "Explore the data. Inspect the model.", "Understand what the catalogue contains and run a reproducible, scenario-based comparison of ranking approaches.")
    df = rec.df
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Restaurants", f"{len(df):,}")
    m2.metric("Cities", f"{df['city'].nunique():,}")
    m3.metric("Median listed price", format_price(df["price_numeric"].median()))
    m4.metric("Rated listings", f"{df['rating_numeric'].notna().mean():.0%}")
    left, right = st.columns(2)
    with left:
        st.subheader("Top cities by listing count")
        city_counts = df["city"].value_counts().head(15).rename_axis("City").reset_index(name="Listings")
        st.plotly_chart(px.bar(city_counts, x="Listings", y="City", orientation="h", template="plotly_white", color_discrete_sequence=["#B71C1C"]), use_container_width=True)
    with right:
        st.subheader("Most common cuisine tags")
        counts = pd.Series([c for row in df["cuisine_list"] for c in row]).value_counts().head(15).rename_axis("Cuisine").reset_index(name="Listings")
        st.plotly_chart(px.bar(counts, x="Listings", y="Cuisine", orientation="h", template="plotly_white", color_discrete_sequence=["#D97706"]), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Rating distribution")
        st.plotly_chart(px.histogram(df.dropna(subset=["rating_numeric"]), x="rating_numeric", nbins=24, template="plotly_white", color_discrete_sequence=["#8E1B24"]), use_container_width=True)
    with right:
        st.subheader("Price tiers")
        tier = df["price_tier"].value_counts().rename_axis("Tier").reset_index(name="Listings")
        st.plotly_chart(px.pie(tier, names="Tier", values="Listings", hole=.55, template="plotly_white", color_discrete_sequence=["#B71C1C", "#F59E0B", "#3E0B10", "#FDE68A"]), use_container_width=True)
    st.markdown("---")
    st.subheader("Model comparison")
    st.markdown("Four ranking approaches are compared on 10 fixed city/cuisine/budget scenarios, all with the same eligibility filters and one listing per restaurant name. **Cuisine Match** only counts shared cuisine tags, so content-based methods score 1.0 by design. Compare the rating, price and diversity columns instead. There are no user interactions in the data, so this is **not** user satisfaction or real-world accuracy.")
    if st.button("▶ Run offline benchmark", type="primary"):
        with st.spinner("Evaluating recommendation scenarios…"):
            benchmark = RecommendationEvaluator(rec).run_benchmark(k=5)
            st.session_state.benchmark = benchmark
    if "benchmark" in st.session_state:
        st.dataframe(st.session_state.benchmark, use_container_width=True, hide_index=True)
        to_csv_download(st.session_state.benchmark, "foodwise_model_benchmark.csv")
    st.caption("Scenario results depend on the supplied catalogue and the hand-authored test cases. The random baseline uses a fixed seed.")

    st.markdown("---")
    st.subheader("Rating prediction model (supervised ML)")
    st.markdown(
        "A gradient boosting model learns to predict a restaurant's **rating** from its cuisine tags, city, price, "
        "delivery time, veg hint and safety label. It is used only to estimate a rating for **unrated** listings "
        "(about 16% of the catalogue), instead of a flat 3.5. Evaluation uses a hold-out split that keeps every outlet of one "
        "restaurant name on the same side, so chains cannot leak between train and test."
    )
    if st.button("▶ Train and evaluate rating model", type="primary"):
        with st.spinner("Training and evaluating (about 10 seconds)…"):
            predictor = RatingPredictor()
            st.session_state.ml_metrics = predictor.evaluate(rec.df)
            st.session_state.ml_importance = predictor.importance_
            st.session_state.ml_sizes = (predictor.train_size_, predictor.test_size_)
    if "ml_metrics" in st.session_state:
        train_n, test_n = st.session_state.ml_sizes
        st.caption(f"Trained on {train_n:,} rated listings, tested on {test_n:,} unseen restaurants. MAE and RMSE are in rating points (lower is better). R² of 0 means no better than predicting the average.")
        st.dataframe(st.session_state.ml_metrics, use_container_width=True, hide_index=True)
        imp = st.session_state.ml_importance
        fig = px.bar(imp.sort_values("MAE increase when shuffled"), x="MAE increase when shuffled", y="Feature",
                     orientation="h", template="plotly_white", color_discrete_sequence=["#B71C1C"],
                     title="Which features matter (permutation importance on the test set)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Ratings are noisy and the catalogue has no review counts, so the model explains only a small part of the variation. "
                "Its real gain is that its estimates are much closer to real ratings than the old flat 3.5. Estimates are labelled as estimates and never shown as real ratings.")

elif page == "About FoodWise":
    hero("THE PROJECT", "FoodWise 2.1 · Royal Spice Edition", "An explainable, content-based restaurant discovery system built with Python, TF-IDF and a transparent ranking layer.")
    st.subheader("How it works")
    steps = [
        ("01 · Validate & clean", "Checks the seven expected source fields, removes duplicate and unusable rows, and parses rating, price and delivery attributes."),
        ("02 · Represent restaurant profiles", "Builds text profiles from cuisine tags, city, price tier and a name-derived vegetarian hint; TF-IDF converts them into sparse vectors."),
        ("03 · Retrieve candidates", "Applies explicit city, budget, rating, delivery-record and dietary preference filters."),
        ("04 · Rank & diversify", "Combines content similarity with rating, price, delivery-time and source safety-label components. A diversity step reduces near-duplicate results."),
        ("05 · Explain", "Shows source-backed reasons and component scores. The displayed match score is a ranking score, not a probability."),
    ]
    for title, body in steps:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(body)
    st.subheader("Important data limitations")
    st.warning("The supplied data has no user IDs, clicks, review text, review counts, coordinates, live opening status or independently verified inspection records. FoodWise therefore does not claim collaborative filtering, real-time availability, map distance, verified hygiene, or guaranteed dietary suitability.")
    st.markdown("**Technology:** Python · Pandas · NumPy · Scikit-learn · Streamlit · Plotly")
    st.markdown("**Run locally:** `pip install -r requirements.txt` then `streamlit run app.py`.")

st.markdown("---")
st.markdown('<div class="small-note">FoodWise 2.1 · Royal Indian Spice & Feast UI · Recommendations are generated from the supplied catalogue and should be independently checked before making dining decisions.</div>', unsafe_allow_html=True)
