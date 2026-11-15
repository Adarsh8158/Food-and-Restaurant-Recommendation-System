"""Formatting helpers shared by the FoodWise interface."""
import pandas as pd

def format_price(value):
    return "Price not listed" if pd.isna(value) else f"₹{float(value):,.0f} for one"
