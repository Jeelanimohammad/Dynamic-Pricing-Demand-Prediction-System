"""
Multi-category realistic synthetic market sales data generator.
Simulates real-world market dynamics including price elasticity, competitor actions,
inventory levels, seasonal surges, marketing ad spend, and customer ratings.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from src.config import RANDOM_SEED, PRODUCT_CATEGORIES, RAW_DATA_FILE
from src.utils.logger import get_logger

logger = get_logger("data_generator")

# Product metadata template catalog
CATALOG_DEFINITIONS = [
    # Electronics
    {"product_id": "ELEC_001", "name": "Pro Wireless Noise-Cancelling Headphones", "category": "Electronics", "base_cost": 85.0, "base_price": 199.99, "base_elasticity": -1.85, "base_daily_volume": 45},
    {"product_id": "ELEC_002", "name": "Ultra HD 4K Smart TV 55-inch", "category": "Electronics", "base_cost": 280.0, "base_price": 549.99, "base_elasticity": -2.20, "base_daily_volume": 20},
    {"product_id": "ELEC_003", "name": "Smart Fitness Watch Gen-4", "category": "Electronics", "base_cost": 50.0, "base_price": 129.99, "base_elasticity": -1.60, "base_daily_volume": 60},
    {"product_id": "ELEC_004", "name": "Ergonomic Mechanical Keyboard", "category": "Electronics", "base_cost": 35.0, "base_price": 89.99, "base_elasticity": -1.40, "base_daily_volume": 40},
    # Apparel & Fashion
    {"product_id": "APP_001", "name": "Premium Merino Wool Sweater", "category": "Apparel & Fashion", "base_cost": 28.0, "base_price": 79.99, "base_elasticity": -1.95, "base_daily_volume": 55},
    {"product_id": "APP_002", "name": "Breathable Performance Running Shoes", "category": "Apparel & Fashion", "base_cost": 42.0, "base_price": 119.99, "base_elasticity": -1.75, "base_daily_volume": 75},
    {"product_id": "APP_003", "name": "Classic Waterproof Trench Coat", "category": "Apparel & Fashion", "base_cost": 55.0, "base_price": 149.99, "base_elasticity": -2.10, "base_daily_volume": 25},
    # Hospitality & Stays
    {"product_id": "HOSP_001", "name": "Deluxe City Center King Room (Per Night)", "category": "Hospitality & Stays", "base_cost": 45.0, "base_price": 175.00, "base_elasticity": -1.90, "base_daily_volume": 85},
    {"product_id": "HOSP_002", "name": "Executive Skyline Suite (Per Night)", "category": "Hospitality & Stays", "base_cost": 90.0, "base_price": 350.00, "base_elasticity": -1.35, "base_daily_volume": 15},
    # Rideshare & Mobility
    {"product_id": "RIDE_001", "name": "Standard Urban Commute Ride (10 km)", "category": "Rideshare & Mobility", "base_cost": 8.0, "base_price": 24.50, "base_elasticity": -2.40, "base_daily_volume": 250},
    {"product_id": "RIDE_002", "name": "Premium Black Airport Transfer", "category": "Rideshare & Mobility", "base_cost": 22.0, "base_price": 65.00, "base_elasticity": -1.25, "base_daily_volume": 70},
    # Home & Living
    {"product_id": "HOME_001", "name": "Espresso Machine & Milk Frother", "category": "Home & Living", "base_cost": 65.0, "base_price": 159.99, "base_elasticity": -1.70, "base_daily_volume": 30},
    {"product_id": "HOME_002", "name": "Cordless Handheld Vacuum Cleaner", "category": "Home & Living", "base_cost": 40.0, "base_price": 99.99, "base_elasticity": -2.05, "base_daily_volume": 45},
]


class MarketDataGenerator:
    """
    Generates rich, realistic historical market sales and pricing data.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        np.random.seed(seed)
        self.catalog = CATALOG_DEFINITIONS

    def generate(
        self,
        days: int = 365,
        start_date: str = "2025-01-01",
        save_csv: bool = True
    ) -> pd.DataFrame:
        """
        Generates simulated daily market transactions for all products in catalog.
        """
        logger.info(f"Generating market dataset for {days} days across {len(self.catalog)} products...")
        np.random.seed(self.seed)

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        records: List[Dict] = []

        customer_segments = ["Budget", "Standard", "Premium", "Enterprise"]
        segment_weights = [0.35, 0.40, 0.20, 0.05]

        for prod in self.catalog:
            product_id = prod["product_id"]
            name = prod["name"]
            category = prod["category"]
            cost_price = prod["base_cost"]
            base_price = prod["base_price"]
            elasticity = prod["base_elasticity"]
            base_vol = prod["base_daily_volume"]

            current_stock = np.random.randint(300, 1000)
            rolling_demand_buffer: List[float] = []

            for day_idx in range(days):
                current_date = start_dt + timedelta(days=day_idx)
                day_of_week = current_date.weekday()
                month = current_date.month
                day_of_month = current_date.day
                is_weekend = 1 if day_of_week in [5, 6] else 0

                # Seasonality factor
                season_factor = 1.0
                if category in ["Electronics", "Apparel & Fashion"]:
                    # Q4 holiday surge (Nov-Dec)
                    if month in [11, 12]:
                        season_factor = 1.45
                    elif month in [7, 8]:
                        season_factor = 1.15
                elif category == "Hospitality & Stays":
                    # Summer peak & weekend surge
                    if month in [6, 7, 8, 12]:
                        season_factor = 1.50
                    if is_weekend:
                        season_factor *= 1.35
                elif category == "Rideshare & Mobility":
                    # Commute surge Mon-Fri, nightlife Fri-Sat
                    if is_weekend:
                        season_factor = 1.25
                    elif day_of_week in [0, 4]:
                        season_factor = 1.20

                # Holiday effect (Black Friday, New Year, Prime Day, Cyber Monday)
                holiday_effect = 0.0
                if month == 11 and 20 <= day_of_month <= 30:
                    holiday_effect = 0.60
                elif month == 12 and 15 <= day_of_month <= 31:
                    holiday_effect = 0.45
                elif month == 7 and 10 <= day_of_month <= 15:
                    holiday_effect = 0.35

                # Competitor price variation (+/- 15% random walk with mean reversion)
                comp_noise = np.random.normal(0, 0.05)
                competitor_price = round(max(cost_price * 1.05, base_price * (1.0 + comp_noise)), 2)

                # Our pricing strategy simulation (mix of discounts, dynamic surge, and base price)
                discount_prob = np.random.rand()
                if discount_prob < 0.20:
                    discount_pct = np.random.choice([0.05, 0.10, 0.15, 0.20, 0.25])
                else:
                    discount_pct = 0.0

                surge_multiplier = 1.0
                if holiday_effect > 0 or (category == "Hospitality & Stays" and is_weekend):
                    surge_multiplier = np.random.uniform(1.05, 1.25)

                price_noise = np.random.uniform(0.95, 1.05)
                applied_price = round(max(cost_price * 1.10, base_price * (1.0 - discount_pct) * surge_multiplier * price_noise), 2)

                # Microeconomic demand formulation:
                # Q = Q_base * (P / P_base)^elasticity * (P_comp / P)^cross_elasticity * Seasonality * Holiday * Ratings * AdSpend + Noise
                cross_elasticity = 0.65
                price_ratio = applied_price / base_price
                comp_ratio = competitor_price / applied_price

                # Ratings (e.g. 4.1 to 4.9)
                rating = round(np.random.uniform(4.0, 4.9), 2)
                review_count = int(np.random.normal(500, 50) + day_idx * 2)
                rating_multiplier = 0.8 + (rating / 5.0) * 0.4

                # Ad spend in USD
                ad_spend = round(max(0.0, np.random.normal(150, 40) * (1.5 if holiday_effect > 0 else 1.0)), 2)
                ad_multiplier = 1.0 + (np.log1p(ad_spend) / 25.0)

                # Segment
                customer_segment = np.random.choice(customer_segments, p=segment_weights)
                segment_multiplier = {"Budget": 1.1, "Standard": 1.0, "Premium": 0.85, "Enterprise": 0.70}[customer_segment]

                # Calculate theoretical demand
                elasticity_impact = (price_ratio ** elasticity)
                comp_impact = (comp_ratio ** cross_elasticity)
                total_multiplier = season_factor * (1.0 + holiday_effect) * rating_multiplier * ad_multiplier * segment_multiplier

                raw_demand = base_vol * elasticity_impact * comp_impact * total_multiplier
                random_shock = np.random.normal(1.0, 0.08)
                expected_demand = max(1.0, raw_demand * random_shock)

                # Inventory constraint (cannot sell more than stock)
                actual_sales_units = int(min(round(expected_demand), current_stock))

                # Update stock and restock if needed
                current_stock -= actual_sales_units
                if current_stock < 80:
                    current_stock += np.random.randint(400, 800)

                # Rolling history
                rolling_demand_buffer.append(actual_sales_units)
                hist_7d = float(np.mean(rolling_demand_buffer[-7:])) if len(rolling_demand_buffer) >= 7 else float(base_vol)
                hist_30d = float(np.mean(rolling_demand_buffer[-30:])) if len(rolling_demand_buffer) >= 30 else float(base_vol)

                revenue = round(applied_price * actual_sales_units, 2)
                total_cost = round(cost_price * actual_sales_units, 2)
                profit = round(revenue - total_cost, 2)

                season_name = "Winter" if month in [12, 1, 2] else "Spring" if month in [3, 4, 5] else "Summer" if month in [6, 7, 8] else "Fall"

                record = {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "product_id": product_id,
                    "product_name": name,
                    "category": category,
                    "price": applied_price,
                    "base_price": base_price,
                    "cost_price": cost_price,
                    "competitor_price": competitor_price,
                    "discount_percent": round(discount_pct * 100, 1),
                    "stock_level": current_stock,
                    "rating": rating,
                    "review_count": review_count,
                    "ad_spend_usd": ad_spend,
                    "holiday_effect": round(holiday_effect, 2),
                    "customer_segment": customer_segment,
                    "season": season_name,
                    "month": month,
                    "day_of_week": day_of_week,
                    "day_of_week_name": current_date.strftime("%A"),
                    "day_of_month": day_of_month,
                    "is_weekend": is_weekend,
                    "historical_demand_7d": round(hist_7d, 2),
                    "historical_demand_30d": round(hist_30d, 2),
                    "demand_units": actual_sales_units,
                    "revenue": revenue,
                    "profit": profit,
                }
                records.append(record)

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["date", "product_id"]).reset_index(drop=True)

        if save_csv:
            RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(RAW_DATA_FILE, index=False)
            logger.info(f"Saved generated raw dataset ({len(df)} rows) to {RAW_DATA_FILE}")

        return df


if __name__ == "__main__":
    gen = MarketDataGenerator()
    df_gen = gen.generate(days=365)
    print(f"Generated {len(df_gen)} records.")
    print(df_gen.head(3))
