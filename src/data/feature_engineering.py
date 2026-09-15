"""
Feature engineering transformations for demand prediction and price elasticity modeling.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from src.utils.logger import get_logger

logger = get_logger("feature_engineering")


class FeatureEngineer:
    """
    Constructs domain-specific pricing and demand forecasting features.
    """

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies economic, temporal, and interaction feature engineering.
        """
        df = df.copy()

        # 1. Economic & Pricing Features
        eps = 1e-6
        df["price_ratio_competitor"] = np.round(df["price"] / np.maximum(df["competitor_price"], eps), 4)
        df["price_diff_competitor"] = np.round(df["price"] - df["competitor_price"], 2)
        df["price_cost_margin"] = np.round((df["price"] - df["cost_price"]) / np.maximum(df["price"], eps), 4)
        df["discount_ratio"] = np.round(df["discount_percent"] / 100.0, 4)

        # 2. Inventory & Stockout pressure
        df["stock_to_demand_ratio"] = np.round(df["stock_level"] / np.maximum(df["historical_demand_7d"], 1.0), 2)
        df["low_stock_flag"] = (df["stock_level"] < 100).astype(int)

        # 3. Cyclical Temporal Features
        if "month" in df.columns:
            df["month_sin"] = np.round(np.sin(2 * np.pi * df["month"] / 12.0), 4)
            df["month_cos"] = np.round(np.cos(2 * np.pi * df["month"] / 12.0), 4)
        if "day_of_week" in df.columns:
            df["dow_sin"] = np.round(np.sin(2 * np.pi * df["day_of_week"] / 7.0), 4)
            df["dow_cos"] = np.round(np.cos(2 * np.pi * df["day_of_week"] / 7.0), 4)

        # 4. Marketing & Social Proof Interactions
        df["ad_spend_log"] = np.round(np.log1p(df["ad_spend_usd"]), 4)
        df["rating_weighted_reviews"] = np.round(df["rating"] * np.log1p(df["review_count"]), 4)

        # 5. Log transforms for Econometric Elasticity modeling
        df["log_price"] = np.round(np.log(np.maximum(df["price"], 0.01)), 4)
        df["log_competitor_price"] = np.round(np.log(np.maximum(df["competitor_price"], 0.01)), 4)
        if "demand_units" in df.columns:
            df["log_demand"] = np.round(np.log(np.maximum(df["demand_units"], 1.0)), 4)

        return df
