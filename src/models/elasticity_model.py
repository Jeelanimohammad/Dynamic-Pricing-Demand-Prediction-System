"""
Econometric Price Elasticity of Demand (PED) and Cross-Price Elasticity estimation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.linear_model import Ridge, LinearRegression
from src.utils.logger import get_logger

logger = get_logger("elasticity_model")


class PriceElasticityModel:
    """
    Econometric log-log model for calculating Point and Arc Price Elasticity of Demand.
    """

    def __init__(self):
        self.product_elasticities: Dict[str, Dict[str, Any]] = {}
        self.category_elasticities: Dict[str, float] = {}
        self.global_elasticity: float = -1.5

    def fit(self, df: pd.DataFrame) -> "PriceElasticityModel":
        """
        Estimates price elasticity coefficients per product and per category using log-log regressions.
        """
        df = df.copy()
        eps = 1e-6

        # Ensure required log columns exist
        if "log_price" not in df.columns:
            df["log_price"] = np.log(np.maximum(df["price"], eps))
        if "log_competitor_price" not in df.columns:
            df["log_competitor_price"] = np.log(np.maximum(df["competitor_price"], eps))
        if "log_demand" not in df.columns:
            df["log_demand"] = np.log(np.maximum(df["demand_units"], 1.0))

        # 1. Global Elasticity
        X_global = df[["log_price", "log_competitor_price", "holiday_effect", "rating"]].fillna(0)
        y_global = df["log_demand"]
        reg_global = LinearRegression().fit(X_global, y_global)
        self.global_elasticity = round(float(reg_global.coef_[0]), 3)

        # 2. Category-level Elasticity
        for cat, cat_group in df.groupby("category"):
            if len(cat_group) > 20:
                X_cat = cat_group[["log_price", "log_competitor_price", "holiday_effect", "rating"]].fillna(0)
                y_cat = cat_group["log_demand"]
                reg_cat = LinearRegression().fit(X_cat, y_cat)
                self.category_elasticities[cat] = round(float(reg_cat.coef_[0]), 3)

        # 3. Product-level Elasticity & Cross-Elasticity
        for prod_id, group in df.groupby("product_id"):
            if len(group) >= 15:
                X_p = group[["log_price", "log_competitor_price", "holiday_effect"]].fillna(0)
                y_p = group["log_demand"]
                reg_p = Ridge(alpha=1.0).fit(X_p, y_p)

                own_ped = float(reg_p.coef_[0])
                cross_ped = float(reg_p.coef_[1])
                category = group["category"].iloc[0]
                prod_name = group["product_name"].iloc[0] if "product_name" in group.columns else prod_id

                # Sanity guardrails for realistic microeconomics (PED is typically negative)
                if own_ped >= 0:
                    own_ped = self.category_elasticities.get(category, self.global_elasticity)

                # Classify elasticity regime
                abs_ped = abs(own_ped)
                if abs_ped > 1.0:
                    regime = "Elastic (Price Sensitive)"
                elif abs_ped < 1.0:
                    regime = "Inelastic (Price Insensitive)"
                else:
                    regime = "Unitary Elastic"

                self.product_elasticities[prod_id] = {
                    "product_id": prod_id,
                    "product_name": prod_name,
                    "category": category,
                    "own_price_elasticity": round(own_ped, 3),
                    "cross_price_elasticity": round(cross_ped, 3),
                    "regime": regime,
                    "sample_count": len(group),
                    "mean_price": round(float(group["price"].mean()), 2),
                    "mean_demand": round(float(group["demand_units"].mean()), 2)
                }

        logger.info(f"Fitted price elasticity models for {len(self.product_elasticities)} products.")
        return self

    def get_product_elasticity(self, product_id: str) -> Dict[str, Any]:
        """
        Returns elasticity profile for a product.
        """
        if product_id in self.product_elasticities:
            return self.product_elasticities[product_id]
        return {
            "product_id": product_id,
            "own_price_elasticity": self.global_elasticity,
            "cross_price_elasticity": 0.50,
            "regime": "Elastic" if abs(self.global_elasticity) > 1 else "Inelastic"
        }

    @staticmethod
    def calculate_arc_elasticity(
        p1: float, q1: float, p2: float, q2: float
    ) -> float:
        """
        Calculates midpoint (Arc) elasticity between two (Price, Quantity) states:
        E_arc = ((Q2 - Q1) / ((Q1 + Q2)/2)) / ((P2 - P1) / ((P1 + P2)/2))
        """
        if p1 == p2 or (p1 + p2) == 0 or (q1 + q2) == 0:
            return 0.0

        pct_delta_q = (q2 - q1) / ((q1 + q2) / 2.0)
        pct_delta_p = (p2 - p1) / ((p1 + p2) / 2.0)

        if pct_delta_p == 0:
            return 0.0

        return round(float(pct_delta_q / pct_delta_p), 3)

    @staticmethod
    def estimate_demand_shift(
        current_price: float,
        new_price: float,
        current_demand: float,
        elasticity: float
    ) -> float:
        """
        Estimates new demand volume following a price change using constant elasticity formulation:
        Q_new = Q_current * (P_new / P_current)^elasticity
        """
        if current_price <= 0 or new_price <= 0 or current_demand <= 0:
            return 0.0

        price_ratio = new_price / current_price
        new_demand = current_demand * (price_ratio ** elasticity)
        return max(0.0, float(new_demand))
