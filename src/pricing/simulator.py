"""
What-If Scenario Simulation and A/B Testing Cohort Simulator.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy import stats
from src.utils.logger import get_logger

logger = get_logger("pricing_simulator")


class PricingSimulator:
    """
    Simulates market scenarios and randomized A/B pricing experiments.
    """

    def __init__(self, optimizer=None):
        self.optimizer = optimizer

    def run_what_if_scenario(
        self,
        base_features: Dict[str, Any],
        price_delta_pct: float = 0.0,
        competitor_delta_pct: float = 0.0,
        ad_spend_multiplier: float = 1.0,
        holiday_effect: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluates demand, revenue, and profit under simulated external shifts.
        """
        scenario_features = dict(base_features)

        # Apply perturbations
        curr_price = float(scenario_features["price"])
        new_price = round(curr_price * (1.0 + price_delta_pct / 100.0), 2)
        scenario_features["price"] = new_price

        curr_comp = float(scenario_features.get("competitor_price", curr_price))
        new_comp = round(curr_comp * (1.0 + competitor_delta_pct / 100.0), 2)
        scenario_features["competitor_price"] = new_comp

        curr_ad = float(scenario_features.get("ad_spend_usd", 100.0))
        scenario_features["ad_spend_usd"] = round(curr_ad * ad_spend_multiplier, 2)

        scenario_features["holiday_effect"] = holiday_effect

        # Predict outcome
        predicted_demand = self.optimizer.predict_demand_at_price(scenario_features, new_price) if self.optimizer else 50.0
        cost_price = float(scenario_features["cost_price"])
        revenue = round(new_price * predicted_demand, 2)
        profit = round((new_price - cost_price) * predicted_demand, 2)
        margin_pct = round(((new_price - cost_price) / max(new_price, 1e-6)) * 100, 2)

        return {
            "scenario_name": f"P: {price_delta_pct:+.1f}%, Comp: {competitor_delta_pct:+.1f}%, AdSpend: {ad_spend_multiplier:.1f}x",
            "simulated_price": new_price,
            "simulated_competitor_price": new_comp,
            "simulated_demand": round(predicted_demand, 2),
            "expected_revenue": revenue,
            "expected_profit": profit,
            "profit_margin_pct": margin_pct,
            "parameters": {
                "price_delta_pct": price_delta_pct,
                "competitor_delta_pct": competitor_delta_pct,
                "ad_spend_multiplier": ad_spend_multiplier,
                "holiday_effect": holiday_effect
            }
        }

    def simulate_ab_test(
        self,
        product_features: Dict[str, Any],
        control_price: float,
        treatment_price: float,
        sample_size: int = 1000,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Simulates randomized A/B test between static baseline price (Control)
        and dynamic optimized price (Treatment). Computes statistical significance (p-value, t-stat).
        """
        np.random.seed(seed)
        cost_price = float(product_features["cost_price"])

        # Base conversion probability
        base_demand_control = self.optimizer.predict_demand_at_price(product_features, control_price) if self.optimizer else 40.0
        base_demand_treat = self.optimizer.predict_demand_at_price(product_features, treatment_price) if self.optimizer else 36.0

        # Purchase conversion probability per visitor impression
        # Assuming baseline impression pool
        p_conv_control = min(0.40, max(0.05, base_demand_control / 100.0))
        p_conv_treat = min(0.40, max(0.05, base_demand_treat / 100.0))

        # Simulate Bernoulli buyer choices
        control_conversions = np.random.binomial(1, p_conv_control, size=sample_size)
        treat_conversions = np.random.binomial(1, p_conv_treat, size=sample_size)

        # Revenue & Profit per visitor
        control_revenues = control_conversions * control_price
        treat_revenues = treat_conversions * treatment_price

        control_profits = control_conversions * (control_price - cost_price)
        treat_profits = treat_conversions * (treatment_price - cost_price)

        # Aggregate Statistics
        total_rev_a = float(np.sum(control_revenues))
        total_rev_b = float(np.sum(treat_revenues))
        total_profit_a = float(np.sum(control_profits))
        total_profit_b = float(np.sum(treat_profits))

        rev_uplift_pct = round(((total_rev_b - total_rev_a) / max(total_rev_a, 1e-6)) * 100, 2)
        profit_uplift_pct = round(((total_profit_b - total_profit_a) / max(total_profit_a, 1e-6)) * 100, 2)

        # Two-sample t-test for revenue per visitor
        t_stat, p_val = stats.ttest_ind(treat_revenues, control_revenues, equal_var=False)

        is_significant = bool(p_val < 0.05)

        return {
            "sample_size_per_variant": sample_size,
            "control_group_A": {
                "name": "Control (Static Base Price)",
                "price": round(control_price, 2),
                "conversions": int(np.sum(control_conversions)),
                "conversion_rate_pct": round(float(np.mean(control_conversions) * 100), 2),
                "total_revenue": round(total_rev_a, 2),
                "total_profit": round(total_profit_a, 2),
                "avg_revenue_per_visitor": round(float(np.mean(control_revenues)), 2)
            },
            "treatment_group_B": {
                "name": "Treatment (Dynamic Optimized Price)",
                "price": round(treatment_price, 2),
                "conversions": int(np.sum(treat_conversions)),
                "conversion_rate_pct": round(float(np.mean(treat_conversions) * 100), 2),
                "total_revenue": round(total_rev_b, 2),
                "total_profit": round(total_profit_b, 2),
                "avg_revenue_per_visitor": round(float(np.mean(treat_revenues)), 2)
            },
            "comparison": {
                "revenue_uplift_percent": rev_uplift_pct,
                "profit_uplift_percent": profit_uplift_pct,
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val), 5),
                "is_statistically_significant": is_significant,
                "confidence_level": "95%+" if is_significant else "< 95%"
            }
        }
