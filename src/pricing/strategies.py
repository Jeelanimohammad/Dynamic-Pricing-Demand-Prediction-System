"""
Rule-based dynamic pricing policies (Surge, Clearance, Competitor-Matching, Margin-Protect).
"""
from typing import Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger("pricing_strategies")


class PricingStrategyEngine:
    """
    Applies business heuristics and market-driven pricing adjustments.
    """

    @staticmethod
    def apply_surge_pricing(
        base_price: float,
        stock_level: int,
        demand_forecast: float,
        is_peak_period: bool = False
    ) -> Dict[str, Any]:
        """
        Surge pricing for high demand or low stock availability.
        Multiplier scales from 1.0x up to 1.40x.
        """
        stock_ratio = stock_level / max(demand_forecast, 1.0)
        multiplier = 1.0
        reason = "Normal market conditions"

        if stock_ratio < 2.0:
            multiplier = 1.25
            reason = "High scarcity: Critical stock level relative to demand"
        elif stock_ratio < 5.0:
            multiplier = 1.15
            reason = "Moderate scarcity: Low stock buffer"

        if is_peak_period:
            multiplier *= 1.10
            reason += " + Peak seasonal demand surge"

        multiplier = min(multiplier, 1.40)  # Maximum surge cap
        recommended_price = round(base_price * multiplier, 2)

        return {
            "strategy": "Surge Pricing",
            "recommended_price": recommended_price,
            "surge_multiplier": round(multiplier, 3),
            "reason": reason
        }

    @staticmethod
    def apply_clearance_markdown(
        current_price: float,
        cost_price: float,
        stock_level: int,
        days_in_inventory: int = 45,
        min_margin_pct: float = 0.10
    ) -> Dict[str, Any]:
        """
        Markdown / Clearance pricing for overstocked or aging inventory.
        Guarantees price never drops below cost * (1 + min_margin_pct).
        """
        min_floor_price = round(cost_price * (1.0 + min_margin_pct), 2)
        markdown_pct = 0.0
        reason = "Standard inventory velocity"

        if days_in_inventory > 90 or stock_level > 500:
            markdown_pct = 0.25
            reason = "Deep clearance: High aging stock inventory"
        elif days_in_inventory > 60 or stock_level > 300:
            markdown_pct = 0.15
            reason = "Moderate markdown: Slow-moving inventory clearance"

        discounted_price = current_price * (1.0 - markdown_pct)
        recommended_price = max(min_floor_price, round(discounted_price, 2))

        return {
            "strategy": "Clearance Markdown",
            "recommended_price": recommended_price,
            "markdown_percent": round(markdown_pct * 100, 1),
            "price_floor": min_floor_price,
            "reason": reason
        }

    @staticmethod
    def apply_competitor_matching(
        current_price: float,
        cost_price: float,
        competitor_price: float,
        undercut_pct: float = 0.02,
        min_margin_pct: float = 0.12
    ) -> Dict[str, Any]:
        """
        Competitor-anchored pricing: Undercuts or matches competitor price while respecting profit floor.
        """
        min_allowed_price = round(cost_price * (1.0 + min_margin_pct), 2)
        target_price = round(competitor_price * (1.0 - undercut_pct), 2)

        if target_price < min_allowed_price:
            recommended_price = min_allowed_price
            reason = f"Floor protected: Target ${target_price} below minimum margin constraint (${min_allowed_price})"
        else:
            recommended_price = target_price
            reason = f"Undercutting competitor (${competitor_price}) by {undercut_pct*100}%"

        return {
            "strategy": "Competitor Matching",
            "recommended_price": recommended_price,
            "competitor_price": competitor_price,
            "min_allowed_price": min_allowed_price,
            "reason": reason
        }
