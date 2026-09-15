"""
Pydantic schemas for API request and response data models.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ProductFeatureInput(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "ELEC_001"})
    category: str = Field(..., json_schema_extra={"example": "Electronics"})
    price: float = Field(..., gt=0, json_schema_extra={"example": 199.99})
    cost_price: float = Field(..., gt=0, json_schema_extra={"example": 85.00})
    competitor_price: float = Field(..., gt=0, json_schema_extra={"example": 205.00})
    stock_level: int = Field(default=250, ge=0)
    discount_percent: float = Field(default=0.0, ge=0, le=100)
    rating: float = Field(default=4.5, ge=1.0, le=5.0)
    review_count: int = Field(default=350, ge=0)
    ad_spend_usd: float = Field(default=150.0, ge=0)
    holiday_effect: float = Field(default=0.0, ge=0, le=1.0)
    customer_segment: str = Field(default="Standard")
    season: str = Field(default="Fall")
    month: int = Field(default=10, ge=1, le=12)
    day_of_week: int = Field(default=2, ge=0, le=6)
    historical_demand_7d: float = Field(default=45.0, ge=0)
    historical_demand_30d: float = Field(default=42.0, ge=0)
    is_weekend: int = Field(default=0, ge=0, le=1)

class DemandPredictionResponse(BaseModel):
    product_id: str
    price: float
    predicted_demand_units: float
    expected_revenue: float
    expected_profit: float
    unit_margin: float
    profit_margin_pct: float
    submodel_predictions: Optional[Dict[str, float]] = None

class OptimizationRequest(BaseModel):
    features: ProductFeatureInput
    objective: str = Field(default="profit", json_schema_extra={"example": "profit"})  # 'profit', 'revenue', 'balanced'
    min_margin_pct: float = Field(default=0.15, ge=0.0, le=0.90)
    max_price_change_pct: float = Field(default=0.30, ge=0.05, le=1.0)
    search_points: int = Field(default=100, ge=20, le=500)

class OptimizationResponse(BaseModel):
    product_id: str
    objective: str
    current_price: float
    optimal_price: float
    price_change_pct: float
    cost_price: float
    competitor_price: float
    min_allowed_price: float
    max_allowed_price: float
    baseline_metrics: Dict[str, Any]
    optimal_metrics: Dict[str, Any]
    elasticity_info: Dict[str, Any]
    price_curve: Dict[str, List[float]]

class WhatIfRequest(BaseModel):
    features: ProductFeatureInput
    price_delta_pct: float = Field(default=0.0)
    competitor_delta_pct: float = Field(default=0.0)
    ad_spend_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)
    holiday_effect: float = Field(default=0.0, ge=0.0, le=1.0)

class ABTestRequest(BaseModel):
    features: ProductFeatureInput
    control_price: float = Field(..., gt=0)
    treatment_price: float = Field(..., gt=0)
    sample_size: int = Field(default=1000, ge=100, le=50000)

class RetrainRequest(BaseModel):
    force_regenerate: bool = Field(default=False)
    simulation_days: int = Field(default=365, ge=30, le=1000)
