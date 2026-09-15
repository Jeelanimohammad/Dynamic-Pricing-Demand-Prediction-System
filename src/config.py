"""
Configuration and constants for Dynamic Pricing & Demand Prediction System.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_DATA_FILE = RAW_DATA_DIR / "market_sales_data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_sales_data.parquet"
FEATURE_METADATA_FILE = PROCESSED_DATA_DIR / "feature_metadata.json"
MODEL_CHECKPOINT_FILE = MODELS_DIR / "demand_predictor_ensemble.joblib"
ELASTICITY_MODEL_FILE = MODELS_DIR / "price_elasticity_model.joblib"
METRICS_FILE = MODELS_DIR / "model_metrics.json"

# Database Configuration
SQLITE_DB_PATH = DATA_DIR / "pricing_system.db"
DATABASE_URI = f"sqlite:///{SQLITE_DB_PATH}"

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Industry Categories
PRODUCT_CATEGORIES = [
    "Electronics",
    "Apparel & Fashion",
    "Hospitality & Stays",
    "Rideshare & Mobility",
    "Home & Living",
]

# Feature definitions
CATEGORICAL_FEATURES = [
    "category",
    "product_id",
    "customer_segment",
    "day_of_week_name",
    "season",
]

NUMERICAL_FEATURES = [
    "price",
    "cost_price",
    "competitor_price",
    "historical_demand_7d",
    "historical_demand_30d",
    "stock_level",
    "discount_percent",
    "price_ratio_competitor",
    "price_cost_margin",
    "rating",
    "review_count",
    "ad_spend_usd",
    "holiday_effect",
    "month",
    "day_of_week",
    "day_of_month",
    "is_weekend",
]

TARGET_COLUMN = "demand_units"

# Optimization Constraints
DEFAULT_MIN_MARGIN_PCT = 0.15      # Minimum 15% profit margin
DEFAULT_MAX_PRICE_CHANGE = 0.30   # Maximum 30% price shift in a single update
DEFAULT_SEARCH_GRANULARITY = 100   # Grid evaluation points for optimization
