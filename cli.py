"""
Command-Line Interface (CLI) for Dynamic Pricing & Demand Prediction System.
"""
import argparse
import sys
import json
import joblib
import pandas as pd
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from src.config import (
    RAW_DATA_FILE,
    MODEL_CHECKPOINT_FILE,
    ELASTICITY_MODEL_FILE,
    METRICS_FILE,
    SQLITE_DB_PATH
)
from src.data.generator import MarketDataGenerator, CATALOG_DEFINITIONS
from src.data.preprocessor import DataPreprocessor
from src.models.train import ModelTrainer
from src.pricing.optimizer import PriceOptimizer
from src.pricing.simulator import PricingSimulator
from src.utils.logger import get_logger

logger = get_logger("cli")


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Pricing & Demand Prediction System CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command 1: generate-data
    gen_parser = subparsers.add_parser("generate-data", help="Generate synthetic market transaction data")
    gen_parser.add_argument("--days", type=int, default=365, help="Number of simulated days")

    # Command 2: train
    train_parser = subparsers.add_parser("train", help="Train forecasting and elasticity models")
    train_parser.add_argument("--force-regenerate", action="store_true", help="Force new dataset generation")
    train_parser.add_argument("--days", type=int, default=365, help="Days of training data")

    # Command 3: optimize
    opt_parser = subparsers.add_parser("optimize", help="Compute optimal price for a product")
    opt_parser.add_argument("--product-id", type=str, default="ELEC_001", help="Product SKU ID")
    opt_parser.add_argument("--price", type=float, default=199.99, help="Current selling price")
    opt_parser.add_argument("--cost", type=float, default=85.00, help="Unit cost price")
    opt_parser.add_argument("--competitor-price", type=float, default=205.00, help="Competitor price")
    opt_parser.add_argument("--objective", type=str, default="profit", choices=["profit", "revenue", "balanced"])
    opt_parser.add_argument("--min-margin", type=float, default=0.15, help="Minimum profit margin fraction (0-1)")

    # Command 4: predict
    pred_parser = subparsers.add_parser("predict", help="Predict demand for product at target price")
    pred_parser.add_argument("--product-id", type=str, default="ELEC_001", help="Product SKU ID")
    pred_parser.add_argument("--price", type=float, default=199.99, help="Target price")
    pred_parser.add_argument("--cost", type=float, default=85.00, help="Cost price")

    # Command 5: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Display model evaluation metrics")

    args = parser.parse_args()

    if args.command == "generate-data":
        generator = MarketDataGenerator()
        df = generator.generate(days=args.days, save_csv=True)
        print(f"✅ Generated {len(df)} transactions across {args.days} days.")

    elif args.command == "train":
        trainer = ModelTrainer()
        result = trainer.train_pipeline(force_regenerate=args.force_regenerate, days=args.days)
        print("✅ Training complete. Evaluation metrics:")
        print(json.dumps(result["evaluation_report"]["model_metrics"], indent=2))

    elif args.command == "optimize":
        if not MODEL_CHECKPOINT_FILE.exists():
            print("Models not found. Training first...")
            ModelTrainer().train_pipeline(force_regenerate=False)

        artifact = joblib.load(MODEL_CHECKPOINT_FILE)
        elast = joblib.load(ELASTICITY_MODEL_FILE)
        optimizer = PriceOptimizer(
            demand_model=artifact["model"],
            transformer=artifact["transformer"],
            elasticity_model=elast
        )

        input_data = {
            "product_id": args.product_id,
            "category": "Electronics",
            "price": args.price,
            "base_price": args.price,
            "cost_price": args.cost,
            "competitor_price": args.competitor_price,
            "stock_level": 300,
            "discount_percent": 0.0,
            "rating": 4.5,
            "review_count": 350,
            "ad_spend_usd": 150.0,
            "holiday_effect": 0.0,
            "customer_segment": "Standard",
            "season": "Fall",
            "month": 10,
            "day_of_week": 2,
            "is_weekend": 0,
            "historical_demand_7d": 45.0,
            "historical_demand_30d": 42.0
        }

        res = optimizer.optimize_price(
            product_features=input_data,
            objective=args.objective,
            min_margin_pct=args.min_margin
        )

        print("\n" + "="*50)
        print(f"🎯 OPTIMIZATION RESULT: {args.product_id}")
        print("="*50)
        print(f"Current Price:      ${res['current_price']:.2f}")
        print(f"Optimal Price:      ${res['optimal_price']:.2f} ({res['price_change_pct']:+.1f}%)")
        print(f"Predicted Demand:   {res['optimal_metrics']['predicted_demand']:.1f} units")
        print(f"Expected Revenue:   ${res['optimal_metrics']['expected_revenue']:,.2f} ({res['optimal_metrics']['revenue_uplift_pct']:+.1f}% uplift)")
        print(f"Expected Profit:    ${res['optimal_metrics']['expected_profit']:,.2f} ({res['optimal_metrics']['profit_uplift_pct']:+.1f}% uplift)")
        print(f"Profit Margin:      {res['optimal_metrics']['profit_margin_pct']:.1f}%")
        print(f"Price Elasticity:   {res['elasticity_info'].get('own_price_elasticity')} ({res['elasticity_info'].get('regime')})")
        print("="*50 + "\n")

    elif args.command == "predict":
        if not MODEL_CHECKPOINT_FILE.exists():
            ModelTrainer().train_pipeline(force_regenerate=False)

        artifact = joblib.load(MODEL_CHECKPOINT_FILE)
        elast = joblib.load(ELASTICITY_MODEL_FILE)
        optimizer = PriceOptimizer(demand_model=artifact["model"], transformer=artifact["transformer"], elasticity_model=elast)

        input_data = {
            "product_id": args.product_id,
            "category": "Electronics",
            "price": args.price,
            "base_price": args.price,
            "cost_price": args.cost,
            "competitor_price": args.price * 1.05,
            "stock_level": 300,
            "discount_percent": 0.0,
            "rating": 4.5,
            "review_count": 350,
            "ad_spend_usd": 150.0,
            "holiday_effect": 0.0,
            "customer_segment": "Standard",
            "season": "Fall",
            "month": 10,
            "day_of_week": 2,
            "is_weekend": 0,
            "historical_demand_7d": 45.0,
            "historical_demand_30d": 42.0
        }

        pred_demand = optimizer.predict_demand_at_price(input_data, args.price)
        rev = args.price * pred_demand
        prof = (args.price - args.cost) * pred_demand

        print(f"Forecasted Demand at ${args.price:.2f}: {pred_demand:.1f} units | Revenue: ${rev:,.2f} | Profit: ${prof:,.2f}")

    elif args.command == "evaluate":
        if METRICS_FILE.exists():
            with open(METRICS_FILE, "r") as f:
                metrics = json.load(f)
            print(json.dumps(metrics, indent=2))
        else:
            print("No metrics file found. Run 'python cli.py train' first.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
