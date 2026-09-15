"""
Data preprocessing, encoding, and relational database / Power BI export pipelines.
"""
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    FEATURE_METADATA_FILE,
    SQLITE_DB_PATH,
    TARGET_COLUMN,
    RANDOM_SEED
)
from src.data.feature_engineering import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger("preprocessor")


class DataPreprocessor:
    """
    Cleans, transforms, and exports sales and pricing market data.
    """

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.preprocessor_pipeline: Optional[ColumnTransformer] = None
        self.categorical_cols = ["category", "product_id", "customer_segment", "season"]
        self.numerical_cols = [
            "price",
            "cost_price",
            "competitor_price",
            "historical_demand_7d",
            "historical_demand_30d",
            "stock_level",
            "discount_percent",
            "price_ratio_competitor",
            "price_diff_competitor",
            "price_cost_margin",
            "discount_ratio",
            "stock_to_demand_ratio",
            "low_stock_flag",
            "rating",
            "review_count",
            "ad_spend_usd",
            "ad_spend_log",
            "rating_weighted_reviews",
            "holiday_effect",
            "month_sin",
            "month_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
        ]

    def build_transformer(self) -> ColumnTransformer:
        """
        Constructs standard ColumnTransformer pipeline for machine learning.
        """
        num_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        transformer = ColumnTransformer(
            transformers=[
                ("num", num_transformer, self.numerical_cols),
                ("cat", cat_transformer, self.categorical_cols)
            ],
            remainder="drop"
        )
        return transformer

    def process_and_split(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size_days: int = 60
    ) -> Dict[str, Any]:
        """
        Executes feature engineering, splits chronologically, fits transformations,
        and saves processed datasets and SQL / Excel databases.
        """
        if df is None:
            if not RAW_DATA_FILE.exists():
                raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_FILE}. Run generator first.")
            df = pd.read_csv(RAW_DATA_FILE)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["date", "product_id"]).reset_index(drop=True)

        # 1. Feature Engineering
        df_featured = self.feature_engineer.create_features(df)

        # 2. Time-series split (Train on earlier dates, test on last test_size_days)
        max_date = df_featured["date"].max()
        split_date = max_date - pd.Timedelta(days=test_size_days)

        train_df = df_featured[df_featured["date"] <= split_date].copy()
        test_df = df_featured[df_featured["date"] > split_date].copy()

        logger.info(f"Split data: Train rows={len(train_df)} (<= {split_date.date()}), Test rows={len(test_df)} (> {split_date.date()})")

        # 3. Fit ColumnTransformer on Train and Transform
        self.preprocessor_pipeline = self.build_transformer()
        X_train_transformed = self.preprocessor_pipeline.fit_transform(train_df)
        X_test_transformed = self.preprocessor_pipeline.transform(test_df)

        y_train = train_df[TARGET_COLUMN].values
        y_test = test_df[TARGET_COLUMN].values

        # 4. Export to SQLite database & Excel for Power BI / SQL analysis
        self.export_to_sqlite(df_featured)
        self.export_to_excel(df_featured)

        # 5. Save metadata
        feature_names = self.get_transformed_feature_names()
        metadata = {
            "num_features": self.numerical_cols,
            "cat_features": self.categorical_cols,
            "transformed_feature_names": feature_names,
            "target_column": TARGET_COLUMN,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "split_date": str(split_date.date())
        }

        with open(FEATURE_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)

        return {
            "train_df": train_df,
            "test_df": test_df,
            "X_train": X_train_transformed,
            "y_train": y_train,
            "X_test": X_test_transformed,
            "y_test": y_test,
            "transformer": self.preprocessor_pipeline,
            "feature_names": feature_names,
            "df_featured": df_featured
        }

    def get_transformed_feature_names(self) -> List[str]:
        """
        Retrieves formatted names of all columns output by the pipeline.
        """
        if self.preprocessor_pipeline is None:
            return self.numerical_cols + self.categorical_cols
        
        feature_names = []
        # Numerical
        feature_names.extend(self.numerical_cols)
        # Categorical One-Hot
        cat_encoder = self.preprocessor_pipeline.named_transformers_["cat"].named_steps["onehot"]
        cat_names = cat_encoder.get_feature_names_out(self.categorical_cols)
        feature_names.extend(list(cat_names))
        return feature_names

    def export_to_sqlite(self, df: pd.DataFrame) -> None:
        """
        Creates SQLite database for business SQL queries and analysis.
        """
        try:
            SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(SQLITE_DB_PATH)
            
            # Export main sales table
            df_export = df.copy()
            df_export["date"] = df_export["date"].astype(str)
            df_export.to_sql("sales_transactions", conn, if_exists="replace", index=False)

            # Export product catalog dimension table
            dim_products = df_export[[
                "product_id", "product_name", "category", "base_price", "cost_price"
            ]].drop_duplicates().reset_index(drop=True)
            dim_products.to_sql("dim_products", conn, if_exists="replace", index=False)

            # Export daily aggregate summary
            daily_agg = df_export.groupby(["date", "category"]).agg({
                "demand_units": "sum",
                "revenue": "sum",
                "profit": "sum",
                "price": "mean",
                "discount_percent": "mean"
            }).reset_index()
            daily_agg.to_sql("daily_category_metrics", conn, if_exists="replace", index=False)

            conn.close()
            logger.info(f"Exported relational tables to SQLite DB at {SQLITE_DB_PATH}")
        except Exception as e:
            logger.warning(f"Failed to export SQLite tables: {e}")

    def export_to_excel(self, df: pd.DataFrame) -> None:
        """
        Exports clean multi-sheet Excel workbook for Power BI and Excel dashboards.
        """
        try:
            excel_path = SQLITE_DB_PATH.parent / "power_bi_sales_data.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                # Summary Sheet
                summary_df = df.groupby(["category", "product_name"]).agg({
                    "demand_units": "sum",
                    "revenue": "sum",
                    "profit": "sum",
                    "price": "mean",
                    "cost_price": "mean",
                    "discount_percent": "mean"
                }).round(2).reset_index()
                summary_df.to_excel(writer, sheet_name="Product_Performance", index=False)

                # Monthly Category Trend Sheet
                df_copy = df.copy()
                df_copy["month_year"] = pd.to_datetime(df_copy["date"]).dt.to_period("M").astype(str)
                trend_df = df_copy.groupby(["month_year", "category"]).agg({
                    "revenue": "sum",
                    "profit": "sum",
                    "demand_units": "sum",
                    "price": "mean"
                }).round(2).reset_index()
                trend_df.to_excel(writer, sheet_name="Monthly_Trends", index=False)

                # Recent Sample Sheet
                df_sample = df.tail(1000).copy()
                df_sample["date"] = df_sample["date"].astype(str)
                df_sample.to_excel(writer, sheet_name="Recent_Transactions", index=False)

            logger.info(f"Exported Power BI / Excel workbook to {excel_path}")
        except Exception as e:
            logger.warning(f"Failed to export Excel workbook: {e}")
