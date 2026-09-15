"""
Dynamic Pricing & Demand Prediction System - Interactive Streamlit Dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sqlite3
from pathlib import Path
import sys
import os

# Add root directory to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    MODEL_CHECKPOINT_FILE,
    ELASTICITY_MODEL_FILE,
    SQLITE_DB_PATH,
    RAW_DATA_FILE,
    METRICS_FILE
)
from src.data.generator import CATALOG_DEFINITIONS, MarketDataGenerator
from src.models.train import ModelTrainer
from src.pricing.optimizer import PriceOptimizer
from src.pricing.simulator import PricingSimulator
from src.pricing.strategies import PricingStrategyEngine
from src.utils.metrics import calculate_financial_metrics

# Page configuration
st.set_page_config(
    page_title="Dynamic Pricing & Demand Prediction AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1e2f 0%, #111119 100%);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    
    .metric-delta-pos {
        font-size: 0.85rem;
        color: #10b981;
        font-weight: 600;
    }
    
    .metric-delta-neg {
        font-size: 0.85rem;
        color: #ef4444;
        font-weight: 600;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-blue { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-green { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-purple { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_system_pipeline():
    """Initializes and caches model artifacts and database connections."""
    if not MODEL_CHECKPOINT_FILE.exists() or not ELASTICITY_MODEL_FILE.exists():
        with st.spinner("⚡ Initializing market simulation and training ML models..."):
            trainer = ModelTrainer()
            trainer.train_pipeline(force_regenerate=True)

    model_artifact = joblib.load(MODEL_CHECKPOINT_FILE)
    elasticity_model = joblib.load(ELASTICITY_MODEL_FILE)

    demand_model = model_artifact["model"]
    transformer = model_artifact["transformer"]

    optimizer = PriceOptimizer(
        demand_model=demand_model,
        transformer=transformer,
        elasticity_model=elasticity_model
    )
    simulator = PricingSimulator(optimizer=optimizer)

    # Load historical sales data
    df_sales = pd.read_csv(RAW_DATA_FILE)
    df_sales["date"] = pd.to_datetime(df_sales["date"])

    return {
        "model_artifact": model_artifact,
        "elasticity_model": elasticity_model,
        "optimizer": optimizer,
        "simulator": simulator,
        "df_sales": df_sales
    }


pipeline = load_system_pipeline()
optimizer: PriceOptimizer = pipeline["optimizer"]
simulator: PricingSimulator = pipeline["simulator"]
elasticity_model = pipeline["elasticity_model"]
df_sales: pd.DataFrame = pipeline["df_sales"]

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/combo-chart.png", width=64)
    st.title("Dynamic Pricing AI")
    st.caption("Machine Learning & Demand Elasticity Engine")
    st.divider()

    menu = st.radio(
        "Navigation",
        [
            "⚡ Real-Time Price Optimizer",
            "📊 Executive Performance & Trends",
            "🔮 Demand Forecasting & What-If",
            "🧪 A/B Testing Cohort Simulator",
            "🧠 Model Diagnostics & Elasticity",
            "💾 SQL Analytics & Data Export"
        ]
    )

    st.divider()
    st.markdown("### System Status")
    st.success("🟢 ML Pipeline Active (Ensemble)")
    st.info(f"📦 Products in Catalog: **{len(CATALOG_DEFINITIONS)}**")
    st.caption("Tech Stack: Scikit-Learn • LightGBM/Ensemble • FastAPI • Streamlit • SQLite • Power BI")

# Header
st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700; color: #f8fafc;">
                ⚡ Dynamic Pricing & Demand Prediction System
            </h1>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                Autonomous revenue & profit maximization using price elasticity curves and gradient boosted ensemble demand forecasting.
            </p>
        </div>
        <div>
            <span class="badge-pill badge-blue">Ensemble ML</span>
            <span class="badge-pill badge-green">SciPy Optimizer</span>
            <span class="badge-pill badge-purple">Log-Log PED</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 1. REAL-TIME PRICE OPTIMIZER
# ==========================================
if menu == "⚡ Real-Time Price Optimizer":
    st.subheader("🎯 Real-Time Price Recommendation & Curve Visualizer")
    st.caption("Adjust market conditions to evaluate revenue/profit curves and calculate mathematical optimum.")

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown("#### ⚙️ Product & Market Inputs")
        
        # Product Selector
        prod_options = {p["product_id"]: f"{p['name']} ({p['category']})" for p in CATALOG_DEFINITIONS}
        selected_prod_id = st.selectbox(
            "Select Product",
            options=list(prod_options.keys()),
            format_func=lambda x: prod_options[x]
        )
        prod_meta = next(p for p in CATALOG_DEFINITIONS if p["product_id"] == selected_prod_id)

        # Interactive Controls
        current_price = st.number_input("Current Selling Price ($)", value=float(prod_meta["base_price"]), step=5.0)
        cost_price = st.number_input("Unit Cost ($)", value=float(prod_meta["base_cost"]), step=2.0)
        competitor_price = st.number_input("Competitor Price ($)", value=round(float(prod_meta["base_price"]) * 1.02, 2), step=5.0)
        stock_level = st.slider("Current Inventory Stock", min_value=10, max_value=1500, value=350, step=10)
        ad_spend = st.slider("Daily Ad Spend ($)", min_value=0.0, max_value=1000.0, value=150.0, step=25.0)
        
        with st.expander("Advanced Market Conditions"):
            rating = st.slider("Product Rating (1-5)", min_value=3.5, max_value=5.0, value=4.5, step=0.1)
            review_count = st.number_input("Review Count", value=450, step=50)
            customer_segment = st.selectbox("Target Segment", ["Standard", "Budget", "Premium", "Enterprise"])
            season = st.selectbox("Season", ["Fall", "Winter", "Spring", "Summer"])
            holiday_effect = st.slider("Holiday / Surge Multiplier", 0.0, 1.0, 0.0, 0.1)
            optimization_goal = st.selectbox("Optimization Goal", ["profit", "revenue", "balanced"])
            min_margin = st.slider("Minimum Profit Margin Guardrail (%)", 5, 50, 15) / 100.0
            max_shift = st.slider("Max Allowed Price Shift (± %)", 10, 60, 30) / 100.0

    # Build feature dict
    input_features = {
        "product_id": selected_prod_id,
        "product_name": prod_meta["name"],
        "category": prod_meta["category"],
        "price": current_price,
        "base_price": prod_meta["base_price"],
        "cost_price": cost_price,
        "competitor_price": competitor_price,
        "stock_level": stock_level,
        "discount_percent": 0.0,
        "rating": rating,
        "review_count": review_count,
        "ad_spend_usd": ad_spend,
        "holiday_effect": holiday_effect,
        "customer_segment": customer_segment,
        "season": season,
        "month": 10,
        "day_of_week": 2,
        "is_weekend": 0,
        "historical_demand_7d": float(prod_meta["base_daily_volume"]),
        "historical_demand_30d": float(prod_meta["base_daily_volume"])
    }

    # Run Optimization
    opt_result = optimizer.optimize_price(
        product_features=input_features,
        objective=optimization_goal,
        min_margin_pct=min_margin,
        max_price_change=max_shift
    )

    with col_right:
        st.markdown("#### 💡 Optimal Pricing Recommendation")

        # KPI Metrics Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        opt_price = opt_result["optimal_price"]
        price_change = opt_result["price_change_pct"]
        rev_uplift = opt_result["optimal_metrics"]["revenue_uplift_pct"]
        prof_uplift = opt_result["optimal_metrics"]["profit_uplift_pct"]

        with kpi1:
            delta_class = "metric-delta-pos" if price_change >= 0 else "metric-delta-neg"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recommended Price</div>
                <div class="metric-value">${opt_price:.2f}</div>
                <div class="{delta_class}">{price_change:+.1f}% vs Current (${current_price:.2f})</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Forecasted Demand</div>
                <div class="metric-value">{opt_result['optimal_metrics']['predicted_demand']:.0f} units</div>
                <div class="metric-delta-pos">Base: {opt_result['baseline_metrics']['predicted_demand']:.0f} units</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            delta_class = "metric-delta-pos" if rev_uplift >= 0 else "metric-delta-neg"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Expected Daily Revenue</div>
                <div class="metric-value">${opt_result['optimal_metrics']['expected_revenue']:,.2f}</div>
                <div class="{delta_class}">{rev_uplift:+.1f}% Uplift</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            delta_class = "metric-delta-pos" if prof_uplift >= 0 else "metric-delta-neg"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Expected Daily Profit</div>
                <div class="metric-value">${opt_result['optimal_metrics']['expected_profit']:,.2f}</div>
                <div class="{delta_class}">{prof_uplift:+.1f}% Uplift</div>
            </div>
            """, unsafe_allow_html=True)

        # Plotly Price vs Demand & Profit Curves
        curve = opt_result["price_curve"]
        fig = go.Figure()

        # Profit Curve
        fig.add_trace(go.Scatter(
            x=curve["prices"],
            y=curve["profits"],
            mode="lines",
            name="Expected Profit ($)",
            line=dict(color="#10b981", width=3)
        ))

        # Revenue Curve
        fig.add_trace(go.Scatter(
            x=curve["prices"],
            y=curve["revenues"],
            mode="lines",
            name="Expected Revenue ($)",
            line=dict(color="#6366f1", width=2, dash="dot")
        ))

        # Demand Curve (Secondary Y Axis)
        fig.add_trace(go.Scatter(
            x=curve["prices"],
            y=curve["demands"],
            mode="lines",
            name="Forecasted Demand (Units)",
            line=dict(color="#f59e0b", width=2),
            yaxis="y2"
        ))

        # Add vertical line at optimal price
        fig.add_vline(
            x=opt_price,
            line_dash="dash",
            line_color="#ec4899",
            annotation_text=f"Optimal: ${opt_price}",
            annotation_position="top left"
        )

        fig.update_layout(
            title="Price Elasticity & Financial Simulation Curve",
            xaxis=dict(title="Selling Price ($)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Financial Value ($)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis2=dict(
                title="Units Demanded",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            height=380
        )

        st.plotly_chart(fig, use_container_width=True)

        # Elasticity Regime Banner
        elast_info = opt_result["elasticity_info"]
        st.info(
            f"📈 **Price Elasticity of Demand (PED):** `{elast_info.get('own_price_elasticity', -1.5)}` "
            f"| **Regime:** `{elast_info.get('regime', 'Elastic')}` "
            f"| **Cross-Price Elasticity with Competitor:** `{elast_info.get('cross_price_elasticity', 0.65)}`"
        )


# ==========================================
# 2. EXECUTIVE PERFORMANCE & TRENDS
# ==========================================
elif menu == "📊 Executive Performance & Trends":
    st.subheader("📊 Executive Business Dashboard & Sales Analytics")
    st.caption("Historical performance breakdown across categories, revenues, and pricing trends.")

    # High-level KPIs
    total_rev = df_sales["revenue"].sum()
    total_profit = df_sales["profit"].sum()
    total_units = df_sales["demand_units"].sum()
    avg_margin = (total_profit / total_rev) * 100

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Gross Revenue</div>
            <div class="metric-value">${total_rev:,.0f}</div>
            <div class="metric-delta-pos">Across 365 Days</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Net Profit</div>
            <div class="metric-value">${total_profit:,.0f}</div>
            <div class="metric-delta-pos">Gross Margin: {avg_margin:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Units Sold</div>
            <div class="metric-value">{total_units:,} units</div>
            <div class="metric-delta-pos">13 Catalog SKUs</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Daily Revenue</div>
            <div class="metric-value">${total_rev/365:,.0f}/day</div>
            <div class="metric-delta-pos">Peak Season: Q4</div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Category Revenue & Profit Breakdown
        cat_agg = df_sales.groupby("category").agg({"revenue": "sum", "profit": "sum"}).reset_index()
        fig_bar = px.bar(
            cat_agg,
            x="category",
            y=["revenue", "profit"],
            barmode="group",
            title="Revenue & Profit by Product Category",
            labels={"value": "USD ($)", "category": "Category", "variable": "Metric"},
            color_discrete_map={"revenue": "#6366f1", "profit": "#10b981"},
            template="plotly_dark"
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Time Series Revenue Trend
        daily_trend = df_sales.groupby("date").agg({"revenue": "sum", "demand_units": "sum"}).reset_index()
        fig_line = px.line(
            daily_trend,
            x="date",
            y="revenue",
            title="Daily Total Revenue Trend (365 Days)",
            labels={"revenue": "Revenue ($)", "date": "Date"},
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8"]
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # Product Performance Table
    st.markdown("#### 📋 SKU Performance & Unit Economics")
    sku_perf = df_sales.groupby(["product_id", "product_name", "category"]).agg({
        "demand_units": "sum",
        "revenue": "sum",
        "profit": "sum",
        "price": "mean",
        "cost_price": "mean",
        "discount_percent": "mean"
    }).round(2).reset_index()

    sku_perf["profit_margin_pct"] = np.round((sku_perf["profit"] / sku_perf["revenue"]) * 100, 2)
    sku_perf = sku_perf.sort_values(by="revenue", ascending=False)
    st.dataframe(sku_perf, use_container_width=True)


# ==========================================
# 3. DEMAND FORECASTING & WHAT-IF
# ==========================================
elif menu == "🔮 Demand Forecasting & What-If":
    st.subheader("🔮 Demand Forecasting & What-If Scenario Matrix")
    st.caption("Simulate how macroeconomic shifts, competitor pricing wars, and marketing adjustments impact demand.")

    prod_options = {p["product_id"]: f"{p['name']} ({p['category']})" for p in CATALOG_DEFINITIONS}
    selected_prod_id = st.selectbox("Select Product for Scenario Analysis", list(prod_options.keys()), format_func=lambda x: prod_options[x])
    prod_meta = next(p for p in CATALOG_DEFINITIONS if p["product_id"] == selected_prod_id)

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown("#### 🎛️ Scenario Parameters")
        p_delta = st.slider("Price Shift (± %)", -30.0, 30.0, 0.0, 2.5)
        comp_delta = st.slider("Competitor Price Shift (± %)", -30.0, 30.0, 0.0, 2.5)
        ad_mult = st.slider("Ad Spend Multiplier", 0.2, 3.0, 1.0, 0.1)
        holiday_val = st.slider("Holiday / Event Surge Level", 0.0, 1.0, 0.0, 0.1)

    base_f = {
        "product_id": selected_prod_id,
        "category": prod_meta["category"],
        "price": float(prod_meta["base_price"]),
        "base_price": float(prod_meta["base_price"]),
        "cost_price": float(prod_meta["base_cost"]),
        "competitor_price": float(prod_meta["base_price"]),
        "stock_level": 300,
        "discount_percent": 0.0,
        "rating": 4.5,
        "review_count": 400,
        "ad_spend_usd": 150.0,
        "holiday_effect": 0.0,
        "customer_segment": "Standard",
        "season": "Fall",
        "month": 10,
        "day_of_week": 2,
        "is_weekend": 0,
        "historical_demand_7d": float(prod_meta["base_daily_volume"]),
        "historical_demand_30d": float(prod_meta["base_daily_volume"])
    }

    scenario_res = simulator.run_what_if_scenario(
        base_features=base_f,
        price_delta_pct=p_delta,
        competitor_delta_pct=comp_delta,
        ad_spend_multiplier=ad_mult,
        holiday_effect=holiday_val
    )

    with col2:
        st.markdown("#### 📊 Projected Outcome")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Simulated Price", f"${scenario_res['simulated_price']:.2f}", f"{p_delta:+.1f}%")
        with c2:
            st.metric("Forecasted Demand", f"{scenario_res['simulated_demand']:.1f} units")
        with c3:
            st.metric("Expected Revenue", f"${scenario_res['expected_revenue']:,.2f}")

        # Multi-scenario sensitivity table
        st.markdown("#### 📈 Price vs Competitor Sensitivity Matrix")
        price_variations = [-15, -10, -5, 0, 5, 10, 15]
        comp_variations = [-15, -10, 0, 10, 15]

        matrix_records = []
        for p_v in price_variations:
            row = {"Price Shift (%)": f"{p_v:+d}%"}
            for c_v in comp_variations:
                sim = simulator.run_what_if_scenario(
                    base_features=base_f,
                    price_delta_pct=p_v,
                    competitor_delta_pct=c_v,
                    ad_spend_multiplier=ad_mult,
                    holiday_effect=holiday_val
                )
                row[f"Comp {c_v:+d}% ($Rev)"] = f"${sim['expected_revenue']:,.0f}"
            matrix_records.append(row)

        st.dataframe(pd.DataFrame(matrix_records), use_container_width=True)


# ==========================================
# 4. A/B TESTING COHORT SIMULATOR
# ==========================================
elif menu == "🧪 A/B Testing Cohort Simulator":
    st.subheader("🧪 Dynamic Pricing vs. Static Pricing A/B Test Simulator")
    st.caption("Monte Carlo randomized trial simulation with statistical hypothesis testing (t-test & p-value).")

    prod_options = {p["product_id"]: f"{p['name']} ({p['category']})" for p in CATALOG_DEFINITIONS}
    selected_prod_id = st.selectbox("Select Product for A/B Experiment", list(prod_options.keys()), format_func=lambda x: prod_options[x])
    prod_meta = next(p for p in CATALOG_DEFINITIONS if p["product_id"] == selected_prod_id)

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown("#### ⚙️ Experiment Setup")
        ctrl_p = st.number_input("Variant A (Control - Static Price $)", value=float(prod_meta["base_price"]))
        treat_p = st.number_input("Variant B (Treatment - Dynamic Price $)", value=round(float(prod_meta["base_price"]) * 1.10, 2))
        n_sample = st.slider("Traffic Impressions per Variant", 500, 20000, 3000, 500)
        run_sim = st.button("🚀 Run A/B Simulation", type="primary", use_container_width=True)

    base_f = {
        "product_id": selected_prod_id,
        "category": prod_meta["category"],
        "price": float(prod_meta["base_price"]),
        "cost_price": float(prod_meta["base_cost"]),
        "competitor_price": float(prod_meta["base_price"]),
        "stock_level": 500,
        "discount_percent": 0.0,
        "rating": 4.5,
        "review_count": 400,
        "ad_spend_usd": 150.0,
        "holiday_effect": 0.0,
        "customer_segment": "Standard",
        "season": "Fall",
        "month": 10,
        "day_of_week": 2,
        "is_weekend": 0,
        "historical_demand_7d": float(prod_meta["base_daily_volume"]),
        "historical_demand_30d": float(prod_meta["base_daily_volume"])
    }

    ab_res = simulator.simulate_ab_test(
        product_features=base_f,
        control_price=ctrl_p,
        treatment_price=treat_p,
        sample_size=n_sample
    )

    with col2:
        st.markdown("#### 📊 Experiment Results & Statistical Significance")
        
        ca, cb = st.columns(2)
        ctrl = ab_res["control_group_A"]
        treat = ab_res["treatment_group_B"]
        comp = ab_res["comparison"]

        with ca:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Variant A (Control) @ ${ctrl['price']}</div>
                <div class="metric-value">${ctrl['total_revenue']:,.2f}</div>
                <div>Conv. Rate: <b>{ctrl['conversion_rate_pct']}%</b> ({ctrl['conversions']} sales)</div>
                <div>Profit: <b>${ctrl['total_profit']:,.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with cb:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Variant B (Treatment) @ ${treat['price']}</div>
                <div class="metric-value">${treat['total_revenue']:,.2f}</div>
                <div>Conv. Rate: <b>{treat['conversion_rate_pct']}%</b> ({treat['conversions']} sales)</div>
                <div>Profit: <b>${treat['total_profit']:,.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)

        # Statistical Banner
        is_sig = comp["is_statistically_significant"]
        if is_sig:
            st.success(
                f"🎉 **Statistically Significant Uplift!** "
                f"| Revenue Uplift: **+{comp['revenue_uplift_percent']}%** "
                f"| Profit Uplift: **+{comp['profit_uplift_percent']}%** "
                f"| p-value: `{comp['p_value']}` (p < 0.05)"
            )
        else:
            st.warning(
                f"⚠️ **Not Statistically Significant yet.** "
                f"| Revenue Uplift: **{comp['revenue_uplift_percent']}%** "
                f"| p-value: `{comp['p_value']}` (Need larger sample size)"
            )


# ==========================================
# 5. MODEL DIAGNOSTICS & ELASTICITY
# ==========================================
elif menu == "🧠 Model Diagnostics & Elasticity":
    st.subheader("🧠 Machine Learning Model Performance & Diagnostics")
    st.caption("Cross-validated metrics across Random Forest, Gradient Boosting, ElasticNet, and Ensemble.")

    import json
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as f:
            metrics_data = json.load(f)
    else:
        metrics_data = {}

    models_dict = metrics_data.get("model_metrics", {})
    if models_dict:
        metrics_df = pd.DataFrame(models_dict).T.reset_index()
        metrics_df.columns = ["Model", "RMSE", "MAE", "MAPE (%)", "WAPE (%)", "R² Score"]
        st.dataframe(metrics_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌟 Top Feature Importance (SHAP / Tree Impurity)")
        top_feats = metrics_data.get("top_features", {})
        if top_feats:
            df_feat = pd.DataFrame(list(top_feats.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
            fig_imp = px.bar(
                df_feat,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Predictive Features for Demand",
                template="plotly_dark",
                color="Importance",
                color_continuous_scale="Blues"
            )
            fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        st.markdown("#### 📐 Econometric Price Elasticity by Product")
        elast_records = []
        for p in CATALOG_DEFINITIONS:
            pid = p["product_id"]
            p_elast = elasticity_model.get_product_elasticity(pid)
            elast_records.append({
                "Product": p["name"][:25] + "...",
                "Category": p["category"],
                "Own PED": p_elast.get("own_price_elasticity", -1.5),
                "Cross PED": p_elast.get("cross_price_elasticity", 0.65),
                "Regime": p_elast.get("regime", "Elastic")
            })
        st.dataframe(pd.DataFrame(elast_records), use_container_width=True)


# ==========================================
# 6. SQL ANALYTICS & DATA EXPORT
# ==========================================
elif menu == "💾 SQL Analytics & Data Export":
    st.subheader("💾 SQL Database Querying & Power BI / Excel Export")
    st.caption("Execute direct SQL queries on historical sales and export datasets for Power BI and Excel.")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("#### 🔍 Interactive SQL Query Console")
        default_query = """SELECT category, 
       COUNT(DISTINCT product_id) as active_skus,
       ROUND(SUM(revenue), 2) as total_revenue,
       ROUND(SUM(profit), 2) as total_profit,
       ROUND(AVG(price), 2) as avg_price,
       ROUND(AVG(discount_percent), 2) as avg_discount_pct
FROM sales_transactions
GROUP BY category
ORDER BY total_revenue DESC;"""

        sql_input = st.text_area("Write SQL Query", value=default_query, height=160)
        
        if st.button("▶️ Execute Query", type="primary"):
            try:
                conn = sqlite3.connect(SQLITE_DB_PATH)
                query_result = pd.read_sql_query(sql_input, conn)
                conn.close()
                st.success(f"Returned {len(query_result)} rows.")
                st.dataframe(query_result, use_container_width=True)
            except Exception as e:
                st.error(f"SQL Error: {e}")

    with col2:
        st.markdown("#### 📤 Power BI & Excel Exports")
        st.info("Ready-to-use exported files formatted for Power BI, Tableau, and Excel reporting.")

        # Download CSV
        csv_bytes = df_sales.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned Sales CSV",
            data=csv_bytes,
            file_name="market_sales_cleaned.csv",
            mime="text/csv",
            use_container_width=True
        )

        excel_path = SQLITE_DB_PATH.parent / "power_bi_sales_data.xlsx"
        if excel_path.exists():
            with open(excel_path, "rb") as f:
                st.download_button(
                    label="📊 Download Power BI Excel Workbook",
                    data=f.read(),
                    file_name="power_bi_sales_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
