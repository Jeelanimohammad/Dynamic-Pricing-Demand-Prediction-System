# Dynamic Pricing & Demand Prediction System

A data-driven machine learning and dynamic pricing optimization system designed to analyze historical market transactions, estimate price elasticity of demand, forecast product demand, and recommend optimal price points for maximizing revenue and profit under real-world business constraints.

---

## Project Overview

In modern retail, e-commerce, hospitality, and on-demand mobility, static pricing fails to capture dynamic market shifts, competitor pricing strategies, seasonality, and inventory fluctuations. This system solves these challenges by combining econometric modeling, ensemble machine learning, and constrained numerical optimization to automate intelligent pricing decisions.

### Key Objectives & Core Capabilities
- **Historical Sales & Demand Analysis:** Ingests and processes multi-category sales transaction data to discover demand patterns, cyclical trends, and seasonality effects.
- **Price Elasticity of Demand (PED) Modeling:** Calculates point and arc price elasticity using log-log regressions to quantify consumer sensitivity and identify elastic vs. inelastic product regimes.
- **Ensemble Demand Forecasting:** Accurately forecasts future product demand volume based on applied price, competitor pricing, marketing spend, rating scores, and temporal attributes.
- **Constrained Price Optimization:** Determines mathematical price points that maximize gross revenue or net profit while strictly enforcing business guardrails (e.g., minimum margin thresholds and maximum price jump caps).
- **Interactive Business Intelligence & Simulation:** Provides real-time elasticity curves, what-if scenario sensitivity matrices, randomized A/B testing simulation with hypothesis testing ($p$-values), and direct SQL / Power BI export integration.

---

## System Architecture & Workflow

1. **Data Ingestion & Feature Engineering:**
   - Cleans historical transactions and computes economic and temporal features (7-day/30-day demand averages, competitor price ratios, unit margins, stockout scarcity indicators, cyclical date encodings).
   - Exports data to relational SQLite database tables (`sales_transactions`, `dim_products`, `daily_category_metrics`) and multi-tab Excel workbooks formatted for Power BI.

2. **Econometric Elasticity Modeling:**
   - Fits log-log demand response models:
     $$\ln(\text{Demand}) = \beta_0 + \beta_1 \ln(\text{Price}) + \beta_2 \ln(\text{Competitor Price}) + \mathbf{\beta} \mathbf{X} + \epsilon$$
   - Extracts own-price elasticity $\beta_1$ and cross-price elasticity $\beta_2$ per product and category.

3. **Demand Prediction Engine:**
   - Employs a weighted ensemble combining Gradient Boosting, Random Forest, and Regularized ElasticNet.
   - Evaluates performance using standard forecasting metrics including RMSE, MAE, MAPE, WAPE, and $R^2$.

4. **Dynamic Pricing Optimization & Heuristic Strategies:**
   - **Revenue / Profit Optimization:** Solves $\max_P (P - C) \cdot \hat{Q}(P; \mathbf{X})$ subject to margin floors and price fluctuation limits.
   - **Surge Pricing:** Automatically scales prices up during inventory scarcity or peak seasonal demand.
   - **Clearance Markdown:** Dynamically discounts aging or excess stock while guaranteeing a profit floor.
   - **Competitor Matching:** Anchors prices to undercut or match competitors while protecting margins.

5. **Visualization & Decision Support:**
   - **Interactive Web Application:** Built with Streamlit and Plotly for real-time pricing simulations, curve exploration, and executive KPI monitoring.
   - **SQL Console:** Built-in querying interface for ad-hoc business reporting.
   - **Power BI & Excel Integration:** Structured exports for downstream executive dashboards.

---

## Tech Stack

- **Programming Language:** Python
- **Data Processing & Feature Engineering:** Pandas, NumPy
- **Machine Learning & Modeling:** Scikit-learn, SciPy
- **Econometric Modeling:** Log-Log Linear Regressions, Point/Arc Price Elasticity
- **Database & Business Intelligence:** SQLite, SQL, Power BI, Excel (openpyxl)
- **Interactive Dashboard:** Streamlit, Plotly
- **API & Backend Integration:** FastAPI, Pydantic, Uvicorn