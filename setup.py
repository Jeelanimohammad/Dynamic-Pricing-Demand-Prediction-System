from setuptools import setup, find_packages

setup(
    name="dynamic_pricing_system",
    version="1.0.0",
    description="Dynamic Pricing & Demand Prediction System using Machine Learning and Constrained Optimization",
    author="Jeelani Mohammad",
    author_email="jeelanimohammad@users.noreply.github.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "streamlit>=1.30.0",
        "plotly>=5.18.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "openpyxl>=3.1.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "dynamic-pricing=cli:main",
        ],
    },
)
