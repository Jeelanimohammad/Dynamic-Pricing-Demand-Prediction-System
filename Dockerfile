# Multi-stage production container
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Generate training data and compile initial model artifacts
RUN python -c "from src.models.train import ModelTrainer; ModelTrainer().train_pipeline(force_regenerate=True)"

# Expose Streamlit & FastAPI ports
EXPOSE 8501 8000

# Default command launches both FastAPI and Streamlit via startup script or Streamlit dashboard
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
