"""
Entrypoint launcher for Dynamic Pricing & Demand Prediction System.
"""
import sys
import subprocess

if __name__ == "__main__":
    print("🚀 Starting Dynamic Pricing & Demand Prediction System...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501"])
    except KeyboardInterrupt:
        print("\n👋 System stopped.")
