@echo off
title Dynamic Pricing API Service
echo ========================================================
echo  Starting Dynamic Pricing FastAPI REST Service
echo ========================================================
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
pause
