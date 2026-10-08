@echo off
title AI Gender Equality Analyzer
echo ============================================================
echo   AI-Based Gender Equality and Workplace Gap Analyzer
echo   IBM SkillsBuild Final Project
echo ============================================================
echo.
echo Starting Streamlit app on http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.
python -m streamlit run app.py --browser.gatherUsageStats false
pause
