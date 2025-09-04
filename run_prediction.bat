@echo off
echo ========================================
echo AAPL Price Prediction with Chronos-Bolt
echo ========================================
echo.

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Testing installation...
python test_installation.py

echo.
echo Running AAPL prediction...
python aapl_prediction.py

echo.
echo Prediction completed! Check the generated files:
echo - aapl_prediction_results.png
echo - aapl_predictions_14days.csv
echo - aapl_historical_data.csv
echo.
pause
