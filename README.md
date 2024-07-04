# SmoothTrend v1.3

SmoothTrend is a comprehensive time series analysis tool that utilizes Holt-Winters, Holt, and Simple Exponential Smoothing methods, as well as ARIMA modeling, to perform advanced trend analysis, stationarity testing, residual analysis, and forecasting.

## Features

- **Data Input**: Supports manual data entry or import from CSV files.
- **Trend Detection**: Utilizes Mann-Kendall and Cox-Stuart tests for trend identification.
- **Stationarity Testing**: Implements the Augmented Dickey-Fuller (ADF) test for stationarity analysis.
- **Seasonality Detection**: Uses Fast Fourier Transform (FFT) and spectral analysis to identify seasonal patterns.
- **Exponential Smoothing**: Fits Holt-Winters, Holt, and Simple Exponential Smoothing models to the data.
- **ARIMA Modeling**: Implements Autoregressive Integrated Moving Average (ARIMA) modeling with automatic order selection.
- **Residual Analysis**: Conducts thorough residual analysis using Ljung-Box, Shapiro-Wilk, and Kolmogorov-Smirnov tests.
- **Heteroscedasticity Tests**: Includes Breusch-Pagan, White, and Spearman's rank correlation tests.
- **Descriptive Statistics**: Calculates comprehensive statistical measures including mean, median, standard deviation, skewness, and kurtosis.
- **Data Visualization**: Generates insightful plots including decomposition, residual, ACF plots, and interactive Plotly charts.
- **Forecasting**: Provides forecasts with prediction intervals and error statistics (MSE, MAE, RMSE, MAPE).
- **Spectral Analysis**: Performs spectral analysis on the time series data.
- **Optimal Parameter Selection**: Automatically selects optimal parameters for each smoothing method.
- **Debug Mode**: Offers detailed viewing of calculations for debugging purposes.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/kaypro283/SmoothTrend.git
    ```
2. Navigate to the project directory:
    ```bash
    cd SmoothTrend
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the `time_series_analysis.py` script to start the program:

```bash
python time_series_analysis.py
