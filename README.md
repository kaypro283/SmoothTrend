# SmoothTrend

SmoothTrend is a comprehensive time series analysis tool that utilizes Holt and Simple Exponential Smoothing methods to perform advanced trend analysis, stationarity testing, residual analysis, and forecasting.

## Features

- **Trend Detection**: Detects trends using Mann-Kendall and Cox-Stuart tests.
- **Stationarity Testing**: Tests for stationarity with the Augmented Dickey-Fuller (ADF) test.
- **Exponential Smoothing**: Fits Holt and Simple Exponential Smoothing models to the data and forecasts future values.
- **Residual Analysis**: Performs detailed residual analysis including Ljung-Box, Shapiro-Wilk, and Kolmogorov-Smirnov tests.
- **Heteroscedasticity Tests**: Includes Breusch-Pagan, White, and Spearman's rank correlation tests.
- **Descriptive Statistics**: Computes mean, median, standard deviation, skewness, kurtosis, and other descriptive statistics.
- **Data Visualization**: Plots decomposition plots, residual plots, and Autocorrelation Function (ACF) plots.
- **Forecasting**: Provides forecasts with prediction intervals and error statistics (MSE, MAE, RMSE, MAPE).
- **Spectral Analysis**: Performs spectral analysis on the data.

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
