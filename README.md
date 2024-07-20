# SmoothTrend: Holt-Winters, Holt, Simple Exponential Smoothing, ARIMA/SARIMA and Trend Analysis Program v2.03

![Version](https://img.shields.io/badge/version-2.03-blue.svg)
![License](https://img.shields.io/badge/license-GPL--2.0-green.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)

SmoothTrend is a Python-based time series analysis program that implements several statistical methods for trend analysis, forecasting, and data visualization. The script incorporates Holt-Winters, Holt, and Simple Exponential Smoothing techniques, as well as ARIMA/SARIMA modeling, to analyze time series data.

The program guides users through the analysis process from data input to forecasting. It accepts data through manual entry or CSV file import. SmoothTrend conducts various statistical tests to identify trends, evaluate stationarity, and detect seasonality in the data.

A key feature of SmoothTrend is its ability to automatically select optimal parameters for each smoothing method. This functionality can be useful for users working with various types of time series data, including economic indicators, environmental patterns, or business metrics.


## Author
Christopher D. van der Kaay (2024)


## Features

- **Data Input**: Supports manual data entry or import from CSV files.
- **Trend Detection**: Utilizes Mann-Kendall and Cox-Stuart tests for trend identification.
- **Stationarity Testing**: Implements the Augmented Dickey-Fuller (ADF) test for stationarity analysis.
- **Seasonality Detection**: Uses Fast Fourier Transform (FFT) and spectral analysis to identify seasonal patterns.
- **Exponential Smoothing**: Fits Holt-Winters, Holt, and Simple Exponential Smoothing models to the data.
- **ARIMA/SARIMA Modeling**: Implements Autoregressive Integrated Moving Average (ARIMA) and Seasonal ARIMA (SARIMA) modeling with automatic order selection.
- **Residual Analysis**: Conducts thorough residual analysis using Ljung-Box, Shapiro-Wilk, and Kolmogorov-Smirnov tests.
- **Heteroscedasticity Tests**: Includes Breusch-Pagan, White, and Spearman's rank correlation tests.
- **Descriptive Statistics**: Calculates comprehensive statistical measures including mean, median, mode, standard deviation, skewness, and kurtosis.
- **Data Visualization**: Generates insightful plots including decomposition, residual, ACF plots, and interactive Plotly charts.
- **Forecasting**: Provides forecasts with prediction intervals and error statistics (MSE, MAE, RMSE, MAPE).
- **Spectral Analysis**: Performs spectral analysis on the time series data.
- **Optimal Parameter Selection**: Automatically selects optimal parameters for each smoothing method.
- **Debug Mode**: Offers detailed viewing of calculations for debugging purposes.
- **Lag Plot Analysis**: Checks for randomness in the time series data.
- **Interactive User Interface**: Guides users through the analysis process with prompts and options.
- **Error Handling**: Includes robust error checking and user-friendly error messages.
- **Parallel Processing**: Improves performance in parameter optimization.
- **Sound Alerts**: Provides sound alerts for warnings and errors.


## Requirements
- Python 3.7+


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
Run the `time_series_analysis_202.py` script to start the program:
```bash
python time_series_analysis_202.py
