# SmoothTrend: Holt-Winters, Holt, Simple Exponential Smoothing, ARIMA and Trend Analysis Program v1.4

![Version](https://img.shields.io/badge/version-1.4-blue.svg)
![License](https://img.shields.io/badge/license-GPL--2.0-green.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)

SmoothTrend is a Python-based time series analysis program that implements several statistical methods for trend analysis, forecasting, and data visualization. The script incorporates Holt-Winters, Holt, and Simple Exponential Smoothing techniques, as well as ARIMA modeling, to analyze time series data.

The program guides users through the analysis process from data input to forecasting. It accepts data through manual entry or CSV file import. SmoothTrend conducts various statistical tests to identify trends, evaluate stationarity, and detect seasonality in the data.

A key feature of SmoothTrend is its ability to automatically select optimal parameters for each smoothing method. This functionality can be useful for users working with various types of time series data, including economic indicators, environmental patterns, or business metrics.


## Author
C. van der Kaay (2024)


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
- **Lag Plot Analysis**: Checks for randomness in the time series data.
- **Interactive User Interface**: Guides users through the analysis process with prompts and options.
- **Error Handling**: Includes robust error checking and user-friendly error messages.


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
Run the `time_series_analysis_14.py` script to start the program:
```bash
python time_series_analysis_14.py
```

Follow the on-screen prompts to input your data and choose analysis options.


## Dependencies
- numpy
- pandas
- statsmodels
- scipy
- matplotlib
- plotly
- pmdarima
- colorama


## Contributing

I welcome contributions to SmoothTrend! If you'd like to contribute, please follow these steps:

1. **Fork the Repository**: Click the 'Fork' button at the top right of this page and clone your fork.

2. **Create a Branch**: Create a new branch for your feature or bug fix.
   ```bash
   git checkout -b feature-branch-name
   ```

3. **Make Changes**: Make your changes in your feature branch.

4. **Follow Coding Standards**: 
   - Follow PEP 8 style guide for Python code.
   - Write clear, commented code.
   - Update documentation for any new features.

5. **Test Your Changes**: Ensure that your changes don't break any existing functionality.

6. **Commit Your Changes**: 
   ```bash
   git commit -m "A brief description of your changes"
   ```

7. **Push to GitHub**: 
   ```bash
   git push origin feature-branch-name
   ```

8. **Submit a Pull Request**: Go to the GitHub page of your fork, and click the 'New Pull Request' button.


### Code of Conduct

By participating in this project, you agree to abide by the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

Thank you for your interest in improving SmoothTrend!


## License
This project is licensed under the GNU General Public License v2.0. See the [LICENSE](LICENSE) file for details.


## Known Limitations
- The program may experience performance issues with extremely large datasets.
- Certain advanced ARIMA configurations may require manual intervention.
- The accuracy of forecasts depends on the quality and nature of the input data.


## Citing This Project
If you use SmoothTrend in your research, please cite it as follows:

```
van der Kaay, C. (2024). SmoothTrend: Holt-Winters, Holt, Simple Exponential Smoothing, ARIMA and Trend Analysis Program [Computer software]. Version 1.4. https://github.com/kaypro283/SmoothTrend
```
