# time_series_analysis_204.py
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# Standard library imports
import os
import random
import webbrowser
import winsound
import time

# Third-party imports for data handling and numerical operations
'''
Use numpy 1.26.4
'''
import numpy as np
import pandas as pd

# Statistical modeling and diagnostics from statsmodels
import statsmodels.tsa.arima.model as arima_model
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from multiprocessing import Pool

# Additional statistical tools from scipy
from scipy import stats
from scipy.stats import binomtest

# Visualization libraries
import matplotlib.pyplot as plt
import plotly.express as px

# Time series analysis with auto_arima from pmdarima
from pmdarima import auto_arima

# Terminal coloring with colorama
from colorama import Back, Fore, Style, init


# Initialize colorama for colored text output in the console
init(autoreset=True)

# Define constants for various text styles and colors
HEADER_BG = Back.BLUE
HEADER_FG = Fore.WHITE
HIGHLIGHT_BG = Back.GREEN
HIGHLIGHT_FG = Fore.WHITE
INFO_FG = Fore.CYAN
WARNING_FG = Fore.RED
PROMPT_FG = Fore.YELLOW
DESC_FG = Fore.YELLOW
OPTION_FG = Fore.GREEN
BORDER = Fore.BLUE
RESET = Style.RESET_ALL

# Minimum number of data points required for analysis
MIN_DATA_POINTS = 10

# List of available foreground colors for random text coloring
COLORS = [
    Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA,
    Fore.CYAN, Fore.WHITE
]

# Function to return text with random colors


def random_color_text(text):
    return ''.join(random.choice(COLORS) + char for char in text) + Style.RESET_ALL


# Function to display the title screen of the program
def display_title_screen():
    title = ("SmoothTrend: Holt-Winters, Holt, Simple Exponential Smoothing, ARIMA/SARIMA "
             "and Trend Analysis Program v2.04")
    author = "Author: C. van der Kaay (2024)"
    options = (
        "1. View full run-down",
        "2. Proceed with the program"
    )

    # Print the title screen with styled text
    print(HEADER_BG + HEADER_FG + Style.BRIGHT + "\n" + "=" * 50)
    print(Fore.LIGHTWHITE_EX + Style.BRIGHT + f"{title.center(50)}")
    print("=" * 50 + Style.RESET_ALL)

    print(INFO_FG + "\nProgram Summary:")
    print("This program performs advanced time series analysis using Holt-Winters, Holt Exponential Smoothing, "
          "Simple Exponential Smoothing, and ARIMA/SARIMA methods. Key features include:")
    print("- Data input: Manual entry or import from CSV files")
    print("- Trend detection using Mann-Kendall and Cox-Stuart tests")
    print("- Stationarity testing with Augmented Dickey-Fuller test")
    print("- Seasonality detection using Fast Fourier Transform (FFT) and spectral analysis")
    print("- Detailed residual analysis including Ljung-Box, Shapiro-Wilk, and Kolmogorov-Smirnov tests")
    print("- Heteroscedasticity tests: Breusch-Pagan, White, and Spearman's rank correlation")
    print("- Descriptive statistics computation: mean, median, mode, standard deviation, skewness, kurtosis, etc.")
    print("- Data visualization: static and interactive plots, decomposition plots, residual plots, and ACF/PACF plots")
    print("- Lag plot analysis to check for randomness")
    print("- Forecasting with prediction intervals and error statistics (MSE, MAE, RMSE, MAPE)")
    print("- Optimal parameter selection for each smoothing method")
    print("- Automatic model selection for ARIMA/SARIMA using auto_arima, handling both seasonal and non-seasonal data")
    print("- Seasonal decomposition for trend, seasonal, and residual components")
    print("- Interactive data visualization using Plotly")
    print("- Parallel processing for improved performance in parameter optimization")
    print("- Comprehensive model selection guidance based on data characteristics")
    print("- Debug mode for detailed calculation viewing")
    print(INFO_FG + "\n" + "=" * 50)
    print(author)
    print("=" * 50 + Style.RESET_ALL)
    print()

    # Print each option on a new line
    for option in options:
        print(option)


# Function to display the full run-down of the program features


def view_full_rundown():
    print(HEADER_BG + HEADER_FG + Style.BRIGHT + "\n" + "=" * 70)
    print("Full Program Run-down".center(70))
    print("=" * 70 + Style.RESET_ALL)
    print(INFO_FG + "\nThis program offers comprehensive time series analysis with the following features:")

    print("\n1. Data Input and Preprocessing:")
    print("   - Manual data entry or import from CSV files")
    print("   - Minimum data point requirement to ensure reliable analysis")
    print("   - Descriptive statistics computation (mean, median, mode, standard deviation, skewness, etc.)")

    print("\n2. Trend Analysis:")
    print("   a. Mann-Kendall test:")
    print("      - Non-parametric test for monotonic trends in time series")
    print("      - Robust to outliers and non-normal distributions")
    print("   b. Cox-Stuart test:")
    print("      - Another non-parametric test for detecting trends")
    print("      - Compares first and second half of the data series")

    print("\n3. Stationarity Testing:")
    print("   - Augmented Dickey-Fuller (ADF) test:")
    print("     - Tests the null hypothesis of a unit root in the time series")
    print("     - Helps determine if differencing is needed for further analysis")

    print("\n4. Seasonality and Spectral Analysis:")
    print("   - Automatic detection of seasonality using Fast Fourier Transform (FFT)")
    print("   - Spectral analysis to identify significant frequencies in the data")
    print("   - Seasonal decomposition to separate trend, seasonal, and residual components")

    print("\n5. Exponential Smoothing Models:")
    print("   a. Holt-Winters Exponential Smoothing:")
    print("      - Triple exponential smoothing for data with trend and seasonality")
    print("      - Optimal selection of alpha, beta, and gamma parameters")
    print("   b. Holt Exponential Smoothing:")
    print("      - Double exponential smoothing for data with trend")
    print("      - Optimal selection of alpha and beta parameters")
    print("   c. Simple Exponential Smoothing:")
    print("      - Single exponential smoothing for data without clear trend or seasonality")
    print("      - Optimal selection of alpha parameter")

    print("\n6. ARIMA/SARIMA Modeling:")
    print("   - Automatic order selection using auto_arima")
    print("   - Fits ARIMA (Autoregressive Integrated Moving Average) or SARIMA (Seasonal ARIMA) model to the data")
    print("   - Automatically detects and handles both seasonal and non-seasonal time series")
    print("   - Uses detected seasonality to inform model selection")
    print("   - Capable of modeling complex patterns including trends, cycles, and seasonality")

    print("\n7. Forecasting:")
    print("   - Generation of point forecasts for user-specified number of periods")
    print("   - Calculation of prediction intervals for forecasts")
    print("   - Computation of error statistics: MSE, MAE, RMSE, MAPE")

    print("\n8. Residual Analysis:")
    print("   a. Ljung-Box test for autocorrelation in residuals")
    print("   b. Shapiro-Wilk test for normality of residuals")
    print("   c. Kolmogorov-Smirnov test for normality of residuals")
    print("   d. Heteroscedasticity tests:")
    print("      - Breusch-Pagan test")
    print("      - White's test")
    print("      - Spearman's rank correlation test")

    print("\n9. Data Visualization:")
    print("   - Static plots using Matplotlib:")
    print("     * Original data plot")
    print("     * Fitted values and forecasts plot")
    print("     * Residual plot")
    print("     * Autocorrelation Function (ACF) plot")
    print("     * Partial autocorrelation Function (PACF) plot")
    print("     * Seasonal decomposition plot")
    print("     * Lag plot")
    print("   - Interactive plot using Plotly (opens in web browser)")

    print("\n10. Additional Features:")
    print("   - Debug mode for viewing detailed calculations during analysis")
    print("   - Option to rerun analysis, start over, or quit at various stages")
    print("   - Colorized console output for improved readability")
    print("   - Sound alerts for warnings and errors")

    print("\nThe program guides you through each step, allowing you to choose the appropriate")
    print("analysis methods based on your data characteristics and research questions.")
    print("Results are presented with both statistical outputs and visualizations to aid interpretation.")

    print("\n" + "=" * 70)
    input(PROMPT_FG + "\nPress Enter to proceed with the program..." + Style.RESET_ALL)


# Prints a formatted menu for selecting a modeling method with descriptions and color-coded text.
def print_menu():
    border_line = BORDER + "-" * 78 + RESET
    print(border_line)
    print(PROMPT_FG + "\nBased on the trend and seasonality analysis we've performed, you can choose an "
          + " " * 18 + "\n"
          + "appropriate modeling method. "
          + " " * 53)
    print("While the analysis can guide your choice, the final decision depends on your " + " " * 13)
    print("understanding of the data and the specific forecasting needs. " + " " * 28)
    print(border_line)
    print(PROMPT_FG + "Here are the available modeling methods:" + " " * 36)
    print(border_line)

    methods = [
        ("Holt-Winters Exponential Smoothing",
         "Best for data with clear trend and seasonality",
         "Handles level, trend, and seasonal components"),
        ("Holt Exponential Smoothing",
         "Suitable for data with trend but no seasonality",
         "Handles level and trend components"),
        ("Simple Exponential Smoothing",
         "Ideal for data without clear trend or seasonality",
         "Focuses on the level component only"),
        ("ARIMA/SARIMA (Autoregressive Integrated Moving Average)",
         "Versatile method that can handle various patterns",
         "Can accommodate trend, seasonality, and complex autocorrelations\nAutomatically detects and applies seasonal "
         "or non-seasonal modeling as appropriate")
    ]

    for i, (title, desc1, desc2) in enumerate(methods, 1):
        print(OPTION_FG + f"{i}. {title}" + " " * (78 - len(title) - 3))
        print(DESC_FG + f"    - {desc1}" + " " * (78 - len(desc1) - 7))
        print(DESC_FG + f"    - {desc2}" + " " * (78 - len(desc2) - 7))
        if i < len(methods):
            print(border_line)

    print(border_line)


# Function that takes residuals as an argument and plots the ACF. Used in
# classes: HoltWintersExponentialSmoothing, HoltExponentialSmoothing, SimpleExponentialSmoothing, and ARIMA.
def plot_residual_acf(residuals):
    max_lags = len(residuals) - 1
    plot_acf(np.array(residuals), lags=np.arange(1, max_lags + 1))
    plt.title('Residual Autocorrelation Function (ACF)', fontsize=14)
    plt.xlabel('Lags', fontsize=12)
    plt.ylabel('Autocorrelation', fontsize=12)
    plt.grid(True)
    plt.show()


def plot_residual_pacf(residuals):
    max_lags = len(residuals) // 2  # Set the maximum number of lags to 50% of the sample size
    plot_pacf(np.array(residuals), lags=max_lags)
    plt.title('Residual Partial Autocorrelation Function (PACF)', fontsize=14)
    plt.xlabel('Lags', fontsize=12)
    plt.ylabel('Partial Autocorrelation', fontsize=12)
    plt.grid(True)
    plt.show()


# Evaluates autocorrelation in residuals using the Ljung-Box test and alerts if there are too few residuals or errors.
def perform_residual_analysis(residuals):
    if len(residuals) <= 1:
        winsound.Beep(1000, 500)
        print(WARNING_FG + "Not enough residuals to perform analysis.")
        return

    try:
        lb_result = acorr_ljungbox(residuals, lags=[min(10, len(residuals) // 2)], return_df=True)
        lb_pvalue = lb_result['lb_pvalue'].iloc[0]
        print(f"\n1. **Ljung-Box Test**")
        print(f"   - **Purpose:** Tests for the presence of autocorrelation in residuals.")
        print(f"   - **p-value:** {lb_pvalue:.4f}")
        if lb_pvalue < 0.05:
            print(WARNING_FG + "   - **Interpretation:** Warning: Residuals may be autocorrelated.")
        else:
            print("   - **Interpretation:** No significant autocorrelation detected in residuals.")
    except Exception as e:
        winsound.Beep(1000, 500)
        print(WARNING_FG + f"Error during residual analysis: {e}")


# Plot the Autocorrelation Function (ACF) of the residuals. Used in
# classes: HoltWintersExponentialSmoothing, HoltExponentialSmoothing, SimpleExponentialSmoothing, and ARIMA.


def plot_residuals(residuals):
    plt.figure(figsize=(10, 6))
    time_periods = range(1, len(residuals))
    plt.plot(time_periods, residuals[1:], marker='o', linestyle='-', color='b')
    plt.title('Residuals Over Time', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Residual', fontsize=12)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    plt.grid(True)
    plt.show()


# Lag plot to check whether a data set or time series is random or not

def plot_lag(data, lag=1):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:-lag], data[lag:])
    plt.xlabel(f'y(t)')
    plt.ylabel(f'y(t+{lag})')
    plt.title(f'Lag Plot (lag={lag})')
    plt.grid(True)
    plt.show()


# Function to calculate descriptive statistics of the given data
def calculate_descriptive_statistics(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    cv = std_dev / mean if mean != 0 else np.nan
    mean_absolute_deviation = np.mean(np.abs(data - mean))
    median_absolute_deviation = np.median(np.abs(data - np.median(data)))
    mode_result = stats.mode(data)

    if hasattr(mode_result, 'mode'):
        mode = mode_result.mode
    else:
        mode = mode_result[0]

    stats_dict = {
        'Count': len(data),
        'Mean': mean,
        'Median': np.median(data),
        'Mode': mode,
        'Range': np.ptp(data),
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data),
        'Variance': np.var(data),
        'Standard Deviation': std_dev,
        'Coefficient of Variation': cv,
        'IQR': stats.iqr(data),
        'Semi-IQR': stats.iqr(data) / 2,
        'Mean Absolute Deviation': mean_absolute_deviation,
        'Median Absolute Deviation': median_absolute_deviation
    }
    return stats_dict

# Function to display descriptive statistics


def display_descriptive_statistics(stats_dict):
    print_section_header("Descriptive Statistics")
    for key, value in stats_dict.items():
        if key == 'Count':
            print(f"{key}: {value}")
        elif key == 'Coefficient of Variation':
            print(f"{key}: {value:.4%}")
        else:
            print(f"{key}: {value:.4f}")

# Mann-Kendall test for trend detection


def mk_test(x, alpha=0.05):
    n = len(x)
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])

    unique_x = np.unique(x)
    g = len(unique_x)
    if n == g:
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
    else:
        tp = np.array([np.sum(x == ux) for ux in unique_x])
        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

    z = 0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)

    p = 2*(1 - stats.norm.cdf(abs(z)))
    h = abs(z) > stats.norm.ppf(1 - alpha / 2)

    return h, p, z


# Cox-Stuart test for trend detection


def cox_stuart_test(data):
    n = len(data)
    m = n // 2
    if n % 2 != 0:
        data = data[:-1]
    diff = [data[i + m] - data[i] for i in range(m)]
    pos_diff = sum(d > 0 for d in diff)
    neg_diff = sum(d < 0 for d in diff)
    result = binomtest(pos_diff, pos_diff + neg_diff, alternative='two-sided')
    p_value = result.pvalue
    return p_value

# Augmented Dickey-Fuller (ADF) test for stationarity


def adf_test(data):
    result = adfuller(data)
    if isinstance(result, (tuple, list)):
        p_value = result[1]
    else:
        p_value = result
    return p_value


# Class for Holt-Winters Exponential Smoothing
class HoltWintersExponentialSmoothing:
    def __init__(self, alpha, beta, gamma, season_length):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_length = season_length
        self.level = None
        self.trend = None
        self.season = []
        self.residuals = []
        self.fitted_values = []

    def fit(self, data):
        if len(data) < 2 * self.season_length:
            winsound.Beep(1000, 500)
            print(WARNING_FG + "Need at least two full season lengths of data to initialize.")
            return

        self.level = np.mean(data[:self.season_length])
        self.trend = (np.mean(data[self.season_length:2*self.season_length]) -
                      np.mean(data[:self.season_length])) / self.season_length
        self.season = [data[i] - self.level for i in range(self.season_length)]

        self.fitted_values = []
        self.residuals = []

        for i in range(len(data)):
            if i == 0:
                fitted_value = self.level + self.trend + self.season[i % self.season_length]
            else:
                last_level = self.level
                self.level = self.alpha * (data[i] - self.season[i % self.season_length]) + \
                    (1 - self.alpha) * (self.level + self.trend)
                self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * self.trend
                self.season[i % self.season_length] = self.gamma * (data[i] - self.level) + \
                    (1 - self.gamma) * self.season[i % self.season_length]
                fitted_value = self.level + self.trend + self.season[i % self.season_length]

            self.fitted_values.append(fitted_value)
            self.residuals.append(data[i] - fitted_value)

        return self.fitted_values

    def forecast(self, periods):
        forecasts = []
        for t in range(1, periods + 1):
            forecasts.append(self.level + t * self.trend + self.season[t % self.season_length])

        mse = np.mean(np.array(self.residuals) ** 2)
        pred_intervals = [(f - 1.96 * np.sqrt(mse), f + 1.96 * np.sqrt(mse)) for f in forecasts]

        return forecasts, pred_intervals

    def calculate_statistics(self, data):
        errors = [d - f for d, f in zip(data, self.fitted_values)]
        mse = sum(e ** 2 for e in errors) / len(errors)
        mae = sum(abs(e) for e in errors) / len(errors)
        rmse = mse ** 0.5

        non_zero_data = [(d, e) for d, e in zip(data, errors) if d != 0]
        if non_zero_data:
            mape = sum(abs(e / d) for d, e in non_zero_data) / len(non_zero_data) * 100
        else:
            mape = float('inf')

        return mse, mae, rmse, mape

    # Perform residual analysis
    def perform_residual_analysis(self):
        perform_residual_analysis(self.residuals)

    def check_residual_normality(self):
        try:
            _, shapiro_p_value = stats.shapiro(self.residuals)
            _, ks_p_value = stats.kstest(self.residuals, 'norm', args=(np.mean(self.residuals), np.std(self.residuals)))
            return shapiro_p_value, ks_p_value
        except Exception as e:
            winsound.Beep(1000, 500)
            print(WARNING_FG + f"Error checking residual normality: {e}")
            return None, None

    def check_heteroscedasticity(self):
        if len(self.residuals) != len(self.fitted_values):
            winsound.Beep(1000, 500)
            print(WARNING_FG + f"Error: Mismatch in the number of residuals ({len(self.residuals)}) "
                               f"and fitted values ({len(self.fitted_values)}).")
            return

        try:
            bp_test = breusch_pagan_test(self.residuals, self.fitted_values)
            print(f"\n4. **Breusch-Pagan Test**")
            print(f"   - **Purpose:** Tests for heteroscedasticity (changing variance) in residuals.")
            print(f"   - **p-value:** {bp_test:.4f}")
            if bp_test < 0.05:
                print(WARNING_FG + "   - **Interpretation:** Warning: Residuals may be heteroscedastic "
                                   "(Breusch-Pagan test).")
            else:
                print("   - **Interpretation:** No significant heteroscedasticity detected in residuals "
                      "(Breusch-Pagan test).")

            white_test = white_test_heteroscedasticity(self.residuals, self.fitted_values)
            print(f"\n5. **White's Test**")
            print(f"   - **Purpose:** Another test for heteroscedasticity in residuals.")
            print(f"   - **p-value:** {white_test:.4f}")
            if white_test < 0.05:
                print(WARNING_FG + "   - **Interpretation:** Warning: Residuals may be heteroscedastic (White's test).")
            else:
                print("   - **Interpretation:** No significant heteroscedasticity detected in residuals "
                      "(White's test).")

            spearman_p_value = heteroscedasticity_test(self.residuals, self.fitted_values)
            print(f"\n6. **Spearman's Rank Correlation Test**")
            print(f"   - **Purpose:** Tests for heteroscedasticity using rank correlation.")
            print(f"   - **p-value:** {spearman_p_value:.4f}")
            if spearman_p_value < 0.05:
                print(WARNING_FG + "   - **Interpretation:** Warning: Residuals may be heteroscedastic "
                                   "(Spearman's rank correlation).")
            else:
                print("   - **Interpretation:** No significant heteroscedasticity detected in residuals "
                      "(Spearman's rank correlation).")

        except Exception as e:
            winsound.Beep(1000, 500)
            print(WARNING_FG + f"Error during heteroscedasticity test: {e}")

    def plot_residuals(self):
        plot_residuals(self.residuals)

    # Plot Autocorrelation Function (ACF) of residuals
    def plot_residual_acf(self):
        plot_residual_acf(self.residuals)

    def plot_residual_pacf(self):
        plot_residual_pacf(self.residuals)

# Class for Holt Exponential Smoothing


class HoltExponentialSmoothing:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = None
        self.residuals = []
        self.fitted_values = []

    # Fit the model to the data
    def fit(self, data):
        if len(data) < 2:
            winsound.Beep(1000, 500)
            print(WARNING_FG + "Need at least two data points to initialize.")
            return

        # Initialize the level and trend
        self.level = np.mean(data[:4])
        self.trend = np.mean(np.diff(data[:4]))

        self.fitted_values = []
        self.residuals = []

        for i in range(len(data)):
            if i == 0:
                fitted_value = self.level
            else:
                last_level = self.level
                self.level = self.alpha * data[i] + (1 - self.alpha) * (self.level + self.trend)
                self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * self.trend
                fitted_value = self.level

            self.fitted_values.append(fitted_value)
            self.residuals.append(data[i] - fitted_value)

        self.fitted_values.pop(0)
        self.residuals.pop(0)

        return self.fitted_values

    # Forecast future values
    def forecast(self, periods):
        forecasts = [self.level + t * self.trend for t in range(1, periods + 1)]

        mse = np.mean(np.array(self.residuals) ** 2)

        pred_intervals = []
        for t in range(1, periods + 1):
            interval_range = 1.96 * np.sqrt(mse * (1 + t * self.alpha + (t ** 2) * self.beta))
            forecast_mean = forecasts[t - 1]
            pred_intervals.append((forecast_mean - interval_range, forecast_mean + interval_range))

        return forecasts, pred_intervals

    # Calculate error statistics
    def calculate_statistics(self, data):
        errors = [d - f for d, f in zip(data[1:], self.fitted_values)]
        mse = sum(e ** 2 for e in errors) / len(errors)
        mae = sum(abs(e) for e in errors) / len(errors)
        rmse = mse ** 0.5

        non_zero_data = [(d, e) for d, e in zip(data[1:], errors) if d != 0]
        if non_zero_data:
            mape = sum(abs(e / d) for d, e in non_zero_data) / len(non_zero_data) * 100
        else:
            mape = float('inf')

        return mse, mae, rmse, mape

    # Perform residual analysis
    def perform_residual_analysis(self):
        perform_residual_analysis(self.residuals)

    # Check normality of residuals
    def check_residual_normality(self):
        try:
            _, shapiro_p_value = stats.shapiro(self.residuals)
            _, ks_p_value = stats.kstest(self.residuals, 'norm', args=(np.mean(self.residuals), np.std(self.residuals)))
            return shapiro_p_value, ks_p_value
        except Exception as e:
            winsound.Beep(1000, 500)
            print(WARNING_FG + f"Error checking residual normality: {e}")
            return None, None

    # Check for heteroscedasticity
    def check_heteroscedasticity(self):
        if len(self.residuals) != len(self.fitted_values):
            winsound.Beep(1000, 500)
            print(WARNING_FG + f"Error: Mismatch in the number of residuals ({len(self.residuals)}) "
                               f"and fitted values ({len(self.fitted_values)}).")
            return

        try:
            bp_test = breusch_pagan_test(self.residuals, self.fitted_values)
            print(f"\n4. **Breusch-Pagan Test**")
            print(f"   - **Purpose:** Tests for heteroscedasticity (changing variance) in residuals.")
            print(f"   - **p-value:** {bp_test:.4f}")
            if bp_test < 0.05:
                print(WARNING_FG + "   - **Interpretation:** Warning: Residuals may be heteroscedastic "
                                   "(Breusch-Pagan test).")
            else:
                print("   - **Interpretation:** No significant heteroscedasticity detected in residuals "
                      "(Breusch-Pagan test).")

            white_test = white_test_heteroscedasticity(self.residuals, self.fitted_values)
            print(f"\n5. **White's Test**")
            print(f"   - **Purpose:** Another test for heteroscedasticity in residuals.")
            print(f"   - **p-value:** {white_test:.4f}")
            if white_test < 0.05:
                print(WARNING_FG + "   - **Interpretation:** Warning: Residuals may be heteroscedastic (White's test).")
            else:
                print("   - **Interpretation:** No significant heteroscedasticity detected in residuals "
                      "(White's test).")

            spearman_p_value = heteroscedasticity_test(self.residuals, self.fitted_values)
            print(f"\n6. **Spearman's Rank Correlation Test**")
            print(f"   - **Purpose:** Tests for heteroscedasticity using rank correlation.")
            print(f"   - **p-value:** {spearman_p_value:.4f}")
            if spearman_p_value < 0.05:
                print(WARNING_FG + "   - **Interpretation:** Warning: Residuals may be heteroscedastic "
                                   "(Spearman's rank correlation).")
            else:
                print("   - **Interpretation:** No significant heteroscedasticity detected in residuals "
                      "(Spearman's rank correlation).")

        except Exception as e:
            winsound.Beep(1000, 500)
            print(WARNING_FG + f"Error during heteroscedasticity test: {e}")

    # Plot residuals over time
    def plot_residuals(self):
        plot_residuals(self.residuals)

    # Plot Autocorrelation Function (ACF) of residuals
    def plot_residual_acf(self):
        plot_residual_acf(self.residuals)

    def plot_residual_pacf(self):
        plot_residual_pacf(self.residuals)


# Class for Simple Exponential Smoothing
class SimpleExponentialSmoothing:
    def __init__(self, alpha):
        self.alpha = alpha
        self.level = None
        self.residuals = []
        self.fitted_values = []

    # Fit the model to the data
    def fit(self, data):
        if len(data) < 2:
            winsound.Beep(1000, 500)
            print(WARNING_FG + "Need at least two data points to initialize.")
            return

        self.level = data[0]
        self.fitted_values = [self.level]

        for i in range(1, len(data)):
            self.level = self.alpha * data[i] + (1 - self.alpha) * self.level
            self.fitted_values.append(self.level)
            self.residuals.append(data[i] - self.level)

        return self.fitted_values

    # Forecast future values
    def forecast(self, periods):
        forecasts = [self.level] * periods
        mse = np.mean(np.array(self.residuals) ** 2)
        pred_intervals = [(f - 1.96 * np.sqrt(mse), f + 1.96 * np.sqrt(mse)) for f in forecasts]
        return forecasts, pred_intervals

    # Calculate error statistics
    def calculate_statistics(self, data):
        errors = [d - f for d, f in zip(data, self.fitted_values)]
        mse = sum(e ** 2 for e in errors) / len(errors)
        mae = sum(abs(e) for e in errors) / len(errors)
        rmse = mse ** 0.5

        non_zero_data = [(d, e) for d, e in zip(data, errors) if d != 0]
        if non_zero_data:
            mape = sum(abs(e / d) for d, e in non_zero_data) / len(non_zero_data) * 100
        else:
            mape = float('inf')

        return mse, mae, rmse, mape

    # Perform residual analysis
    def perform_residual_analysis(self):
        perform_residual_analysis(self.residuals)

    # Check normality of residuals
    def check_residual_normality(self):
        try:
            _, shapiro_p_value = stats.shapiro(self.residuals)
            _, ks_p_value = stats.kstest(self.residuals, 'norm', args=(np.mean(self.residuals), np.std(self.residuals)))
            return shapiro_p_value, ks_p_value
        except Exception as e:
            winsound.Beep(1000, 500)
            print(WARNING_FG + f"Error checking residual normality: {e}")
            return None, None

    # Check for heteroscedasticity
    def check_heteroscedasticity(self):
        if len(self.residuals) != len(self.fitted_values):
            winsound.Beep(1000, 500)
            print(WARNING_FG + f"Error: Mismatch in the number of residuals ({len(self.residuals)}) "
                               f"and fitted values ({len(self.fitted_values)}).")
            return

        try:
            bp_test = breusch_pagan_test(self.residuals, self.fitted_values)
            print(f"\n4. **Breusch-Pagan Test**")
            print(f"   - **Purpose:** Tests for heteroscedasticity (changing variance) in residuals.")
            print(f"   - **p-value:** {bp_test:.4f}")
            if bp_test < 0.05:
                print(WARNING_FG + "   - **Interpretation:** Warning: Residuals may be heteroscedastic "
                                   "(Breusch-Pagan test).")
            else:
                print("   - **Interpretation:** No significant heteroscedasticity detected in residuals "
                      "(Breusch-Pagan test).")

            white_test = white_test_heteroscedasticity(self.residuals, self.fitted_values)
            print(f"\n5. **White's Test**")
            print(f"   - **Purpose:** Another test for heteroscedasticity in residuals.")
            print(f"   - **p-value:** {white_test:.4f}")
            if white_test < 0.05:
                print(WARNING_FG + "   - **Interpretation:** Warning: Residuals may be heteroscedastic "
                                   "(White's test).")
            else:
                print("   - **Interpretation:** No significant heteroscedasticity detected in residuals "
                      "(White's test).")

            spearman_p_value = heteroscedasticity_test(self.residuals, self.fitted_values)
            print(f"\n6. **Spearman's Rank Correlation Test**")
            print(f"   - **Purpose:** Tests for heteroscedasticity using rank correlation.")
            print(f"   - **p-value:** {spearman_p_value:.4f}")
            if spearman_p_value < 0.05:
                print(WARNING_FG + "   - **Interpretation:** Warning: Residuals may be heteroscedastic "
                                   "(Spearman's rank correlation).")
            else:
                print("   - **Interpretation:** No significant heteroscedasticity detected in residuals"
                      " (Spearman's rank correlation).")

        except Exception as e:
            winsound.Beep(1000, 500)
            print(WARNING_FG + f"Error during heteroscedasticity test: {e}")

    # Plot residuals over time
    def plot_residuals(self):
        plot_residuals(self.residuals)

    # Plot Autocorrelation Function (ACF) of residuals
    def plot_residual_acf(self):
        plot_residual_acf(self.residuals)

    # Plot Partial Autocorrelation Function (PACF) of residuals
    def plot_residual_pacf(self):
        plot_residual_pacf(self.residuals)


class CustomARIMA:
    def __init__(self, seasonal_period=None):
        self.model = None
        self.results = None
        self.residuals = []
        self.fitted_values = []
        self.order = None
        self.seasonal_order = None
        self.seasonal_period = seasonal_period
        self.auto_model = None

    def fit(self, data):
        # Use pmdarima's auto_arima to find the best ARIMA order
        if self.seasonal_period:
            auto_model = auto_arima(data, seasonal=True, m=self.seasonal_period,
                                    trace=True, error_action='ignore', suppress_warnings=True)
            self.order = auto_model.order
            self.seasonal_order = auto_model.seasonal_order
            self.model = arima_model.ARIMA(data, order=self.order, seasonal_order=self.seasonal_order)
        else:
            auto_model = auto_arima(data, seasonal=False, trace=True,
                                    error_action='ignore', suppress_warnings=True)
            self.order = auto_model.order
            self.model = arima_model.ARIMA(data, order=self.order)

        self.results = self.model.fit()
        self.fitted_values = self.results.fittedvalues
        self.residuals = self.results.resid

        if self.seasonal_order:
            print(INFO_FG + f"Auto ARIMA selected a seasonal ARIMA{self.order}{self.seasonal_order} model.")
        else:
            print(INFO_FG + f"Auto ARIMA selected a non-seasonal ARIMA{self.order} model.")

        print(f"AIC of the selected model: {self.results.aic:.2f}")

        return self.fitted_values

    def forecast(self, periods):
        forecast_result = self.results.get_forecast(steps=periods)
        forecasts = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()

        if isinstance(conf_int, pd.DataFrame):
            pred_intervals = list(zip(conf_int.iloc[:, 0], conf_int.iloc[:, 1]))
        else:  # numpy array
            pred_intervals = list(zip(conf_int[:, 0], conf_int[:, 1]))

        return forecasts, pred_intervals

    def calculate_statistics(self, data):
        errors = np.array(self.residuals)
        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(mse)
        data_np = np.array(data)
        mape = np.mean(np.abs(errors / data_np)) * 100 if np.all(data_np != 0) else float('inf')
        return mse, mae, rmse, mape

    def perform_residual_analysis(self):
        perform_residual_analysis(self.residuals)

    def check_residual_normality(self):
        try:
            _, shapiro_p_value = stats.shapiro(self.residuals)
            _, ks_p_value = stats.kstest(self.residuals, 'norm', args=(np.mean(self.residuals), np.std(self.residuals)))
            return shapiro_p_value, ks_p_value
        except Exception as e:
            winsound.Beep(1000, 500)
            print(WARNING_FG + f"Error checking residual normality: {e}")
            return None, None

    def check_heteroscedasticity(self):
        if len(self.residuals) != len(self.fitted_values):
            print(WARNING_FG + f"Error: Mismatch in the number of residuals ({len(self.residuals)}) "
                               f"and fitted values ({len(self.fitted_values)})." + Style.RESET_ALL)
            return

        try:
            bp_test = breusch_pagan_test(self.residuals, self.fitted_values)
            white_test = white_test_heteroscedasticity(self.residuals, self.fitted_values)
            spearman_p_value = heteroscedasticity_test(self.residuals, self.fitted_values)

            print(f"\n4. **Breusch-Pagan Test**")
            print(f"   - **Purpose:** Tests for heteroscedasticity (changing variance) in residuals.")
            print(f"   - **p-value:** {bp_test:.4f}")
            if bp_test < 0.05:
                print(WARNING_FG + "   - **Interpretation:** Reject the null hypothesis. "
                                   "There is evidence to suggest the residuals are heteroscedastic." + Style.RESET_ALL)
            else:
                print("   - **Interpretation:** Fail to reject the null hypothesis. There is not enough evidence "
                      "to conclude the residuals are heteroscedastic.")

            print(f"\n5. **White's Test**")
            print(f"   - **Purpose:** Another test for heteroscedasticity in residuals.")
            print(f"   - **p-value:** {white_test:.4f}")
            if white_test < 0.05:
                print(WARNING_FG + "   - **Interpretation:** Reject the null hypothesis. "
                                   "There is evidence to suggest the residuals are heteroscedastic." + Style.RESET_ALL)
            else:
                print("   - **Interpretation:** Fail to reject the null hypothesis. There is not enough evidence "
                      "to conclude the residuals are heteroscedastic.")

            print(f"\n6. **Spearman's Rank Correlation Test**")
            print(f"   - **Purpose:** Tests for heteroscedasticity using rank correlation.")
            print(f"   - **p-value:** {spearman_p_value:.4f}")
            if spearman_p_value < 0.05:
                print(WARNING_FG + "   - **Interpretation:** Reject the null hypothesis. "
                                   "There is evidence to suggest the residuals are heteroscedastic." + Style.RESET_ALL)
            else:
                print("   - **Interpretation:** Fail to reject the null hypothesis. There is not enough evidence "
                      "to conclude the residuals are heteroscedastic.")

        except Exception as e:
            winsound.Beep(1000, 500)
            print(WARNING_FG + f"Error during heteroscedasticity test: {e}")
            return None, None, None

    def plot_residuals(self):
        plot_residuals(self.residuals)

    def plot_residual_acf(self):
        plot_residual_acf(self.residuals)

    def plot_residual_pacf(self):
        plot_residual_pacf(self.residuals)


# Plot the actual data, fitted values, forecasts, and prediction intervals
def plot_results(data, fitted_values, forecasts, pred_intervals, exclude_initial_fitted=1):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(data)), data, label='Actual Data', marker='o', linestyle='-', color='b')
    plt.plot(range(exclude_initial_fitted + 1, len(fitted_values) + 1), fitted_values[exclude_initial_fitted:],
             label='Fitted Values', linestyle='--', color='g')

    forecast_range = range(len(data), len(data) + len(forecasts))
    plt.plot(forecast_range, forecasts, label='Forecasts', linestyle=':', color='r')

    lower_bounds = [pi[0] for pi in pred_intervals]
    upper_bounds = [pi[1] for pi in pred_intervals]
    plt.fill_between(forecast_range, lower_bounds, upper_bounds, color='gray', alpha=0.3, label='Prediction Interval')

    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Exponential Smoothing: Data, Fitted Values, and Forecasts', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

# Perform the Ljung-Box test for autocorrelation in residuals


def ljung_box_test(residuals, lags=None):
    if lags is None:
        lags = min(10, len(residuals) // 2)
    result = acorr_ljungbox(residuals, lags=[lags])
    return result['lb_pvalue'].iloc[0]

# Test for heteroscedasticity in residuals using Spearman's rank correlation


def heteroscedasticity_test(residuals, fitted_values):
    abs_residuals = np.abs(residuals)
    correlation, p_value = stats.spearmanr(fitted_values, abs_residuals)
    return p_value

# Perform the Breusch-Pagan test for heteroscedasticity


def breusch_pagan_test(residuals, fitted_values):
    x = sm.add_constant(fitted_values)
    test_result = het_breuschpagan(residuals, x)
    return test_result[1] if isinstance(test_result, (tuple, list)) else test_result

# Perform White's test for heteroscedasticity


def white_test_heteroscedasticity(residuals, fitted_values):
    x = sm.add_constant(fitted_values)
    test_result = het_white(residuals, x)
    return test_result[1] if isinstance(test_result, (tuple, list)) else test_result

# Find the optimal alpha and beta values for Holt Exponential Smoothing


def find_optimal_alpha_beta(data, alphas, betas, debug=False):
    start_time = time.time()
    best_mse = float('inf')
    best_alpha = None
    best_beta = None

    for alpha in alphas:
        for beta in betas:
            model = HoltExponentialSmoothing(alpha, beta)
            model.fit(data)
            mse, _, _, _ = model.calculate_statistics(data)
            if debug:
                print(f"Alpha: {alpha:.2f}, Beta: {beta:.2f}, MSE: {mse:.2f}")
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
                best_beta = beta
    end_time = time.time()
    elapsed_time = end_time - start_time

    return best_alpha, best_beta, best_mse, elapsed_time


# Find the optimal alpha value for Simple Exponential Smoothing
def find_optimal_alpha(data, alphas, debug=False):
    start_time = time.time()
    best_mse = float('inf')
    best_alpha = None

    for alpha in alphas:
        model = SimpleExponentialSmoothing(alpha)
        model.fit(data)
        mse, _, _, _ = model.calculate_statistics(data)
        if debug:
            print(f"Alpha: {alpha:.2f}, MSE: {mse:.2f}")
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    end_time = time.time()
    elapsed_time = end_time - start_time

    return best_alpha, best_mse, elapsed_time

# Find the optimal alpha, beta, and gamma values for Holt-Winters Exponential Smoothing using parallel processing


def evaluate_params(args):
    data, alpha, beta, gamma, season_length = args
    model = HoltWintersExponentialSmoothing(alpha, beta, gamma, season_length)
    try:
        model.fit(data)
        mse, _, _, _ = model.calculate_statistics(data)
        return alpha, beta, gamma, mse
    except ZeroDivisionError:
        return alpha, beta, gamma, float('inf')


def find_optimal_holt_winters_params(data, alphas, betas, gammas, season_length, debug=False):
    start_time = time.time()

    param_combinations = [(data, alpha, beta, gamma, season_length)
                          for alpha in alphas
                          for beta in betas
                          for gamma in gammas]

    with Pool() as pool_obj:  # Renamed to avoid shadowing
        results = pool_obj.map(evaluate_params, param_combinations)

    valid_results = [r for r in results if r[3] != float('inf')]

    if not valid_results:
        print(WARNING_FG + "No valid parameter combinations found. "
                           "Please ensure you have sufficient data for analysis.")
        return None, None, None, None, None

    best_params = min(valid_results, key=lambda x: x[3])
    best_alpha, best_beta, best_gamma, best_mse = best_params

    end_time = time.time()
    elapsed_time = end_time - start_time

    if debug:
        print(f"Number of parameter combinations tested: {len(param_combinations)}")
        print(f"Number of valid combinations: {len(valid_results)}")

    return best_alpha, best_beta, best_gamma, best_mse, elapsed_time

# Script to detect seasonality using FFT


def detect_seasonality(data):
    n = len(data)
    fft_vals = np.fft.fft(data - np.mean(data))
    fft_freqs = np.fft.fftfreq(n)

    # Remove the zero frequency (mean) component
    fft_vals[0] = 0

    # Get the power spectrum
    power = np.abs(fft_vals) ** 2

    # Identify the peak in the power spectrum
    peak_freq = np.argmax(power)
    peak_period = np.abs(1 / fft_freqs[peak_freq])

    return int(round(peak_period))


# Print a section header with a specific style


def print_section_header(title):
    print(HIGHLIGHT_BG + HIGHLIGHT_FG + Style.BRIGHT + "\n" + "=" * 50)
    print(f"{title.center(50)}")
    print("=" * 50 + Style.RESET_ALL)

# Perform spectral analysis on the data


def spectral_analysis(data):
    fft_vals = np.fft.fft(data)
    fft_freqs = np.fft.fftfreq(len(data))

    plt.figure(figsize=(12, 6))
    plt.stem(fft_freqs, np.abs(fft_vals), 'b', markerfmt=" ", basefmt="-b")
    plt.title('Spectral Analysis (Periodogram)', fontsize=14)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True)
    plt.show()

    threshold = np.max(np.abs(fft_vals)) * 0.1
    peaks = np.where(np.abs(fft_vals) > threshold)[0]
    return fft_freqs[peaks]


# Line chart for actual data

def plot_actual_data(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Actual Data', marker='o', linestyle='-', color='b')
    plt.title('Actual Data Plot', fontsize=14)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()


# Function to create an interactive Plotly plot

def plot_interactive_data(data):
    fig = px.line(
        x=range(len(data)),
        y=data,
        title='Actual Data Plot',
        labels={'x': 'Time Period', 'y': 'Value'}
    )
    fig.update_traces(mode='lines+markers')

    # Save the figure as an HTML file
    file_path = 'interactive_plot.html'
    fig.write_html(file_path)

    # Open the HTML file in the default web browser
    webbrowser.open('file://' + os.path.realpath(file_path))


# Plot the decomposition of the data into trend, seasonal, and residual components

def plot_decomposition(data):
    period = detect_seasonality(data)
    if len(data) < 2 * period:
        print(WARNING_FG + f"Not enough data points to perform decomposition. "
                           f"At least {2 * period} observations are required.")
        return

    decomposition = seasonal_decompose(data, model='additive', period=period)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(data, label='Original')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(residual, label='Residual')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


# Perform the full analysis on the data
def perform_full_analysis(data, debug=False):
    while True:
        print_section_header("Trend Analysis")
        mk_result = mk_test(data)
        cox_stuart_p = cox_stuart_test(data)
        adf_p = adf_test(data)

        mk_trend_message = "Trend detected" if mk_result[0] else "No trend detected"
        print(f"Mann-Kendall Test: {mk_trend_message} (p-value: {mk_result[1]:.4f})")
        print(f"Cox-Stuart Test: {'Trend detected' if cox_stuart_p < 0.05 else 'No trend detected'} "
              f"(p-value: {cox_stuart_p:.4f})")
        print(f"ADF Test: {'Stationarity detected' if adf_p < 0.05 else 'No stationarity detected'} "
              f"(p-value: {adf_p:.4f})")

        if not mk_result[0] and cox_stuart_p >= 0.05 and adf_p >= 0.05:
            print(WARNING_FG + "Warning: No trend or stationarity detected.")
        else:
            print("Trend or stationarity detected. Proceeding with the analysis.")

        plot_actual_data(data)

        plot_interactive_data(data)

        print_section_header("Decomposition Analysis")
        period = detect_seasonality(data)
        if len(data) < 2 * period:
            print(WARNING_FG + f"Not enough data points to perform decomposition. "
                               f"At least {2 * period} observations are required. Skipping decomposition.")
        else:
            print(f"Detected seasonality period: {period}")
            plot_decomposition(data)

        print_section_header("Lag Plot Analysis")
        plot_lag(data)

        print_section_header("Spectral Analysis")
        significant_frequencies = spectral_analysis(data)
        if len(significant_frequencies) > 0:
            print(f"Significant frequencies detected: {significant_frequencies}")
        else:
            print("No significant seasonal frequencies detected.")

        descriptive_stats = calculate_descriptive_statistics(data)
        display_descriptive_statistics(descriptive_stats)

        # Prompt for model choice after displaying trend analysis data
        while True:
            print_menu()
            method_choice = input(PROMPT_FG + "Enter your choice (1, 2, 3, or 4): " + RESET).strip()

            if method_choice in ['1', '2', '3', '4']:
                break
            else:
                print(WARNING_FG + "Invalid choice. Please enter either '1', '2', '3', or '4'." + RESET)

        if method_choice == '1':
            # Display the warning message for Holt-Winters method
            print(WARNING_FG + Style.BRIGHT + "\nPlease be aware: Holt-Winters Exponential Smoothing is "
                                              "computationally intensive and may take several minutes"
                                              " to complete. Your computer is not locked up.\n" + Style.RESET_ALL)

            alphas = np.arange(0.01, .99, 0.01)
            betas = np.arange(0.01, .99, 0.01)
            gammas = np.arange(0.01, .99, 0.01)
            season_length = detect_seasonality(data)
            (best_alpha, best_beta, best_gamma, best_mse,
             elapsed_time) = find_optimal_holt_winters_params(data, alphas, betas, gammas, season_length, debug=debug)

            if best_alpha is None or best_beta is None or best_gamma is None:
                print(WARNING_FG + "Returning to model selection menu due to insufficient data for "
                                   "Holt-Winters method.")
                continue

            print_section_header("Optimal Parameters")
            print(f"Alpha: {best_alpha:.2f}")
            print(f"Beta: {best_beta:.2f}")
            print(f"Gamma: {best_gamma:.2f}")
            print(f"MSE: {best_mse:.2f}")
            print(f"Time taken to find optimal parameters: {elapsed_time:.2f} seconds")

            model = HoltWintersExponentialSmoothing(best_alpha, best_beta, best_gamma, season_length)
        elif method_choice == '2':
            alphas = np.arange(0.01, .99, 0.01)
            betas = np.arange(0.01, .99, 0.01)
            best_alpha, best_beta, best_mse, elapsed_time = find_optimal_alpha_beta(data, alphas, betas, debug=debug)
            print_section_header("Optimal Parameters")
            print(f"Alpha: {best_alpha:.2f}")
            print(f"Beta: {best_beta:.2f}")
            print(f"MSE: {best_mse:.2f}")
            print(f"Time taken to find optimal parameters: {elapsed_time:.2f} seconds")

            model = HoltExponentialSmoothing(best_alpha, best_beta)
        elif method_choice == '3':
            alphas = np.arange(0.01, .99, 0.01)
            best_alpha, best_mse, elapsed_time = find_optimal_alpha(data, alphas, debug=debug)
            print_section_header("Optimal Parameters")
            print(f"Alpha: {best_alpha:.2f}")
            print(f"MSE: {best_mse:.2f}")
            print(f"Time taken to find optimal parameters: {elapsed_time:.2f} seconds")

            model = SimpleExponentialSmoothing(best_alpha)
        else:  # method_choice == '4'
            season_length = detect_seasonality(data)
            if season_length > 1 and len(data) >= 2 * season_length:
                print(f"Detected seasonality with period: {season_length}")
                model = CustomARIMA(seasonal_period=season_length)
            else:
                print("No significant seasonality detected or insufficient data. Using non-seasonal ARIMA.")
                model = CustomARIMA()

        fitted_values = model.fit(data)

        while True:
            try:
                periods_input = input(PROMPT_FG + "Enter the number of forecast periods: ")
                periods = int(periods_input)
                break
            except ValueError:
                winsound.Beep(1000, 500)
                print(WARNING_FG + "Please enter a valid number for the forecast periods.")

        forecasts, pred_intervals = model.forecast(periods)

        mse, mae, rmse, mape = model.calculate_statistics(data)

        print_section_header("Statistics")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")

        print_section_header("Forecasted Values and Prediction Intervals")
        for i, (forecast, pred_interval) in enumerate(zip(forecasts, pred_intervals), 1):
            print(f"Period {i}: Forecast = {forecast:.2f}, PI = ({pred_interval[0]:.2f}, {pred_interval[1]:.2f})")

        plot_results(data, fitted_values, forecasts, pred_intervals)

        print_section_header("Performing Residual Analysis")
        model.perform_residual_analysis()

        shapiro_p_value, ks_p_value = model.check_residual_normality()
        if shapiro_p_value is not None:
            print(f"\n2. **Shapiro-Wilk Test**")
            print(f"   - **Purpose:** Tests for normality in the distribution of residuals.")
            print(f"   - **p-value:** {shapiro_p_value:.4f}")
            if shapiro_p_value > 0.05:
                print("   - **Interpretation:** Fail to reject the null hypothesis. There is not enough evidence "
                      "to conclude the residuals are not normally distributed.")
            else:
                print(WARNING_FG + "   - **Interpretation:** Reject the null hypothesis. "
                                   "There is evidence to suggest the residuals are not normally "
                                   "distributed." + Style.RESET_ALL)

        if ks_p_value is not None:
            print(f"\n3. **Kolmogorov-Smirnov Test**")
            print(f"   - **Purpose:** Another test for normality in the distribution of residuals.")
            print(f"   - **p-value:** {ks_p_value:.4f}")
            if ks_p_value > 0.05:
                print("   - **Interpretation:** Fail to reject the null hypothesis. There is not enough evidence "
                      "to conclude the residuals are not normally distributed.")
            else:
                print(WARNING_FG + "   - **Interpretation:** Reject the null hypothesis. "
                                   "There is evidence to suggest the residuals are not normally "
                                   "distributed." + Style.RESET_ALL)

        model.check_heteroscedasticity()
        model.plot_residuals()

        print_section_header("Plotting Residual ACF and PACF")
        model.plot_residual_acf()
        model.plot_residual_pacf()

        print("\nSummary of Key Findings from the ACF Plot:")
        acf_values = sm.tsa.acf(model.residuals, fft=False)
        significant_lags = np.where(np.abs(acf_values[1:]) > 1.96 / np.sqrt(len(model.residuals)))[0] + 1
        if len(significant_lags) == 0:
            print("No significant autocorrelations detected at any lags based on the ACF plot.")
        else:
            print(WARNING_FG + f"Significant autocorrelations detected at lags: {significant_lags} based on the "
                               f"ACF plot.")

        print("\nSummary of Key Findings from the PACF Plot:")
        pacf_values = pacf(model.residuals, nlags=len(model.residuals) // 2)
        significant_pacf_lags = np.where(np.abs(pacf_values[1:]) > 1.96 / np.sqrt(len(model.residuals)))[0] + 1
        if len(significant_pacf_lags) == 0:
            print("No significant partial autocorrelations detected at any lags based on the PACF plot.")
        else:
            print(WARNING_FG + f"Significant partial autocorrelations detected at lags: {significant_pacf_lags} based "
                               f"on the PACF plot.")

        # Perform Ljung-Box test
        lb_result = sm.stats.acorr_ljungbox(model.residuals, lags=[min(10, len(model.residuals) // 2)], return_df=True)
        lb_pvalue = lb_result['lb_pvalue'].iloc[0]

        if lb_pvalue > 0.05:
            print(
                f"Ljung-Box test (p-value: {lb_pvalue:.4f}) suggests no significant autocorrelation in the residuals.")
        else:
            print(WARNING_FG + f"Ljung-Box test (p-value: {lb_pvalue:.4f}) suggests significant autocorrelation in "
                               f"the residuals.")

        if len(significant_lags) == 0 and lb_pvalue > 0.05:
            print("Both ACF plot and Ljung-Box test indicate no significant autocorrelation in the residuals.")
        elif len(significant_lags) > 0 and lb_pvalue <= 0.05:
            print(WARNING_FG + "Both ACF plot and Ljung-Box test indicate significant autocorrelation in the "
                               "residuals.")
        else:
            print(WARNING_FG + "ACF plot and Ljung-Box test provide conflicting information about autocorrelation "
                               "in the residuals.")

        while True:
            rerun_choice = input(PROMPT_FG + "\nWould you like to rerun the analysis (r), start over (s), "
                                             "or quit (q)? ").strip().lower()
            if rerun_choice in ['r', 's', 'q']:
                return rerun_choice
            else:
                winsound.Beep(1000, 500)
                print(WARNING_FG + "Invalid choice. Please enter 'r' to rerun, 's' to start over, or 'q' to quit.")


def main():
    while True:
        display_title_screen()
        while True:
            choice = input(PROMPT_FG + "Enter your choice (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            else:
                winsound.Beep(1000, 500)
                print(WARNING_FG + "Invalid choice. Please enter either '1' or '2'.")

        if choice == '1':
            view_full_rundown()

        while True:
            data = []
            print(PROMPT_FG + "Would you like to manually input the data or import it from a CSV file?")
            print("1. Manually input the data")
            print("2. Import data from CSV file")

            choice = input(PROMPT_FG + "Enter your choice (1 or 2): ").strip()

            if choice == '1':
                print(PROMPT_FG + "Enter your data points one by one. Type 'done' when you are finished.")
                while True:
                    if len(data) < MIN_DATA_POINTS:
                        entry = input(PROMPT_FG + "Enter data point (or 'done' to finish): ")
                        if entry.lower() == 'done' and len(data) < MIN_DATA_POINTS:
                            winsound.Beep(1000, 500)
                            print(WARNING_FG + f"Please enter at least {MIN_DATA_POINTS} data points.")
                            continue
                        try:
                            value = float(entry)
                            data.append(value)
                        except ValueError:
                            winsound.Beep(1000, 500)
                            print(WARNING_FG + "Please enter a valid number.")
                    else:
                        entry = input(PROMPT_FG + "Enter data point (or 'done' to finish): ")
                        if entry.lower() == 'done':
                            break
                        try:
                            value = float(entry)
                            data.append(value)
                        except ValueError:
                            winsound.Beep(1000, 500)
                            print(WARNING_FG + "Please enter a valid number.")

            elif choice == '2':
                filename = input(PROMPT_FG + "Enter the path to the CSV file: ").strip()
                try:
                    data = pd.read_csv(filename, header=None).iloc[:, 0].tolist()
                    data = [float(i) for i in data]
                    if len(data) < MIN_DATA_POINTS:
                        winsound.Beep(1000, 500)
                        print(WARNING_FG + f"The file contains less than {MIN_DATA_POINTS} data points. "
                                           f"Please provide more data.")
                        continue
                except Exception as e:
                    winsound.Beep(1000, 500)
                    print(WARNING_FG + f"Error reading CSV file: {e}")
                    continue
            else:
                winsound.Beep(1000, 500)
                print(WARNING_FG + "Invalid choice. Please select either 1 or 2.")
                continue

            if len(data) < MIN_DATA_POINTS:
                winsound.Beep(1000, 500)
                print(WARNING_FG + "Not enough data points to proceed. Exiting.")
                return

            data = np.array(data)

            debug_input = input(PROMPT_FG + "Do you want to enable debug mode to see all "
                                            "calculations? (yes/no): ").strip().lower()
            debug = debug_input == 'yes'

            while True:
                rerun_choice = perform_full_analysis(data, debug)
                if rerun_choice == 'r':
                    continue  # Re-run the analysis
                elif rerun_choice == 's':
                    break  # Start over
                elif rerun_choice == 'q':
                    print("\n" + "="*60)
                    print(INFO_FG + Style.BRIGHT + "Thank you for using SmoothTrend!")
                    print(INFO_FG + "Program Created by Christopher D. van der Kaay, Ph.D.")
                    print(random_color_text("Exiting the program. Goodbye!"))
                    print("="*60 + Style.RESET_ALL)
                    time.sleep(10)  # Pause for 5 seconds before exiting
                    return

            if rerun_choice == 's':
                break


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
