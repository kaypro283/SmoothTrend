# time_series_analysis.py
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

# Import necessary libraries for statistical analysis, plotting, and sound alerts


import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import binomtest
from colorama import Fore, Style, Back, init
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import winsound
import webbrowser
import random
import os
import plotly.express as px

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
    title = ("SmoothTrend: Holt-Winters, Holt, and Simple Exponential Smoothing "
             "and Trend Analysis Program v1.2")
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
    print("This program performs advanced time series analysis using Holt-Winters, Holt Exponential Smoothing, and")
    print("Simple Exponential Smoothing methods. Key features include:")
    print("- Trend detection using Mann-Kendall and Cox-Stuart tests")
    print("- Stationarity testing with Augmented Dickey-Fuller test")
    print("- Detailed residual analysis including Ljung-Box, Shapiro-Wilk, and Kolmogorov-Smirnov tests")
    print("- Heteroscedasticity tests: Breusch-Pagan, White, and Spearman's rank correlation")
    print("- Descriptive statistics computation: mean, median, standard deviation, skewness, etc.")
    print("- Data visualization: decomposition plots, residual plots, and ACF plots")
    print("- Forecasting with prediction intervals and error statistics (MSE, MAE, RMSE, MAPE)")

    print(INFO_FG + "\n" + "=" * 50)
    print(author)
    print("=" * 50 + Style.RESET_ALL)
    print()

    # Print each option on a new line
    for option in options:
        print(option)

# Function to display the full run-down of the program features


def view_full_rundown():
    print(HEADER_BG + HEADER_FG + Style.BRIGHT + "\n" + "=" * 50)
    print("Full Program Run-down".center(50))
    print("=" * 50 + Style.RESET_ALL)
    print(INFO_FG + "\nThis program includes the following features:")
    print("1. Mann-Kendall test for trend detection")
    print("2. Cox-Stuart test for trend detection")
    print("3. Augmented Dickey-Fuller (ADF) test for stationarity")
    print("4. Holt-Winters Exponential Smoothing model fitting and forecasting")
    print("5. Holt Exponential Smoothing model fitting and forecasting")
    print("6. Simple Exponential Smoothing model fitting and forecasting")
    print("7. Residual analysis including Ljung-Box, Shapiro-Wilk, and Kolmogorov-Smirnov tests")
    print("8. Heteroscedasticity tests: Breusch-Pagan, White, and Spearman's rank correlation")
    print("9. Plotting of data, residuals, and Autocorrelation Function (ACF)")
    print("10. Spectral and decomposition analysis")
    print("\nEach section of the program will guide you through the necessary inputs and display the "
          "results accordingly.")
    print("=" * 50)
    input(PROMPT_FG + "\nPress Enter to proceed with the program..." + Style.RESET_ALL)

# Function to calculate descriptive statistics of the given data


def calculate_descriptive_statistics(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    cv = std_dev / mean if mean != 0 else np.nan
    mean_absolute_deviation = np.mean(np.abs(data - mean))
    median_absolute_deviation = np.median(np.abs(data - np.median(data)))
    stats_dict = {
        'Count': len(data),
        'Mean': mean,
        'Median': np.median(data),
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

    def perform_residual_analysis(self):
        if len(self.residuals) <= 1:
            winsound.Beep(1000, 500)
            print(WARNING_FG + "Not enough residuals to perform analysis.")
            return

        try:
            lb_result = acorr_ljungbox(self.residuals, lags=[min(10, len(self.residuals) // 2)], return_df=True)
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
        plt.figure(figsize=(10, 6))
        plt.plot(self.residuals, marker='o', linestyle='-', color='b')
        plt.title('Residuals Over Time', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Residual', fontsize=12)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
        plt.grid(True)
        plt.show()

    def plot_residual_acf(self):
        plot_acf(np.array(self.residuals))
        plt.title('Residual Autocorrelation Function (ACF)', fontsize=14)
        plt.xlabel('Lags', fontsize=12)
        plt.ylabel('Autocorrelation', fontsize=12)
        plt.grid(True)
        plt.show()


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
        if len(self.residuals) <= 1:
            winsound.Beep(1000, 500)
            print(WARNING_FG + "Not enough residuals to perform analysis.")
            return

        try:
            lb_result = acorr_ljungbox(self.residuals, lags=[min(10, len(self.residuals) // 2)], return_df=True)
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
        plt.figure(figsize=(10, 6))
        plt.plot(self.residuals, marker='o', linestyle='-', color='b')
        plt.title('Residuals Over Time', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Residual', fontsize=12)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
        plt.grid(True)
        plt.show()

    # Plot Autocorrelation Function (ACF) of residuals
    def plot_residual_acf(self):
        plot_acf(np.array(self.residuals))
        plt.title('Residual Autocorrelation Function (ACF)', fontsize=14)
        plt.xlabel('Lags', fontsize=12)
        plt.ylabel('Autocorrelation', fontsize=12)
        plt.grid(True)
        plt.show()


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
        if len(self.residuals) <= 1:
            winsound.Beep(1000, 500)
            print(WARNING_FG + "Not enough residuals to perform analysis.")
            return

        try:
            lb_result = acorr_ljungbox(self.residuals, lags=[min(10, len(self.residuals) // 2)], return_df=True)
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
        plt.figure(figsize=(10, 6))
        plt.plot(self.residuals, marker='o', linestyle='-', color='b')
        plt.title('Residuals Over Time', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Residual', fontsize=12)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
        plt.grid(True)
        plt.show()

    # Plot Autocorrelation Function (ACF) of residuals
    def plot_residual_acf(self):
        plot_acf(np.array(self.residuals))
        plt.title('Residual Autocorrelation Function (ACF)', fontsize=14)
        plt.xlabel('Lags', fontsize=12)
        plt.ylabel('Autocorrelation', fontsize=12)
        plt.grid(True)
        plt.show()

# Plot the actual data, fitted values, forecasts, and prediction intervals


def plot_results(data, fitted_values, forecasts, pred_intervals):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(data)), data, label='Actual Data', marker='o', linestyle='-', color='b')
    plt.plot(range(1, len(fitted_values) + 1), fitted_values, label='Fitted Values', linestyle='--', color='g')

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

    return best_alpha, best_beta, best_mse


# Find the optimal alpha value for Simple Exponential Smoothing
def find_optimal_alpha(data, alphas, debug=False):
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

    return best_alpha, best_mse

# Find the optimal alpha, beta, and gamma values for Holt-Winters Exponential Smoothing


def find_optimal_holt_winters_params(data, alphas, betas, gammas, season_length, debug=False):
    best_mse = float('inf')
    best_alpha = None
    best_beta = None
    best_gamma = None

    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                model = HoltWintersExponentialSmoothing(alpha, beta, gamma, season_length)
                try:
                    model.fit(data)
                    mse, _, _, _ = model.calculate_statistics(data)
                    if debug:
                        print(f"Alpha: {alpha:.2f}, Beta: {beta:.2f}, Gamma: {gamma:.2f}, MSE: {mse:.2f}")
                    if mse < best_mse:
                        best_mse = mse
                        best_alpha = alpha
                        best_beta = beta
                        best_gamma = gamma
                except ZeroDivisionError:
                    print(WARNING_FG + "ZeroDivisionError: Not enough data points to calculate MSE. "
                                       "Please ensure you have sufficient data for analysis.")
                    return None, None, None, None

    return best_alpha, best_beta, best_gamma, best_mse

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

        print_section_header("Decomposition Analysis")
        period = detect_seasonality(data)
        if len(data) < 2 * period:
            print(WARNING_FG + f"Not enough data points to perform decomposition. "
                               f"At least {2 * period} observations are required. Skipping decomposition.")
        else:
            print(f"Detected seasonality period: {period}")
            plot_decomposition(data)

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
            print(PROMPT_FG + "\nBased on the trend analysis, choose the smoothing method:")
            print("1. Holt-Winters Exponential Smoothing")
            print("2. Holt Exponential Smoothing")
            print("3. Simple Exponential Smoothing")

            method_choice = input(PROMPT_FG + "Enter your choice (1, 2, or 3): ").strip()

            if method_choice in ['1', '2', '3']:
                break
            else:
                print(WARNING_FG + "Invalid choice. Please enter either '1', '2', or '3'.")

        if method_choice == '1':
            # Display the warning message for Holt-Winters method
            print(WARNING_FG + Style.BRIGHT + "\nPlease be aware: Holt-Winters Exponential Smoothing is "
                                              "computationally intensive and may take several minutes"
                                              " to complete. Your computer is not locked up.\n" + Style.RESET_ALL)

            alphas = np.arange(0.01, .99, 0.01)
            betas = np.arange(0.01, .99, 0.01)
            gammas = np.arange(0.01, .99, 0.01)
            season_length = detect_seasonality(data)
            best_alpha, best_beta, best_gamma, best_mse = find_optimal_holt_winters_params(data, alphas, betas, gammas,
                                                                                           season_length, debug=debug)
            if best_alpha is None or best_beta is None or best_gamma is None:
                print(WARNING_FG + "Returning to model selection menu due to insufficient data for "
                                   "Holt-Winters method.")
                continue

            print_section_header("Optimal Parameters")
            print(f"Alpha: {best_alpha:.2f}")
            print(f"Beta: {best_beta:.2f}")
            print(f"Gamma: {best_gamma:.2f}")
            print(f"MSE: {best_mse:.2f}")

            model = HoltWintersExponentialSmoothing(best_alpha, best_beta, best_gamma, season_length)
        elif method_choice == '2':
            alphas = np.arange(0.01, .99, 0.01)
            betas = np.arange(0.01, .99, 0.01)
            best_alpha, best_beta, best_mse = find_optimal_alpha_beta(data, alphas, betas, debug=debug)
            print_section_header("Optimal Parameters")
            print(f"Alpha: {best_alpha:.2f}")
            print(f"Beta: {best_beta:.2f}")
            print(f"MSE: {best_mse:.2f}")

            model = HoltExponentialSmoothing(best_alpha, best_beta)
        else:  # method_choice == '3'
            alphas = np.arange(0.01, .99, 0.01)
            best_alpha, best_mse = find_optimal_alpha(data, alphas, debug=debug)
            print_section_header("Optimal Parameters")
            print(f"Alpha: {best_alpha:.2f}")
            print(f"MSE: {best_mse:.2f}")

            model = SimpleExponentialSmoothing(best_alpha)

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
            print(f"   - **Interpretation:** Residuals follow a normal distribution.")
        if ks_p_value is not None:
            print(f"\n3. **Kolmogorov-Smirnov Test**")
            print(f"   - **Purpose:** Another test for normality in the distribution of residuals.")
            print(f"   - **p-value:** {ks_p_value:.4f}")
            print(f"   - **Interpretation:** Residuals follow a normal distribution.")

        model.check_heteroscedasticity()
        model.plot_residuals()

        print_section_header("Plotting Residual ACF")
        model.plot_residual_acf()

        print("\nSummary of Key Findings from the ACF Plot:")
        acf_values = sm.tsa.acf(model.residuals, fft=False)
        significant_lags = np.where(np.abs(acf_values) > 1.96 / np.sqrt(len(model.residuals)))[0]
        if len(significant_lags) == 0:
            print("No significant autocorrelations detected at any lags.")
        else:
            print(f"Significant autocorrelations detected at lags: {significant_lags}")
        print("Ljung-Box test p-value reinforces that no significant autocorrelation is detected in the residuals.")

        while True:
            rerun_choice = input(PROMPT_FG + "\nWould you like to rerun the analysis (r), start over (s), "
                                             "or quit (q)? ").strip().lower()
            if rerun_choice == 'r':
                break  # Re-run the analysis
            elif rerun_choice == 's':
                return  # Start over
            elif rerun_choice == 'q':
                print(random_color_text("Exiting the program. Goodbye!"))
                return
            else:
                winsound.Beep(1000, 500)
                print(WARNING_FG + "Invalid choice. Please enter 'r' to rerun, 's' to start over, or 'q' to quit.")


# Main function to run the program
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

            plot_actual_data(data)  # Plot actual data before performing analysis
            plot_interactive_data(data)  # Create interactive plot
            perform_full_analysis(data, debug)

            while True:
                rerun_choice = input(PROMPT_FG + "\nWould you like to rerun the analysis (r), start over (s), "
                                                 "or quit (q)? ").strip().lower()
                if rerun_choice == 'r':
                    perform_full_analysis(data, debug)
                elif rerun_choice == 's':
                    break
                elif rerun_choice == 'q':
                    print(random_color_text("Exiting the program. Goodbye!"))
                    return
                else:
                    winsound.Beep(1000, 500)
                    print(WARNING_FG + "Invalid choice. Please enter 'r' to rerun, 's' to start over, or 'q' to quit.")

            if rerun_choice == 's':
                break


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
