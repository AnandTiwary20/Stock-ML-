#  Stock Market Analysis & Price Prediction using Python 🧠

A complete data science and Machine Learning project that analyzes stock market data, visualizes trends, and predicts future prices using machine learning and deep learning models. Built using Python and its powerful ecosystem of data libraries.

---

##  New Features
-  **Improved LSTM model** with **95% confidence interval** on predictions
-  **5%  lower prediction error** over 180+ days of data
-  Enhanced analysis for **long-term trends** using multi-day moving averages
-  User-selectable stock ticker and date range for analysis (via Streamlit UI)

---

Here’s a high-level overview of how the project functions:

1. **Stock Data Collection**  
   - Uses the `yfinance` API to download historical data for any selected company.
   - Includes OHLC (Open, High, Low, Close) and Volume data.

2. **Data Preprocessing**
   - Cleans the data: handles null values, formats dates.
   - Normalizes data for use in deep learning models.
   - Creates 100-day and 200-day moving averages for trend visualization.

3. **Exploratory Data Analysis (EDA)**
   - Visualizes historical trends using `matplotlib` and `seaborn`.
   - Shows price patterns, volume distribution, and volatility.

4. **Model Training (LSTM)**
   - Prepares data as time sequences for LSTM input.
   - Builds and trains a deep LSTM network using `Keras` and `TensorFlow`.
   - Splits data into training and testing sets to evaluate performance.

5. **Prediction & Evaluation**
   - Predicts future closing prices using trained LSTM.
   - Compares predicted vs actual prices.
   - Plots results and calculates RMSE (Root Mean Squared Error).
   - Visualizes prediction confidence bands with 95% certainty.

6. **Interactive Web App (Optional)**
   - A `Streamlit` app allows users to enter a stock ticker and view:
     - Price charts
     - Moving averages
     - Future predictions

## 🛠️ Tech Stack

| Purpose               | Library / Tool        |
|-----------------------|-----------------------|
| Data Fetching         | `yfinance`            |
| Data Manipulation     | `pandas`, `numpy`     |
| Visualization         | `matplotlib`, `seaborn` |
| Machine Learning      | `scikit-learn`        |
| Deep Learning         | `keras`, `tensorflow` |
| Web App UI (Optional) | `streamlit`           |

---

## Key Features

-  **Fetch stock data** from Yahoo Finance using `yfinance`
-  **Clean and preprocess** stock time series data
-  **Visualize**:
  - Daily closing prices
  - 100-day and 200-day moving averages
  - Correlation heatmaps
  - Volume and volatility
-  **Train predictive models**:
  - Linear Regression for baseline forecasts
  - LSTM (Long Short-Term Memory) neural network for sequential prediction
-  **Evaluate** predictions using:
  - RMSE (Root Mean Square Error)
  - Confidence intervals
-  **Enhanced prediction accuracy** with optimized LSTM hyperparameters

---


