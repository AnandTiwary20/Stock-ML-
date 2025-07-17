import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt


COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'accent': '#2ca02c',
    'volume': 'rgba(128, 128, 128, 0.3)',
    'background': '#f8f9fa',
    'grid': '#e0e0e0',
    'text': '#2c3e50'
}

plt.rcParams['figure.facecolor'] = COLORS['background']
plt.rcParams['axes.facecolor'] = COLORS['background']
plt.rcParams['grid.color'] = COLORS['grid']
plt.rcParams['text.color'] = COLORS['text']
plt.rcParams['axes.labelcolor'] = COLORS['text']
plt.rcParams['xtick.color'] = COLORS['text']
plt.rcParams['ytick.color'] = COLORS['text']

model = load_model("Stock Predictions Model.keras")


st.markdown("""
<div style='background-color:#f0f2f6; padding:10px; border-radius:5px; margin-bottom:20px;'>
    <p style='text-align:center; color:#1f77b4; font-weight:bold; margin:0;'>
     yfinance - Enter any valid stock symbol to analyze
    </p>
</div>
""", unsafe_allow_html=True)

st.header('Stock Market Year Ahead Analysis')

# Sidebar for user inputs
with st.sidebar:
    st.subheader('Stock Analysis Settings')
    
    # Exchange selection
    exchange = st.selectbox(
        'Select Stock Exchange',
        ('NSE', 'BSE', 'NYSE', 'NASDAQ', 'Other')
    )
    
    # Stock symbol input with exchange-specific examples
    if exchange == 'NSE':
        example = 'RELIANCE'  # Example for NSE
        suffix = '.NS'
    elif exchange == 'BSE':
        example = '500325'  # Example for BSE (Reliance Industries)
        suffix = '.BO'
    elif exchange == 'NYSE':
        example = 'GE'  # Example for NYSE
        suffix = ''
    elif exchange == 'NASDAQ':
        example = 'AAPL'  # Example for NASDAQ
        suffix = ''
    else:  # Other
        example = 'GOOG'
        suffix = ''
    
    # Stock symbol input
    stock = st.text_input(f'Enter Stock Symbol (e.g., {example})', example).strip().upper()
    
    # Add exchange suffix if needed
    if exchange in ['NSE', 'BSE'] and not stock.endswith(suffix):
        stock += suffix
    
    # Date range selection with reasonable defaults
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', 
                                 value=pd.to_datetime('2020-01-01'), 
                                 min_value=pd.to_datetime('1970-01-01'),
                                 max_value=pd.to_datetime('today'))
    with col2:
        end_date = st.date_input('End Date', 
                               value=pd.to_datetime('today'),
                               min_value=pd.to_datetime('1970-01-01'),
                               max_value=pd.to_datetime('today'))
    
    # Validate date range
    if start_date >= end_date:
        st.error('Error: End date must be after start date')
        st.stop()

# Convert dates to string format for yfinance
start = start_date.strftime('%Y-%m-%d')
end = end_date.strftime('%Y-%m-%d')

def fetch_stock_data(symbol, start_date, end_date):
    """Helper function to fetch stock data with better error handling"""
    try:
        # Try downloading with the symbol as-is first
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        # If no data, try with .NS suffix for NSE stocks
        if data.empty and not symbol.endswith('.NS'):
            st.info(f"Trying with NSE suffix for {symbol}...")
            symbol_ns = f"{symbol}.NS"
            data = yf.download(symbol_ns, start=start_date, end=end_date, progress=False)
            if not data.empty:
                return data, symbol_ns  # Return the data and the modified symbol
        
        # If still no data, try with .BO for BSE
        if data.empty and not (symbol.endswith('.NS') or symbol.endswith('.BO')):
            st.info(f"Trying with BSE suffix for {symbol}...")
            symbol_bo = f"{symbol}.BO"
            data = yf.download(symbol_bo, start=start_date, end=end_date, progress=False)
            if not data.empty:
                return data, symbol_bo  # Return the data and the modified symbol
        
        return data, symbol
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, symbol

try:
    # Try to download the stock data with enhanced handling
    data, stock = fetch_stock_data(stock, start, end)
    
    # Check if data was returned
    if data is None or data.empty:
        st.error(f"No data found for stock symbol '{stock}'. Please check the symbol and try again.")
        st.info("""
        Common issues:
        1. Make sure the stock symbol is correct
        2. For Indian stocks, try with .NS (NSE) or .BO (BSE) suffix
        3. Try a different date range
        """)
        st.stop()
        
    # Check if we have the 'Close' column
    if 'Close' not in data.columns:
        st.error(f"No price data available for {stock}. The data contains: {', '.join(data.columns)}")
        st.stop()
        
    # Display the data
    st.subheader('Stock Data')
    st.write(data)
    
except Exception as e:
    st.error(f"Error fetching data for {stock}: {str(e)}")
    st.info("Common issues:\n"
            "1. Check if the stock symbol is correct\n"
            "2. Try a different date range\n"
            "3. Check your internet connection")
    st.stop()

# Calculate minimum required data points
min_data_points = 30  # Reduced minimum requirement

# Check if we have enough data
if len(data) < min_data_points:
    st.error(f"Error: Not enough historical data available. Found {len(data)} points, but need at least {min_data_points}. Please try a different stock or a longer date range.")
    st.stop()

# Calculate training set size (80% of available data, but at least 20 points)
train_ratio = 0.8
min_train_size = 20  # Reduced minimum training size

train_size = max(min_train_size, int(len(data) * train_ratio))
# Ensure we leave at least some data for testing
if train_size >= len(data) - 5:  # Leave at least 5 points for testing
    train_size = max(min_train_size, len(data) - 5)  # Adjust to leave at least 5 points for testing

# Split the data
data_train = data.iloc[0:train_size][['Close']].copy()
data_test = data.iloc[train_size:][['Close']].copy()

# Check if we have valid data after split
if len(data_train) < min_train_size or len(data_test) < 5:
    st.error(f"Error: Could not create valid training and testing sets. Found {len(data_train)} training and {len(data_test)} testing points. Try a longer date range.")
    st.stop()

# Log data split for debugging
st.sidebar.info(f"Using {len(data_train)} points for training and {len(data_test)} points for testing.")

# Initialize and fit scaler on training data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

try:
    # Scale the training data
    if len(data_train) < 1:
        raise ValueError("Training data is empty")
        
    data_train_scaled = scaler.fit_transform(data_train)
    
    # Get last 100 days from training data for sequence creation
    pas_100_days = data_train.tail(100)
    
    # Create test data with lookback window
    test_data = pd.concat([pas_100_days, data_test])
    
    if len(test_data) < 1:
        raise ValueError("Test data is empty after concatenation")
        
    # Scale the test data using the same scaler
    test_data_scaled = scaler.transform(test_data)
    
    # For LSTM, we need to create sequences
    # Use a smaller sequence length for smaller datasets
    max_sequence_length = min(30, len(test_data_scaled) - 1)  # Maximum of 30 or available data points - 1
    min_sequence_length = 5  # Minimum sequence length
    
    # Choose sequence length based on available data
    sequence_length = min(30, max(min_sequence_length, len(test_data_scaled) // 3))
    
    st.sidebar.info(f"Using sequence length: {sequence_length} (based on available data)")
    
    if len(test_data_scaled) <= sequence_length:
        st.warning(f"Warning: Limited data available. Using all available points for sequence creation.")
        sequence_length = max(1, len(test_data_scaled) - 1)  # Use all but one point if possible
    
    X_test = []
    # Create as many sequences as possible from the available data
    for i in range(sequence_length, len(test_data_scaled)):
        X_test.append(test_data_scaled[i-sequence_length:i, 0])
    
    if not X_test:  # If we couldn't create any sequences
        # Try with a smaller sequence length as a fallback
        sequence_length = min(5, len(test_data_scaled) - 1)
        X_test = [test_data_scaled[-sequence_length:, 0]]  # Just use the last available sequence
        
        if not X_test:
            raise ValueError("Could not create any valid sequences from the available data")
    
    X_test = np.array(X_test)
    
    # Check if we have any valid sequences
    if len(X_test) == 0:
        raise ValueError("No valid sequences could be created from the test data")
    
    # Reshape for LSTM [samples, time steps, features]
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Try to load the model and make predictions
    try:
        # Make predictions
        predicted_price = model.predict(X_test)
        
        # Check if predictions were made
        if len(predicted_price) == 0:
            raise ValueError("Model did not return any predictions")
            
        # Inverse transform the predictions
        predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1)).flatten()
        
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        st.stop()
    
except Exception as e:
    st.error(f"Error in data preprocessing: {str(e)}")
    st.stop()


st.subheader('Technical Analysis')


ma_20_days = data.Close.rolling(20).mean()
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()


tab1, tab2, tab3 = st.tabs(["Moving Averages", "Price Prediction", "Volume Analysis"])

with tab1:
  
    fig1 = plt.figure(figsize=(14, 8), facecolor=COLORS['background'])
    ax1 = plt.gca()
    
    # Plot price with a gradient fill
    close_col = 'Close'  # Define close_col here
    ax1.plot(data.index, data[close_col], 
            label='Close Price', 
            color=COLORS['primary'], 
            linewidth=2.5,
            alpha=0.9)
    
   
    y1 = np.full(len(data[close_col]), data[close_col].min())
    y2 = data[close_col].values.ravel()
    x = data.index
    ax1.fill_between(x, 
                    y1, 
                    y2,
                    color=COLORS['primary'], 
                    alpha=0.1)
    
    
    ma_styles = [
        (ma_20_days, COLORS['secondary'], '--', 1.2, '20-day MA'),
        (ma_50_days, '#d62728', '-', 1.5, '50-day MA'),
        (ma_100_days, '#9467bd', '--', 1.2, '100-day MA'),
        (ma_200_days, '#2ca02c', '-', 1.8, '200-day MA')
    ]
    
    for ma, color, ls, lw, label in ma_styles:
        ax1.plot(data.index, ma, color=color, linestyle=ls, linewidth=lw, alpha=0.9, label=label)
    
    # Customize the chart
    ax1.set_facecolor(COLORS['background'])
    ax1.grid(True, linestyle='--', alpha=0.4, color=COLORS['grid'])
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax1.set_ylabel('Price', fontsize=12, fontweight='bold', color=COLORS['text'])
    
   
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:,.0f}".format(x)))
    
    
    current_price = float(data[close_col].iloc[-1])  
    stock_str = str(stock) 
    ax1.set_title(
        f"{stock_str} - Current Price: ${current_price:,.2f}", 
        fontsize=16, 
        fontweight='bold',
        pad=20,
        color=COLORS['text']
    )
    
 
    legend = ax1.legend(
        loc='upper left', 
        frameon=True, 
        framealpha=0.9, 
        facecolor='white',
        edgecolor=COLORS['grid'],
        fontsize=10
    )
    
    #
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    
    plt.tight_layout()
    
    
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.95)
    
   
    st.pyplot(fig1)


with tab2:
    st.subheader('Price Prediction Settings')
    
    
    col1, col2 = st.columns(2)
    with col1:
        prediction_days = st.slider('Number of Days to Show', 30, 365, 90, 
                                  help='Select how many days of historical data to display')
        confidence_level = st.slider('Confidence Level', 70, 99, 90, 5,
                                   help='Confidence interval for prediction range')
    
  
    # Prepare sequences for prediction
    sequence_length = 200  # or whatever sequence length your model expects
    x = []
    y = []
    
    # Convert scaled data to list for easier manipulation
    scaled_data = test_data_scaled.flatten().tolist()
    
    # Create sequences
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Reshape for LSTM [samples, time steps, features]
    if len(x) > 0:
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        
        # Make predictions
        predict = model.predict(x)
        
        # Inverse transform the predictions
        predict = scaler.inverse_transform(predict.reshape(-1, 1)).flatten()
        y_actual = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        
        # Limit data to selected number of days
        y_actual = y_actual[-prediction_days:]
        predict = predict[-prediction_days:]
        
        
        test_dates = data.index[-len(y_actual):]
        
       
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_actual, predict)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, predict)
        r2 = r2_score(y_actual, predict)
        
        # Display metrics
        st.subheader('Model Performance')
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric('MSE', f"{mse:.2f}")
        with metric_cols[1]:
            st.metric('RMSE', f"{rmse:.2f}")
        with metric_cols[2]:
            st.metric('MAE', f"{mae:.2f}")
        with metric_cols[3]:
            st.metric('R² Score', f"{r2:.4f}")
        
        # Create prediction plot
        fig_pred = plt.figure(figsize=(14, 7), facecolor=COLORS['background'])
        ax = plt.gca()
        
        # Plot actual test data
        ax.plot(test_dates, y_actual, 
                label='Actual Price', 
                color=COLORS['primary'], 
                linewidth=2)
                
        # Plot predicted data
        ax.plot(test_dates, predict, 
                label='Predicted Price', 
                color=COLORS['accent'], 
                linestyle='--',
                linewidth=2)
                
        # Add confidence interval if selected
        if confidence_level:
            confidence = confidence_level / 100.0
            residuals = y_actual - predict
            std_dev = np.std(residuals)
            margin = std_dev * (1.0 + confidence)
            
            ax.fill_between(test_dates, 
                          predict - margin, 
                          predict + margin,
                          color=COLORS['volume'],
                          alpha=0.3,
                          label=f'{confidence_level}% Confidence Interval')
        
        # Customize the plot
        ax.set_title(f'Stock Price Prediction for {stock.upper()}', 
                    fontsize=16, 
                    pad=20, 
                    color=COLORS['text'])
        ax.set_xlabel('Date', fontsize=12, labelpad=10, color=COLORS['text'])
        ax.set_ylabel('Price ($)', fontsize=12, labelpad=10, color=COLORS['text'])
        ax.grid(color=COLORS['grid'], linestyle='--', alpha=0.7)
        ax.legend(loc='upper left', frameon=True, facecolor='white')
        
        # Format x-axis dates
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot in Streamlit
        st.pyplot(fig_pred)
        
       
        st.subheader('Prediction Summary')
        pred_cols = st.columns(3)
        with pred_cols[0]:
            st.metric('Current Price', f"${y[-1]:.2f}")
        with pred_cols[1]:
            price_change = ((predict[-1] - y[-1]) / y[-1]) * 100
            st.metric('Predicted Price', 
                     f"${predict[-1][0]:.2f}", 
                     f"{price_change[0]:.2f}%")
        with pred_cols[2]:
            st.metric('Confidence', f"{confidence_level}%")
    else:
        st.warning("No prediction data available. Please ensure the model is properly loaded and data is available.")

# Add footer
st.markdown("""
---
<div style='text-align: center; color: #6c757d; padding: 20px 0;'>
    Made with ❤️ by Anand
</div>
""", unsafe_allow_html=True)
