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

stock = st.text_input('Enter Stock Symnbol', 'GOOG')
start = '2012-01-01'
end = '2025-12-21'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_train], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)


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
    
  
    x = []
    y = []
    
  
    data_test_scale_list = data_test_scale.tolist()
    
    for i in range(200, len(data_test_scale_list)):
        x.append(data_test_scale_list[i-200:i])
        y.append(data_test_scale_list[i][0])  
    
    x, y = np.array(x), np.array(y)
    
    #
    if len(x) > 0:
        predict = model.predict(x)
        scale = 1/scaler.scale_[0]  # Get the scale factor for the first feature
        predict = predict * scale
        y = y * scale
        
        # Limit data to selected number of days
        y = y[-prediction_days:]
        predict = predict[-prediction_days:]
        dates = data.index[-prediction_days:]
        
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y, predict)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predict)
        r2 = r2_score(y, predict)
        
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
        
        # Plot historical data
        ax.plot(dates, y, 
                label='Historical Price', 
                color=COLORS['primary'], 
                linewidth=2,
                alpha=0.9)
        
        #
        ax.plot(dates, predict, 
                label='Predicted Price', 
                color=COLORS['secondary'], 
                linestyle='--', 
                linewidth=2,
                alpha=0.9)
        
        
        confidence = (100 - confidence_level) / 200  
        std_dev = np.std(predict)
        upper_bound = predict * (1 + confidence)
        lower_bound = predict * (1 - confidence)
        
        ax.fill_between(dates, 
                       lower_bound.flatten(), 
                       upper_bound.flatten(), 
                       color=COLORS['secondary'], 
                       alpha=0.2,
                       label=f'{confidence_level}% Confidence Interval')
        
        # Customize the plot
        ax.set_title(f'{stock} - Price Prediction', 
                    fontsize=16, 
                    pad=20, 
                    color=COLORS['text'])
        ax.set_xlabel('Date', fontsize=12, color=COLORS['text'])
        ax.set_ylabel('Price', fontsize=12, color=COLORS['text'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:,.2f}".format(x)))
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.4, color=COLORS['grid'])
        
   
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.tight_layout()
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
