import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from datetime import date, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def prediction(stock, n_days,selected_option):
    a = selected_option + "_scaled"
    df = yf.download(stock, period='60d')
    df.reset_index(inplace=True)
    
    if df.empty:
        st.error("No data available for the selected stock.")
        return
    
    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler()
    df[a] = scaler.fit_transform(df[[selected_option]])
    
    # Prepare data for LSTM
    X = df[[a]].values
    Y = df[[a]].shift(-1).fillna(method='bfill').values
    
    x_train, x_test = X[:-n_days], X[-n_days:]
    y_train, y_test = Y[:-n_days], Y[-n_days:]
    
    # Reshape data for LSTM input
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    
    # Create LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=1)
    
    # Predict future values
    predicted_scaled = model.predict(x_test)
    
    # Inverse transform to get actual values
    predicted = scaler.inverse_transform(predicted_scaled)
    
    # Generate dates for plotting
    dates = [date.today() + timedelta(days=i) for i in range(n_days)]
    
    # Create and format the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=predicted.flatten(), mode='lines+markers', name='predicted'))
    fig.update_layout(title="Predicted " + selected_option + " Price of Next " + str(n_days) + " Days",
                      xaxis_title="Date", yaxis_title=selected_option + " Price")
    
    st.plotly_chart(fig)

def plot_stock_history(stock, d,date1,date2):
    today = datetime.today().date()
    if date1>=today or date2>=today:
        st.error("Start and end dates cannot be today's date or future dates.")
        return

    stock1 = yf.Ticker(stock)
    df = stock1.history(period=d)

    if df.empty:
        st.error("No data available for the selected stock.")
        return
    # Create a Candlestick trace with dates as x-axis
    candlestick = go.Candlestick(x=df.index, low=df['Low'], high=df['High'], close=df['Close'], open=df['Open'])

    # Create a Figure with the Candlestick trace
    fig = go.Figure(data=[candlestick])

    # Update the layout to include title and axis labels
    fig.update_layout(title=f"Stock Price History for {stock}", xaxis_title="Date", yaxis_title="Price")

    # Show the Plotly chart in Streamlit
    st.plotly_chart(fig)

def calculate_and_display_evm(stock):
    if stock=="":
        st.error("provide stock symbol")
        return

    df = yf.download(stock, period='60d')

    if df.empty:
        st.error("No data available for the selected stock.")
        return

    df.index = pd.to_datetime(df.index)  # Convert index to datetime format
    df['EVM'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Volatility'] = df['High'] - df['Low']
    df['EVM_MA'] = df['EVM'].rolling(window=10).mean()
    df['Bollinger_Upper'] = df['EVM_MA'] + 2 * df['Volatility']
    df['Bollinger_Lower'] = df['EVM_MA'] - 2 * df['Volatility']

    # Display EVM data
    st.write("### Expected Value of Maximum (EVM) Analysis")
    st.write(df)
    
    # Create and display the graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Stock'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EVM'], mode='lines', name='EVM'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Upper'], mode='lines', name='Bollinger Upper'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Lower'], mode='lines', name='Bollinger Lower'))
    fig.update_layout(title="Expected Value of Maximum (EVM) Analysis", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig)

def compare_stocks(s1,s2,s,e):
    today = datetime.today().date()
    if s >= today or e >= today:
        st.error("Start and end dates cannot be today's date or future dates.")
        return
    
    if s >= e:
        st.error("Start date should be before the end date.")
        return
        
    stock_data1 = yf.download(s1, start=s, end=e)
    stock_data2 = yf.download(s2, start=s, end=e)

    if stock_data1.empty or stock_data2.empty:
        st.error("No data available for the selected stock.")
        return

    # Create and display the comparison graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['Close'], mode='lines', name=s1))
    fig.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['Close'], mode='lines', name=s2))
    fig.update_layout(title="Stock Price Comparison", xaxis_title="Date", yaxis_title="Close Price")
    st.plotly_chart(fig)
