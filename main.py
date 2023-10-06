import streamlit as st
import function
import yfinance as yf

st.sidebar.title("Stock prediction")
st.sidebar.image('https://th.bing.com/th/id/OIP.oPT0GXDc3nEF6EAWfHZfwgHaE8?pid=ImgDet&w=1000&h=667&rs=1')

user_menu = st.sidebar.radio(
    'Select an Option',
    ('Predict Stock Price', 'Stock Price History', 'Average of Stock', 'Compare 2 Stocks')
)

if user_menu == "Predict Stock Price":
    selected_stock = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
    selected_days = st.number_input("Select days", 1)
    st.write(f"You entered: {selected_stock}")
    st.write(f"You selected: {selected_days}")
    options = ["Close", "Open", "High", "Low"]
    selected_option = st.selectbox("Select an option:", options)
    function.prediction(selected_stock.upper(), selected_days, selected_option)

if user_menu == "Stock Price History":
    selected_stock = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
    date1 = st.date_input("Select Starting Date")
    date2 = st.date_input("Select Ending Date")
    date_difference = (date2 - date1).days
    d = str(date_difference) + "d"
    function.plot_stock_history(selected_stock.upper(), d,date1,date2)

if user_menu == "Average of Stock":
    selected_stock = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
    function.calculate_and_display_evm(selected_stock.upper())

if user_menu == "Compare 2 Stocks":
    s1 = st.text_input("Enter Stock Symbol 1 (e.g., AAPL)", "AAPL")
    s2 = st.text_input("Enter Stock Symbol 2 (e.g., MSFT)", "MSFT")
    s_date = st.date_input("Start Date")
    e_date = st.date_input("End Date")
    function.compare_stocks(s1.upper(), s2.upper(), s_date, e_date)
