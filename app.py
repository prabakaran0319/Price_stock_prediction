import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved optimized model and scaler
with open('optimized_random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Title of the application
st.title("📈Stock Price Prediction")

# Layout with columns for side-by-side input fields (4 columns) and optimal spacing
col1, col2, col3 ,col4 = st.columns([1, 1, 1, 1])

with col1:
    company_amazon = st.radio("Is the company Amazon?", options=[0, 1], index=1)
    company_apple = st.radio("Is the company Apple?", options=[0, 1], index=1)
    company_facebook = st.radio("Is the company Facebook?", options=[0, 1], index=1)

with col2:
    company_google = st.radio("Is the company Google?", options=[0, 1], index=1)
    company_netflix = st.radio("Is the company Netflix?", options=[0, 1], index=1)
    volume = st.number_input("Enter Volume", min_value=0.0, value=1000000.0, step=1.0)

with col3:
    market_cap = st.number_input("Enter Market Cap", min_value=0.0, value=500000000.0, step=1000000.0)
    pe_ratio = st.number_input("Enter PE Ratio", min_value=0.0, value=10.0, step=0.1)
    eps = st.number_input("Enter EPS", min_value=0.0, value=2.0, step=0.1)

with col4:
    forward_pe = st.number_input("Enter Forward PE", min_value=0.0, value=15.0, step=0.1)
    net_income = st.number_input("Enter Net Income", min_value=0.0, value=10000000.0, step=1000000.0)
    debt_to_equity = st.number_input("Enter Debt to Equity", min_value=0.0, value=0.5, step=0.01)

# Second row of input fields (using 4 columns)
col5, col6, col7, col8 = st.columns([1, 1, 1, 1])

with col5:
    roe = st.number_input("Enter Return on Equity (ROE)", min_value=0.0, value=0.15, step=0.01)
    beta_5y = st.number_input("Enter Beta (5Y)", min_value=0.0, value=1.0, step=0.01)

with col6:
    current_ratio = st.number_input("Enter Current Ratio", min_value=0.0, value=1.5, step=0.1)
    dividends_paid = st.number_input("Enter Dividends Paid", min_value=0.0, value=1000000.0, step=100000.0)

with col7:
    dividend_yield = st.number_input("Enter Dividend Yield", min_value=0.0, value=2.5, step=0.1)
    quarterly_revenue_growth = st.number_input("Enter Quarterly Revenue Growth", min_value=0.0, value=0.05, step=0.01)

with col8:
    target_price = st.number_input("Enter Target Price", min_value=0.0, value=1000.0, step=10.0)
    free_cash_flow = st.number_input("Enter Free Cash Flow", min_value=0.0, value=50000000.0, step=1000000.0)

# Third row of input fields (using 4 columns)
col9, col10, col11, col12 = st.columns([1, 1, 1, 1])

with col9:
    operating_margin = st.number_input("Enter Operating Margin", min_value=0.0, value=0.2, step=0.01)
    profit_margin = st.number_input("Enter Profit Margin", min_value=0.0, value=0.1, step=0.01)

with col10:
    quick_ratio = st.number_input("Enter Quick Ratio", min_value=0.0, value=1.2, step=0.1)
    price_to_book_ratio = st.number_input("Enter Price to Book Ratio", min_value=0.0, value=3.0, step=0.1)

with col11:
    enterprise_value = st.number_input("Enter Enterprise Value", min_value=0.0, value=100000000.0, step=1000000.0)
    total_debt = st.number_input("Enter Total Debt", min_value=0.0, value=50000000.0, step=1000000.0)

with col12:
    annual_dividend_rate = st.number_input("Enter Annual Dividend Rate", min_value=0.0, value=1.0, step=0.1)
    timestamp = st.number_input("Enter Timestamp", min_value=0.0, value=1667270400.0, step=100000.0)
    
# Define feature names (make sure these match the features used during model training)
feature_names = [
    'Company_Amazon', 'Company_Apple', 'Company_Facebook', 'Company_Google',
    'Company_Netflix', 'Volume', 'Market Cap', 'PE Ratio', 'EPS', 'Forward PE',
    'Net Income', 'Debt to Equity', 'Return on Equity (ROE)', 'Current Ratio',
    'Dividends Paid', 'Dividend Yield', 'Quarterly Revenue Growth', 'Target Price',
    'Free Cash Flow', 'Operating Margin', 'Profit Margin', 'Quick Ratio', 'Price to Book Ratio',
    'Enterprise Value', 'Total Debt', 'Beta (5Y)', 'Annual Dividend Rate', 'Timestamp'
]

# Collect inputs into a single array
user_input = np.array([[ 
    company_amazon, company_apple, company_facebook, company_google, company_netflix,
    volume, market_cap, pe_ratio, eps, forward_pe, net_income, debt_to_equity, roe,
    current_ratio, dividends_paid, dividend_yield, quarterly_revenue_growth, target_price,
    free_cash_flow, operating_margin, profit_margin, quick_ratio, price_to_book_ratio,
    enterprise_value, total_debt, beta_5y, annual_dividend_rate, timestamp
]])

# Convert user input to a DataFrame with feature names for compatibility with the scaler
user_input_df = pd.DataFrame(user_input, columns=feature_names)

# Scale the input using the same scaler used during training
user_input_scaled = scaler.transform(user_input_df)

# Make prediction using the trained model
prediction = model.predict(user_input_scaled)

# Display the prediction
st.write(f"Predicted Stock Close Price: {prediction[0]:.2f}")

# Footer with your name
st.markdown("---")
st.markdown("<p style='text-align: right; font-size: 16px;'>Developed by <b>D. Prabakaran</b></p>", unsafe_allow_html=True)
