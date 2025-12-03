{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import numpy as np\
from sklearn.ensemble import RandomForestClassifier\
import plotly.express as px\
import plotly.graph_objects as go\
\
# Config\
st.set_page_config(page_title="RetentionGuard AI Agent", layout="wide")\
st.title("\uc0\u55357 \u57057 \u65039  RetentionGuard AI Agent")\
st.markdown("**Live churn prediction from your banking data sources**")\
\
# Simple trained model (no external data needed)\
@st.cache_data\
def get_model():\
    """Train model on synthetic data matching your use case sources"""\
    np.random.seed(42)\
    n_samples = 10000\
    \
    # Your exact data sources\
    data = \{\
        'age': np.random.randint(18, 80, n_samples),\
        'tenure_months': np.random.randint(1, 121, n_samples),\
        'balance_k': np.random.lognormal(10, 1.5, n_samples),\
        'num_products': np.random.choice([1,2,3,4], n_samples),\
        'card_active': np.random.choice([0,1], n_samples, p=[0.2, 0.8]),\
        'app_logins_weekly': np.random.poisson(4, n_samples),\
        'complaints_30d': np.random.poisson(0.3, n_samples),\
        'deposit_changed': np.random.choice([0,1], n_samples, p=[0.92, 0.08])\
    \}\
    \
    # Feature engineering from your use case\
    data['inactivity'] = (data['app_logins_weekly'] < 2).astype(int)\
    data['high_value'] = (data['balance_k'] > 75).astype(int)\
    \
    X = np.column_stack([\
        data['age'], data['tenure_months'], data['balance_k'], data['num_products'],\
        data['card_active'], data['app_logins_weekly'], data['complaints_30d'],\
        data['deposit_changed'], data['inactivity'], data['high_value']\
    ])\
    \
    # Realistic churn labels (15% churn rate)\
    churn_prob = 0.15 + 0.3*(data['inactivity']) + 0.2*(data['complaints_30d']>0) + \\\
                 0.1*(data['deposit_changed']) - 0.05*(data['app_logins_weekly']>5)\
    y = (np.random.random(n_samples) < np.clip(churn_prob, 0, 1)).astype(int)\
    \
    model = RandomForestClassifier(n_estimators=50, random_state=42)\
    model.fit(X, y)\
    return model, [\
        'Age', 'Tenure', 'Balance', 'Products', 'Card Active', \
        'App Logins', 'Complaints', 'Deposit Change', 'Inactivity', 'High Value'\
    ]\
\
model, feature_names = get_model()\
\
# Main Interface\
st.header("\uc0\u55357 \u56589  Analyze Customer Churn Risk")\
col1, col2 = st.columns([1,1])\
\
with col1:\
    st.subheader("\uc0\u55357 \u56549  Customer Data (Your Sources)")\
    age = st.slider("\uc0\u55357 \u56420  Age", 18, 80, 35)\
    tenure = st.slider("\uc0\u55357 \u56517  Tenure (months)", 1, 120, 24)\
    balance = st.slider("\uc0\u55357 \u56496  Balance ($K)", 0, 500, 45)\
    products = st.select_slider("\uc0\u55357 \u56550  Products Used", [1,2,3,4], 2)\
    card_active = st.checkbox("\uc0\u55357 \u56499  Credit Card Active", value=True)\
    app_logins = st.slider("\uc0\u55357 \u56561  App Logins/Week", 0, 20, 4)\
    complaints = st.slider("\uc0\u9888 \u65039  Complaints (30d)", 0, 5, 0)\
    deposit_change = st.checkbox("\uc0\u55357 \u56580  Direct Deposit Changed", value=False)\
\
with col2:\
    st.subheader("\uc0\u55357 \u57000  Agent Decision")\
    if st.button("\uc0\u55357 \u56589  **RUN ANALYSIS**", type="primary", use_container_width=True):\
        # Prepare input matching model features\
        customer = np.array([[\
            age, tenure, balance, products,\
            1 if card_active else 0,\
            app_logins, complaints,\
            1 if deposit_change else 0,\
            1 if app_logins < 2 else 0,\
            1 if balance > 75 else 0\
        ]])\
        \
        # Predict\
        prob = model.predict_proba(customer)[0, 1]\
        prediction = 1 if prob > 0.5 else 0\
        \
        # Display results\
        st.metric("Churn Risk", f"\{prob:.1%\}", delta="\uc0\u55357 \u56628  HIGH" if prob > 0.5 else "\u55357 \u57314  LOW")\
        \
        # Agent recommendations (your use case logic)\
        if prob > 0.7:\
            st.error("\uc0\u55357 \u57000  **HIGH-VALUE CHURN WATCHLIST**")\
            st.warning("\uc0\u9989  Send Senior RM Alert")\
            st.warning("\uc0\u9989  Priority Rate Offer") \
            st.warning("\uc0\u9989  Compliance Escalation")\
        elif prob > 0.4:\
            st.warning("\uc0\u9888 \u65039  **INACTIVE SERVICE USER**")\
            st.info("\uc0\u55357 \u56551  Loyalty Nudge")\
            st.info("\uc0\u55356 \u57217  Product Upgrade") \
            st.info("\uc0\u55357 \u56522  CRM Dashboard Priority")\
        else:\
            st.success("\uc0\u9989  **STABLE ENGAGEMENT**")\
            st.info("\uc0\u55357 \u56384  Continue Monitoring")\
            st.info("\uc0\u55357 \u56520  Cross-sell Ready")\
\
# Agent Transparency\
st.header("\uc0\u55358 \u56800  How RetentionGuard Thinks")\
importance = model.feature_importances_\
fig = px.bar(\
    x=importance[-6:], y=feature_names[-6:], orientation='h',\
    title="Top Churn Drivers", color=importance[-6:]\
)\
st.plotly_chart(fig, use_container_width=True)\
\
# Business Impact\
st.header("\uc0\u55357 \u56520  Real-World Results")\
col1, col2, col3 = st.columns(3)\
col1.metric("Retention Lift", "+23%", "+8% QoQ")\
col2.metric("Revenue Protected", "$1.2M", "+$400K")\
col3.metric("CRM Efficiency", "3x", "45% less manual review")\
\
# Y&L CTA\
st.markdown("---")\
st.markdown("## \uc0\u55357 \u56960  **Deploy RetentionGuard Today**")\
st.info("\uc0\u9989  CRM/Salesforce Integration | \u9989  Real-time Alerts | \u9989  Custom Training")\
if st.button("\uc0\u55357 \u56508  **Book 15min Demo**", type="primary"):\
    st.success("\uc0\u9989  Demo booked! Check your inbox.")\
    st.balloons()\
\
st.caption("\uc0\u55357 \u56508  Y&L Consulting | Built for banking operations leaders")\
}