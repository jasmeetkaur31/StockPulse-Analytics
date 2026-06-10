import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from prophet import Prophet
# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="StockPulse Analytics",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #1f2937;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

[data-testid="stMetricValue"] {
    font-size: 30px;
    font-weight: bold;
}

[data-testid="stMetricLabel"] {
    font-size: 16px;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📊 StockPulse Analytics")

st.sidebar.info("""
Analyze stock performance,
company fundamentals,
revenue growth,
and market trends.
""")

# --------------------------------------------------
# COMPANY SELECTION
# --------------------------------------------------

companies = {
    "Tesla": "TSLA",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "NVIDIA": "NVDA"
}


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
# 📈 StockPulse Analytics
### Real-Time Financial Intelligence Platform
""")

col_left, col_right = st.columns([3, 1])

with col_left:
    st.markdown(
        "Analyze stock performance, revenue growth, and market trends."
    )

with col_right:
    company = st.selectbox(
        "Company",
        list(companies.keys())
    )

ticker = companies[company]
stock = yf.Ticker(ticker)

st.markdown(f"### Currently Analyzing: {company} ({ticker})")

@st.cache_data(ttl=3600)
def load_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="5y")
    info = stock.info
    return data, info

# --------------------------------------------------
# STOCK DATA
# --------------------------------------------------
stock = yf.Ticker(ticker)
data, info = load_stock_data(ticker)

data["Returns"] = data["Close"].pct_change() * 100

data["MA50"] = data["Close"].rolling(50).mean()

data["MA200"] = data["Close"].rolling(200).mean()

volatility = data["Returns"].std()

if volatility < 1:
    risk = "🟢 Low"
elif volatility < 2:
    risk = "🟡 Moderate"
else:
    risk = "🔴 High"

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Current Price",
    f"${info.get('currentPrice', 'N/A')}"
)

market_cap = info.get("marketCap", 0)

if market_cap >= 1e12:
    market_cap_display = f"${market_cap/1e12:.2f}T"
else:
    market_cap_display = f"${market_cap/1e9:.2f}B"

col2.metric(
    "Market Cap",
    market_cap_display
)

col3.metric(
    "52W High",
    f"${info.get('fiftyTwoWeekHigh', 'N/A')}"
)

col4.metric(
    "Volatility",
    f"{volatility:.2f}%"
)

st.markdown(f"### Risk Level: {risk}")
# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Overview",
        "💰 Financials",
        "🏆 Comparison",
        "📊 Forecasting"
    ]
)

# ==================================================
# OVERVIEW
# ==================================================

with tab1:

    st.subheader("Stock Price Trend")

    fig1 = px.area(
        data,
        x=data.index,
        y="Close",
        title=f"{company} Stock Price"
    )

    fig1.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.subheader("Daily Returns")

    fig2 = px.line(
        data,
        x=data.index,
        y="Returns"
    )

    fig2.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Moving Average Analysis")

    fig3 = px.line(
        data,
        x=data.index,
        y=["Close", "MA50", "MA200"]
    )

    fig3.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# ==================================================
# FINANCIALS
# ==================================================

with tab2:

    st.subheader("Financial Performance")

    try:

        financials = stock.quarterly_financials

        if not financials.empty:

            if "Total Revenue" in financials.index:

                revenue = financials.loc["Total Revenue"]

                revenue_df = pd.DataFrame({
                    "Quarter": revenue.index.astype(str),
                    "Revenue": revenue.values
                })

                revenue_df = revenue_df.sort_values(
                    by="Quarter"
                )

                fig4 = px.bar(
                    revenue_df,
                    x="Quarter",
                    y="Revenue",
                    title=f"{company} Quarterly Revenue"
                )

                fig4.update_layout(
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig4,
                    use_container_width=True
                )

                if len(revenue) >= 2:

                    latest = revenue.iloc[0]
                    previous = revenue.iloc[1]

                    growth = (
                        (latest - previous)
                        / previous
                    ) * 100

                    st.metric(
                        "Quarterly Revenue Growth",
                        f"{growth:.2f}%"
                    )

                revenue_df["Revenue Growth %"] = (
                    revenue_df["Revenue"].pct_change() * 100
                )

                fig5 = px.line(
                    revenue_df,
                    x="Quarter",
                    y="Revenue Growth %",
                    title=f"{company} Revenue Growth Trend"
                )

                fig5.update_layout(
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig5,
                    use_container_width=True
                )

            else:
                st.warning("Revenue data unavailable.")

    except Exception as e:

        st.warning(
            f"Financial data unavailable: {e}"
        )

# ==================================================
# COMPARISON
# ==================================================

with tab3:

    comparison_data = []

    for name, symbol in companies.items():

        company_stock = yf.Ticker(symbol)

        company_info = company_stock.info

        comparison_data.append({
            "Company": name,
            "Current Price":
                company_info.get("currentPrice"),

            "Market Cap (B)":
                round(
                    company_info.get(
                        "marketCap",
                        0
                    ) / 1e9,
                    2
                )
        })

    comparison_df = pd.DataFrame(
        comparison_data
    )

    comparison_df = comparison_df.sort_values(
        by="Market Cap (B)",
        ascending=False
    )

    st.subheader(
        "Top Companies by Market Cap"
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    fig6 = px.bar(
        comparison_df,
        x="Company",
        y="Market Cap (B)",
        title="Market Capitalization Comparison"
    )

    fig6.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

# ==================================================
# FORECASTING
# ==================================================
with tab4:

    st.subheader("30-Day Stock Price Forecast")

    forecast_df = (
        data.reset_index()[["Date", "Close"]]
        .rename(columns={
            "Date": "ds",
            "Close": "y"
        })
    )

    forecast_df["ds"] = forecast_df["ds"].dt.tz_localize(None)

    forecast_df = forecast_df.tail(730)

    model = Prophet(
        daily_seasonality=True
    )

    model.fit(forecast_df)

    future = model.make_future_dataframe(
        periods=30
    )

    forecast = model.predict(future)

    fig7 = px.line(
        forecast,
        x="ds",
        y="yhat",
        title=f"{company} 30-Day Forecast"
    )

    fig7.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig7,
        use_container_width=True
    )

latest_prediction = forecast["yhat"].iloc[-1]

current_price = data["Close"].iloc[-1]

expected_change = (
    (latest_prediction - current_price)
    / current_price
) * 100

st.metric(
    "Expected 30-Day Change",
    f"{expected_change:.2f}%"
)

st.info(
    """
    Forecasts are based on historical price patterns only.
    They should not be considered investment advice.
    """
)

# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------

st.download_button(
    label="📥 Download Stock Data",
    data=data.to_csv().encode("utf-8"),
    file_name=f"{ticker}_stock_data.csv",
    mime="text/csv"
)