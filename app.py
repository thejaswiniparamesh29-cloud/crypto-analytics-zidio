import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Crypto Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 1rem; }
    h1 { color: #00d4ff; text-align: center; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #00d4ff33;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Cryptocurrency Time Series Analytics Dashboard")
st.markdown("<p style='text-align:center; color:#aaa;'>Zidio Development — Data Analytics Internship Project</p>", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
coin = st.sidebar.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"])
period = st.sidebar.selectbox("Historical Period", ["1y", "2y", "3y", "5y"], index=1)
model_choice = st.sidebar.multiselect("Forecasting Models", ["ARIMA", "Prophet"], default=["ARIMA", "Prophet"])
forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)

@st.cache_data(ttl=3600)
def load_data(symbol, period):
    df = yf.Ticker(symbol).history(period=period)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[['Open','High','Low','Close','Volume']].ffill().bfill()
    df['Returns']       = df['Close'].pct_change()
    df['MA_7']          = df['Close'].rolling(7).mean()
    df['MA_30']         = df['Close'].rolling(30).mean()
    df['MA_90']         = df['Close'].rolling(90).mean()
    df['Volatility_30'] = df['Returns'].rolling(30).std()
    df['Upper_BB']      = df['MA_30'] + df['Close'].rolling(30).std() * 2
    df['Lower_BB']      = df['MA_30'] - df['Close'].rolling(30).std() * 2
    delta               = df['Close'].diff()
    gain                = delta.where(delta > 0, 0).rolling(14).mean()
    loss                = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI']           = 100 - (100 / (1 + gain / loss))
    return df.dropna()

with st.spinner("📥 Fetching live data..."):
    df = load_data(coin, period)

# ── KPI Row ───────────────────────────────────────────────
latest  = df['Close'].iloc[-1]
prev    = df['Close'].iloc[-2]
chg_pct = (latest - prev) / prev * 100
high52  = df['Close'].rolling(365).max().iloc[-1]
low52   = df['Close'].rolling(365).min().iloc[-1]
vol     = df['Volatility_30'].iloc[-1] * 100
rsi_val = df['RSI'].iloc[-1]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💰 Price",         f"${latest:,.2f}",  f"{chg_pct:+.2f}%")
c2.metric("📈 52W High",      f"${high52:,.2f}")
c3.metric("📉 52W Low",       f"${low52:,.2f}")
c4.metric("⚡ Volatility",    f"{vol:.2f}%")
c5.metric("📡 RSI (14)",      f"{rsi_val:.1f}",
          "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))

st.divider()

# ── Tab Layout ────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price Analysis", "📉 Volatility", "🔮 Forecasting", "🔍 EDA", "📋 Data"])

# ── TAB 1: Price ──────────────────────────────────────────
with tab1:
    st.subheader(f"{coin} Price Chart")
    show_bb  = st.checkbox("Show Bollinger Bands", True)
    show_mas = st.checkbox("Show Moving Averages", True)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='OHLC'
    ))
    if show_mas:
        for col, color, label in [('MA_7','orange','MA 7'),('MA_30','red','MA 30'),('MA_90','lime','MA 90')]:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=label, line=dict(color=color, width=1.2)))
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_BB'], name='Upper BB',
                                  line=dict(color='cyan', dash='dash', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_BB'], name='Lower BB',
                                  line=dict(color='cyan', dash='dash', width=1),
                                  fill='tonexty', fillcolor='rgba(0,212,255,0.05)'))
    fig.update_layout(template='plotly_dark', height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(df.tail(90), x=df.tail(90).index, y='Volume',
                      title='Trading Volume (Last 90 Days)',
                      color='Volume', color_continuous_scale='Blues',
                      template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.histogram(df, x='Returns', nbins=80,
                             title='Daily Return Distribution',
                             template='plotly_dark',
                             color_discrete_sequence=['#00d4ff'])
        fig3.add_vline(x=df['Returns'].mean(), line_dash='dash', line_color='red',
                       annotation_text=f"Mean: {df['Returns'].mean():.4f}")
        st.plotly_chart(fig3, use_container_width=True)

# ── TAB 2: Volatility ─────────────────────────────────────
with tab2:
    st.subheader("Volatility & Risk Analysis")

    roll_max  = df['Close'].cummax()
    drawdown  = (df['Close'] - roll_max) / roll_max * 100
    ann_vol   = df['Volatility_30'] * np.sqrt(365) * 100
    sharpe    = (df['Returns'].rolling(30).mean() * 365) / (df['Volatility_30'] * np.sqrt(365))

    col1, col2, col3 = st.columns(3)
    col1.metric("Max Drawdown",    f"{drawdown.min():.2f}%")
    col2.metric("Avg Annual Vol",  f"{ann_vol.mean():.2f}%")
    col3.metric("Avg Sharpe",      f"{sharpe.mean():.3f}")

    fig = make_subplots(rows=3, cols=1, subplot_titles=[
        'Annualized Volatility (%)', 'Drawdown from Peak (%)', 'Rolling Sharpe Ratio (30d)'
    ], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=df.index, y=ann_vol,  fill='tozeroy', name='Ann. Vol',  line=dict(color='#00d4ff')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=drawdown, fill='tozeroy', name='Drawdown',  line=dict(color='red')),     row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sharpe,   name='Sharpe', line=dict(color='lime')), row=3, col=1)
    fig.add_hline(y=0, line_color='white', row=3, col=1)
    fig.add_hline(y=1, line_dash='dash', line_color='green', row=3, col=1)
    fig.update_layout(template='plotly_dark', height=700, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Simulated Sentiment
    st.subheader("📰 Market Sentiment (Simulated)")
    st.info("In production, replace with real Twitter/Reddit NLP sentiment scores.")
    np.random.seed(42)
    n         = len(df)
    sentiment = pd.Series(
        np.convolve(np.random.randn(n), np.ones(7)/7, mode='same'),
        index=df.index
    )
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=df.index, y=sentiment,
        fill='tozeroy', name='Sentiment',
        line=dict(color='#00d4ff', width=1),
        fillcolor='rgba(0,212,255,0.15)'
    ))
    fig_s.add_hline(y=0, line_color='white', line_dash='dash')
    fig_s.update_layout(template='plotly_dark', height=300,
                         title='Simulated Market Sentiment Score')
    st.plotly_chart(fig_s, use_container_width=True)

# ── TAB 3: Forecasting ────────────────────────────────────
with tab3:
    st.subheader(f"🔮 {forecast_days}-Day Price Forecast")

    if not model_choice:
        st.warning("Please select at least one model from the sidebar.")
    else:
        results = {}

        if "ARIMA" in model_choice:
            with st.spinner("⚙️ Training ARIMA..."):
                try:
                    series       = df['Close']
                    test_days    = 30
                    train, test  = series.iloc[:-test_days], series.iloc[-test_days:]
                    arima_model  = ARIMA(train, order=(5, 1, 0)).fit()
                    test_forecast = arima_model.forecast(steps=test_days)
                    full_model   = ARIMA(series, order=(5, 1, 0)).fit()
                    future_vals  = full_model.forecast(steps=forecast_days)
                    future_dates = pd.date_range(start=series.index[-1] + timedelta(days=1), periods=forecast_days)
                    mae  = mean_absolute_error(test, test_forecast)
                    rmse = np.sqrt(mean_squared_error(test, test_forecast))
                    mape = np.mean(np.abs((test.values - test_forecast.values) / test.values)) * 100
                    results['ARIMA'] = {
                        'future_dates': future_dates, 'future_vals': future_vals,
                        'mae': mae, 'rmse': rmse, 'mape': mape
                    }
                    st.success(f"✅ ARIMA — MAE: ${mae:,.0f} | RMSE: ${rmse:,.0f} | MAPE: {mape:.2f}%")
                except Exception as e:
                    st.error(f"ARIMA failed: {e}")

        if "Prophet" in model_choice:
            with st.spinner("⚙️ Training Prophet..."):
                try:
                    prophet_df           = df[['Close']].reset_index()
                    prophet_df.columns   = ['ds', 'y']
                    prophet_df['ds']     = pd.to_datetime(prophet_df['ds'])
                    test_days            = 30
                    train_p              = prophet_df.iloc[:-test_days]
                    test_p               = prophet_df.iloc[-test_days:]
                    m_test               = Prophet(daily_seasonality=True, weekly_seasonality=True,
                                                   yearly_seasonality=True, changepoint_prior_scale=0.05)
                    m_test.fit(train_p)
                    fut_test             = m_test.make_future_dataframe(periods=test_days)
                    fc_test              = m_test.predict(fut_test).tail(test_days)
                    mae                  = mean_absolute_error(test_p['y'].values, fc_test['yhat'].values)
                    rmse                 = np.sqrt(mean_squared_error(test_p['y'].values, fc_test['yhat'].values))
                    mape                 = np.mean(np.abs((test_p['y'].values - fc_test['yhat'].values) / test_p['y'].values)) * 100
                    m_full               = Prophet(daily_seasonality=True, weekly_seasonality=True,
                                                   yearly_seasonality=True, changepoint_prior_scale=0.05)
                    m_full.fit(prophet_df)
                    fut_full             = m_full.make_future_dataframe(periods=forecast_days)
                    fc_full              = m_full.predict(fut_full)
                    future_prophet       = fc_full.tail(forecast_days)
                    results['Prophet'] = {
                        'future_dates': future_prophet['ds'],
                        'future_vals':  future_prophet['yhat'],
                        'lower':        future_prophet['yhat_lower'],
                        'upper':        future_prophet['yhat_upper'],
                        'mae': mae, 'rmse': rmse, 'mape': mape
                    }
                    st.success(f"✅ Prophet — MAE: ${mae:,.0f} | RMSE: ${rmse:,.0f} | MAPE: {mape:.2f}%")
                except Exception as e:
                    st.error(f"Prophet failed: {e}")

        # Combined Forecast Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index[-90:], y=df['Close'].iloc[-90:],
            name='Historical', line=dict(color='white', width=2)
        ))
        colors_map = {'ARIMA': 'red', 'Prophet': 'orange'}
        for model_name, res in results.items():
            fig.add_trace(go.Scatter(
                x=res['future_dates'], y=res['future_vals'],
                name=f'{model_name} Forecast',
                line=dict(color=colors_map[model_name], dash='dash', width=2)
            ))
            if 'lower' in res:
                fig.add_trace(go.Scatter(
                    x=list(res['future_dates']) + list(res['future_dates'])[::-1],
                    y=list(res['upper']) + list(res['lower'])[::-1],
                    fill='toself', fillcolor='rgba(255,165,0,0.1)',
                    line=dict(color='rgba(0,0,0,0)'), name='Prophet CI'
                ))
        fig.update_layout(
            template='plotly_dark', height=500,
            title=f'{coin} — {forecast_days}-Day Price Forecast',
            xaxis_title='Date', yaxis_title='Price (USD)'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Model Comparison Table
        if results:
            st.subheader("📊 Model Comparison")
            comp_data = {
                'Model': list(results.keys()),
                'MAE ($)':  [f"${r['mae']:,.0f}"  for r in results.values()],
                'RMSE ($)': [f"${r['rmse']:,.0f}" for r in results.values()],
                'MAPE (%)': [f"{r['mape']:.2f}%"  for r in results.values()],
            }
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

# ── TAB 4: EDA ────────────────────────────────────────────
with tab4:
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        corr_cols = ['Close','Volume','Returns','Volatility_30','RSI','MA_7','MA_30']
        fig_corr  = px.imshow(df[corr_cols].corr(), text_auto='.2f',
                              color_continuous_scale='RdBu_r',
                              title='Feature Correlation Heatmap',
                              template='plotly_dark')
        st.plotly_chart(fig_corr, use_container_width=True)
    with col2:
        fig_box = px.box(df.resample('ME')['Returns'].apply(list).explode().reset_index(),
                         x='Date', y='Returns',
                         title='Monthly Return Distribution',
                         template='plotly_dark',
                         color_discrete_sequence=['#00d4ff'])
        st.plotly_chart(fig_box, use_container_width=True)

    fig_rsi = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=['Close Price', 'RSI (14)'],
                             vertical_spacing=0.05)
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#00d4ff')), row=1, col=1)
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'],   name='RSI',   line=dict(color='orange')),  row=2, col=1)
    fig_rsi.add_hline(y=70, line_dash='dash', line_color='red',   annotation_text='Overbought', row=2, col=1)
    fig_rsi.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='Oversold',   row=2, col=1)
    fig_rsi.update_layout(template='plotly_dark', height=500, showlegend=False)
    st.plotly_chart(fig_rsi, use_container_width=True)

# ── TAB 5: Raw Data ───────────────────────────────────────
with tab5:
    st.subheader("📋 Raw Data")
    st.dataframe(df.tail(100).sort_index(ascending=False).style.format({
        'Open': '${:,.2f}', 'High': '${:,.2f}', 'Low': '${:,.2f}',
        'Close': '${:,.2f}', 'Returns': '{:.4f}', 'RSI': '{:.1f}'
    }), use_container_width=True)

    csv = df.to_csv().encode('utf-8')
    st.download_button("⬇️ Download Full Data as CSV", csv, f"{coin}_data.csv", "text/csv")

st.divider()
st.markdown("<p style='text-align:center; color:#555;'>Zidio Development | Data Analytics Internship Project | Built with Streamlit + Plotly</p>", unsafe_allow_html=True)
