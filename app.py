import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from src.screener import screen_momentum_stocks, rank_momentum_stocks
import time

# Page configuration
st.set_page_config(
    page_title="Momentum Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-color: #0066cc;
            --secondary-color: #f63366;
            --success-color: #09ab3b;
            --warning-color: #ffa421;
        }
        
        /* Card styling */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .metric-card-green {
            background: linear-gradient(135deg, #09ab3b 0%, #07c784 100%);
        }
        
        .metric-card-red {
            background: linear-gradient(135deg, #f63366 0%, #ff6b9d 100%);
        }
        
        /* Header styling */
        .header-title {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #0066cc, #f63366);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        /* Table styling */
        .dataframe {
            font-size: 0.9rem;
        }
        
        /* Status badge */
        .status-pass {
            background-color: #09ab3b;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        
        .status-fail {
            background-color: #f63366;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)


class MockMarketDataGenerator:
    """Generate realistic mock market data for simulations."""
    
    def __init__(self, base_price=5.0, volatility=0.03, base_volume=100000, scenario_type='normal'):
        self.base_price = base_price
        self.volatility = volatility
        self.base_volume = base_volume
        self.scenario_type = scenario_type
        self.price_history = [base_price]
        self.volume_history = [base_volume]
    
    def generate_normal_day(self):
        daily_change = np.random.normal(0, self.volatility)
        new_price = self.price_history[-1] * (1 + daily_change)
        new_volume = self.base_volume * np.random.uniform(0.8, 1.2)
        self.price_history.append(new_price)
        self.volume_history.append(new_volume)
    
    def generate_spike_day(self, volume_multiplier=5.0, momentum_direction=1):
        daily_change = np.random.normal(0, self.volatility * 2) + (momentum_direction * 0.05)
        new_price = self.price_history[-1] * (1 + daily_change)
        new_volume = self.base_volume * volume_multiplier
        self.price_history.append(new_price)
        self.volume_history.append(new_volume)
    
    def get_dataframe(self, days=30):
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Close': self.price_history[-days:],
            'Volume': self.volume_history[-days:]
        })
        df['Open'] = df['Close'] * np.random.uniform(0.98, 1.02, len(df))
        df['High'] = df['Close'] * np.random.uniform(1.01, 1.05, len(df))
        df['Low'] = df['Close'] * np.random.uniform(0.95, 0.99, len(df))
        return df.set_index('Date')


def generate_scenario(scenario_name, num_days=30):
    """Generate mock data for selected scenario."""
    
    if scenario_name == "📊 Normal Consolidation":
        gen = MockMarketDataGenerator(base_price=4.5, volatility=0.02, base_volume=100000)
        for _ in range(num_days - 1):
            gen.generate_normal_day()
        return gen.get_dataframe(num_days), "normal"
    
    elif scenario_name == "🚀 Volume Spike + Uptrend":
        gen = MockMarketDataGenerator(base_price=3.5, volatility=0.02, base_volume=80000)
        for _ in range(num_days - 2):
            gen.generate_normal_day()
        gen.generate_spike_day(volume_multiplier=5.0, momentum_direction=1)
        return gen.get_dataframe(num_days), "bullish"
    
    elif scenario_name == "⚠️ Volume Spike + Downtrend":
        gen = MockMarketDataGenerator(base_price=6.0, volatility=0.02, base_volume=120000)
        for _ in range(num_days - 2):
            gen.generate_normal_day()
        gen.generate_spike_day(volume_multiplier=4.0, momentum_direction=-1)
        return gen.get_dataframe(num_days), "bearish"
    
    elif scenario_name == "💥 Penny Stock Explosion":
        gen = MockMarketDataGenerator(base_price=0.45, volatility=0.04, base_volume=500000)
        for i in range(num_days - 1):
            if i % 5 == 4:
                gen.generate_spike_day(volume_multiplier=3.0, momentum_direction=1)
            else:
                gen.generate_normal_day()
        gen.generate_spike_day(volume_multiplier=8.0, momentum_direction=1)
        return gen.get_dataframe(num_days), "extreme"
    
    elif scenario_name == "⛔ Price Out of Range":
        gen = MockMarketDataGenerator(base_price=27.0, volatility=0.02, base_volume=50000)
        for _ in range(num_days - 2):
            gen.generate_normal_day()
        gen.generate_spike_day(volume_multiplier=5.0, momentum_direction=1)
        return gen.get_dataframe(num_days), "out_of_range"
    
    else:
        gen = MockMarketDataGenerator(base_price=5.0, volatility=0.02, base_volume=100000)
        for _ in range(num_days - 1):
            gen.generate_normal_day()
        return gen.get_dataframe(num_days), "normal"


def calculate_metrics(df):
    """Calculate all screener metrics."""
    ema_9 = df['Close'].ewm(span=9, adjust=False).mean()
    current_price = df['Close'].iloc[-1]
    avg_volume = df['Volume'].iloc[:-1].mean()
    current_volume = df['Volume'].iloc[-1]
    rvol = current_volume / avg_volume if avg_volume > 0 else 0
    
    price_change_5d = ((current_price - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100) if len(df) > 5 else 0
    price_change_all = ((current_price - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)
    
    breakout_pct = ((current_price - ema_9.iloc[-1]) / ema_9.iloc[-1] * 100)
    
    return {
        'current_price': current_price,
        'ema_9': ema_9.iloc[-1],
        'rvol': rvol,
        'current_volume': current_volume,
        'avg_volume': avg_volume,
        'breakout_pct': breakout_pct,
        'price_change_5d': price_change_5d,
        'price_change_all': price_change_all,
        'ema_9_series': ema_9
    }


def check_filters(metrics, market_cap_max=500_000_000, price_min=1.0, price_max=15.0, rvol_threshold=3.0):
    """Check all screener filters."""
    price = metrics['current_price']
    rvol = metrics['rvol']
    ema = metrics['ema_9']
    
    checks = {
        'price_range': (price_min <= price <= price_max, f"${price_min}-${price_max}", f"${price:.2f}"),
        'volume': (rvol >= rvol_threshold, f">= {rvol_threshold}x", f"{rvol:.2f}x"),
        'breakout': (price > ema, f"> EMA ${ema:.2f}", f"${price:.2f}" if price > ema else "✗"),
    }
    
    all_passed = all(check[0] for check in checks.values())
    return checks, all_passed


# ============================================================================
# STREAMLIT APP STARTS HERE
# ============================================================================

# Header
st.markdown("# 📈 Momentum Stock Screener - Live Simulator")
st.markdown("### Real-time trading opportunity detection with visual analysis")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("---")
    
    scenario = st.selectbox(
        "🎯 Select Scenario",
        [
            "📊 Normal Consolidation",
            "🚀 Volume Spike + Uptrend",
            "⚠️ Volume Spike + Downtrend",
            "💥 Penny Stock Explosion",
            "⛔ Price Out of Range"
        ],
        help="Choose a trading scenario to simulate"
    )
    
    st.markdown("---")
    st.markdown("### Screener Settings")
    
    rvol_threshold = st.slider(
        "Relative Volume (RVOL) Threshold",
        min_value=1.5,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="How many times average volume must spike"
    )
    
    price_min = st.slider(
        "Minimum Price",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Minimum stock price filter"
    )
    
    price_max = st.slider(
        "Maximum Price",
        min_value=5.0,
        max_value=50.0,
        value=15.0,
        step=1.0,
        help="Maximum stock price filter"
    )
    
    st.markdown("---")
    num_days = st.slider(
        "Historical Days",
        min_value=10,
        max_value=60,
        value=30,
        step=5,
        help="Days of historical data"
    )
    
    st.markdown("---")
    animate = st.checkbox("🎬 Animate Results", value=True, help="Show animated transitions")


# Generate data
df = None
scenario_type = None

# Button to generate/update
col1, col2 = st.columns([2, 1])
with col1:
    if st.button("▶️ Run Simulation", use_container_width=True):
        with st.spinner("🔄 Generating market data..."):
            time.sleep(0.5)
            df, scenario_type = generate_scenario(scenario, num_days)
            st.session_state.df = df
            st.session_state.scenario_type = scenario_type

with col2:
    if st.button("🔄 Refresh", use_container_width=True):
        if 'df' in st.session_state:
            pass

# Load from session if available
if 'df' in st.session_state:
    df = st.session_state.df
    scenario_type = st.session_state.scenario_type
else:
    df, scenario_type = generate_scenario(scenario, num_days)
    st.session_state.df = df
    st.session_state.scenario_type = scenario_type

# Calculate metrics
metrics = calculate_metrics(df)
checks, all_passed = check_filters(
    metrics,
    price_min=price_min,
    price_max=price_max,
    rvol_threshold=rvol_threshold
)

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.markdown("---")

# Status banner
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if all_passed:
        st.success("✅ STOCK QUALIFIES - MOMENTUM DETECTED!", icon="✨")
    else:
        st.error("❌ STOCK REJECTED - Criteria Not Met", icon="⛔")

st.markdown("---")

# Key metrics row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Current Price",
        f"${metrics['current_price']:.4f}",
        delta=f"+{metrics['price_change_5d']:.2f}%" if metrics['price_change_5d'] > 0 else f"{metrics['price_change_5d']:.2f}%",
        delta_color="normal" if metrics['price_change_5d'] > 0 else "inverse"
    )

with col2:
    st.metric(
        "EMA-9",
        f"${metrics['ema_9']:.4f}",
        delta=f"{metrics['breakout_pct']:.2f}% away"
    )

with col3:
    color = "green" if metrics['rvol'] >= rvol_threshold else "red"
    st.metric(
        "RVOL",
        f"{metrics['rvol']:.2f}x",
        delta=f"Target: {rvol_threshold}x",
        delta_color="normal" if metrics['rvol'] >= rvol_threshold else "off"
    )

with col4:
    st.metric(
        "Current Volume",
        f"{metrics['current_volume']:,.0f}",
        delta=f"Avg: {metrics['avg_volume']:,.0f}"
    )

with col5:
    st.metric(
        "30-Day Change",
        f"{metrics['price_change_all']:.2f}%",
        delta="All-time" if len(df) == 30 else f"{len(df)}d"
    )

st.markdown("---")

# Charts section
tab1, tab2, tab3 = st.tabs(["📊 Price & Volume", "🔍 Filter Analysis", "📋 Raw Data"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price chart
        fig = go.Figure()
        
        # Close price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#0066cc', width=3),
            fill=None
        ))
        
        # EMA-9
        fig.add_trace(go.Scatter(
            x=df.index,
            y=metrics['ema_9_series'],
            mode='lines',
            name='EMA-9',
            line=dict(color='#f63366', width=2, dash='dash'),
        ))
        
        # Highlight breakout zone
        fig.add_hrect(
            y0=metrics['ema_9'], y1=metrics['current_price'],
            fillcolor="#09ab3b", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Breakout Zone", annotation_position="right",
        ) if metrics['current_price'] > metrics['ema_9'] else None
        
        fig.update_layout(
            title="Price Movement & EMA-9 Breakout",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Price Stats")
        st.metric("High", f"${df['High'].max():.4f}")
        st.metric("Low", f"${df['Low'].min():.4f}")
        st.metric("Open (Today)", f"${df['Open'].iloc[-1]:.4f}")
        st.metric("Close (Today)", f"${df['Close'].iloc[-1]:.4f}")
    
    # Volume chart
    colors = ['#09ab3b' if v >= metrics['avg_volume'] * 1.5 else '#667eea' for v in df['Volume']]
    fig_vol = go.Figure()
    
    fig_vol.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=colors,
        marker_line_color='rgba(0,0,0,0)'
    ))
    
    # Average line
    fig_vol.add_hline(
        y=metrics['avg_volume'],
        line_dash="dash",
        line_color="#f63366",
        annotation_text=f"Avg: {metrics['avg_volume']:,.0f}",
        annotation_position="right",
    )
    
    # Current volume
    fig_vol.add_hline(
        y=metrics['current_volume'],
        line_dash="dot",
        line_color="#09ab3b",
        annotation_text=f"Current: {metrics['current_volume']:,.0f}",
        annotation_position="right",
    )
    
    fig_vol.update_layout(
        title="Volume Analysis (Green = Spike)",
        xaxis_title="Date",
        yaxis_title="Volume",
        hovermode='x unified',
        template='plotly_dark',
        height=300,
        showlegend=False
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with tab2:
    st.markdown("### 🔍 Filter Criteria Analysis")
    
    filter_results = []
    for check_name, (passed, requirement, actual) in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        filter_results.append({
            'Filter': check_name.replace('_', ' ').title(),
            'Status': status,
            'Requirement': requirement,
            'Actual': actual
        })
    
    filter_df = pd.DataFrame(filter_results)
    
    # Color the table
    def highlight_status(val):
        if '✅' in val:
            return 'background-color: #09ab3b; color: white; font-weight: bold'
        elif '❌' in val:
            return 'background-color: #f63366; color: white; font-weight: bold'
        return ''
    
    styled_df = filter_df.style.applymap(highlight_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Decision Logic")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_pass = checks['price_range'][0]
        st.markdown(f"**Price Range Filter**")
        st.info(f"{'✅ PASS' if price_pass else '❌ FAIL'}\n\nStock price must be between ${price_min} - ${price_max} to focus on low-cap, high-volatility opportunities.")
    
    with col2:
        volume_pass = checks['volume'][0]
        st.markdown(f"**Volume Spike Filter**")
        st.info(f"{'✅ PASS' if volume_pass else '❌ FAIL'}\n\nRVOL >= {rvol_threshold}x confirms smart money is accumulating.")
    
    with col3:
        breakout_pass = checks['breakout'][0]
        st.markdown(f"**Breakout Filter**")
        st.info(f"{'✅ PASS' if breakout_pass else '❌ FAIL'}\n\nPrice must be above EMA-9 to confirm bullish momentum, not just selling.")

with tab3:
    st.markdown("### 📋 Full Data Table")
    display_df = df.copy()
    display_df['EMA-9'] = metrics['ema_9_series']
    display_df = display_df.round(4)
    st.dataframe(display_df, use_container_width=True, height=400)

st.markdown("---")

# Scenario explanation
st.markdown("### 📖 Scenario Description")

scenario_descriptions = {
    "normal": "**Normal Consolidation** - Stock is stable with regular volume. No momentum signal detected.",
    "bullish": "**Volume Spike + Uptrend** - TEXTBOOK momentum setup! High volume confirms buying, price above EMA-9 confirms uptrend. ✅ READY TO TRADE",
    "bearish": "**False Signal** - High volume but price moving DOWN. This is distribution, not accumulation. Correctly filtered out.",
    "extreme": "**Penny Stock Explosion** - Ultra-low price with extreme volume spike and strong uptrend. High risk/reward opportunity.",
    "out_of_range": "**Outside Parameters** - Great momentum signals but price is above $15. Doesn't fit our screener's target profile."
}

st.info(scenario_descriptions.get(scenario_type, "Simulation scenario"))

st.markdown("---")
st.markdown("""
### 💡 How to Use This Screener

1. **Real Trading**: Connect to live market data via yfinance or your broker's API
2. **Backtesting**: Run historical scans to test different market conditions
3. **Alerts**: Set up notifications when stocks match ALL criteria
4. **Position Sizing**: Use RVOL and market cap to size your positions
5. **Risk Management**: Always use stop-losses below the EMA-9

### ⚠️ Remember
- This screener identifies *potential* opportunities, not guaranteed trades
- Volume spikes can indicate profit-taking (selling) or accumulation (buying)
- Combine with other indicators (support/resistance, RSI, MACD) for better accuracy
- Always trade with proper risk management and position sizing
""")
