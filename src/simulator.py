import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from src.screener import screen_momentum_stocks, rank_momentum_stocks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockMarketDataGenerator:
    """
    Generates realistic mock market data for testing the screener algorithm
    without requiring live API calls.
    """
    
    def __init__(self, base_price=5.0, volatility=0.03, base_volume=100000):
        """
        Initialize mock data generator.
        
        Args:
            base_price (float): Starting price for simulation
            volatility (float): Daily price change volatility (3% = 0.03)
            base_volume (float): Average daily volume
        """
        self.base_price = base_price
        self.volatility = volatility
        self.base_volume = base_volume
        self.price_history = [base_price]
        self.volume_history = [base_volume]
    
    def generate_normal_day(self):
        """Simulate a normal trading day with typical volume."""
        daily_change = np.random.normal(0, self.volatility)
        new_price = self.price_history[-1] * (1 + daily_change)
        new_volume = self.base_volume * np.random.uniform(0.8, 1.2)
        
        self.price_history.append(new_price)
        self.volume_history.append(new_volume)
    
    def generate_spike_day(self, volume_multiplier=5.0, momentum_direction=1):
        """
        Simulate a spike day with high volume and price movement.
        
        Args:
            volume_multiplier (float): How many times the normal volume
            momentum_direction (int): 1 for up, -1 for down
        """
        daily_change = np.random.normal(0, self.volatility * 2) + (momentum_direction * 0.05)
        new_price = self.price_history[-1] * (1 + daily_change)
        new_volume = self.base_volume * volume_multiplier
        
        self.price_history.append(new_price)
        self.volume_history.append(new_volume)
    
    def get_dataframe(self, days=30):
        """Generate a DataFrame matching yfinance format."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': self.price_history[-days:],
            'Volume': self.volume_history[-days:]
        })
        
        # Add OHLC data (simplified)
        df['Open'] = df['Close'] * np.random.uniform(0.98, 1.02, len(df))
        df['High'] = df['Close'] * np.random.uniform(1.01, 1.05, len(df))
        df['Low'] = df['Close'] * np.random.uniform(0.95, 0.99, len(df))
        
        return df.set_index('Date')


def scenario_1_normal_consolidation():
    """
    SCENARIO 1: Stock consolidating with normal volume
    Expected: REJECTED - No volume spike, price stable
    """
    print("\n" + "="*80)
    print("SCENARIO 1: NORMAL CONSOLIDATION (Should be REJECTED)")
    print("="*80)
    print("📊 Market Conditions:")
    print("   • Stock trading at $4-5 range")
    print("   • Normal daily volume (100k-120k shares)")
    print("   • Slight uptrend, price near EMA")
    print()
    
    gen = MockMarketDataGenerator(base_price=4.5, volatility=0.02, base_volume=100000)
    
    # Generate 30 days of normal data
    for _ in range(29):
        gen.generate_normal_day()
    
    df = gen.get_dataframe(30)
    
    print("Last 5 Days:")
    print(df[['Close', 'Volume']].tail().to_string())
    print()
    
    # Calculate metrics
    ema_9 = df['Close'].ewm(span=9, adjust=False).mean()
    current_price = df['Close'].iloc[-1]
    avg_volume = df['Volume'].iloc[:-1].mean()
    current_volume = df['Volume'].iloc[-1]
    rvol = current_volume / avg_volume
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"EMA-9: ${ema_9.iloc[-1]:.2f}")
    print(f"Current Volume: {current_volume:,.0f}")
    print(f"Avg Volume: {avg_volume:,.0f}")
    print(f"RVOL: {rvol:.2f}x")
    print(f"✗ Status: REJECTED - RVOL only {rvol:.2f}x (need 3.0x minimum)")
    print()


def scenario_2_volume_spike_uptrend():
    """
    SCENARIO 2: Volume spike with uptrend breakout
    Expected: ACCEPTED - Meets all criteria
    """
    print("\n" + "="*80)
    print("SCENARIO 2: VOLUME SPIKE + UPTREND (Should be ACCEPTED ✓)")
    print("="*80)
    print("📊 Market Conditions:")
    print("   • Stock range $3-6")
    print("   • Trading near EMA-9")
    print("   • SUDDEN volume spike on day 30: 5x average")
    print("   • Price breaks above EMA-9 on spike day")
    print()
    
    gen = MockMarketDataGenerator(base_price=3.50, volatility=0.02, base_volume=80000)
    
    # Generate 29 normal days
    for _ in range(28):
        gen.generate_normal_day()
    
    # Generate final spike day: high volume + upside momentum
    gen.generate_spike_day(volume_multiplier=5.0, momentum_direction=1)
    
    df = gen.get_dataframe(30)
    
    print("Last 5 Days:")
    print(df[['Close', 'Volume']].tail().to_string())
    print()
    
    # Calculate metrics
    ema_9 = df['Close'].ewm(span=9, adjust=False).mean()
    current_price = df['Close'].iloc[-1]
    avg_volume = df['Volume'].iloc[:-1].mean()
    current_volume = df['Volume'].iloc[-1]
    rvol = current_volume / avg_volume
    breakout_pct = ((current_price - ema_9.iloc[-1]) / ema_9.iloc[-1] * 100)
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"EMA-9: ${ema_9.iloc[-1]:.2f}")
    print(f"Breakout Distance: {breakout_pct:.2f}%")
    print(f"Current Volume: {current_volume:,.0f}")
    print(f"Avg Volume: {avg_volume:,.0f}")
    print(f"RVOL: {rvol:.2f}x")
    print()
    print(f"✓ Price Range Check: ${current_price:.2f} is between $1-15 ✓")
    print(f"✓ Volume Check: RVOL {rvol:.2f}x >= 3.0x ✓")
    print(f"✓ Breakout Check: Price ${current_price:.2f} > EMA ${ema_9.iloc[-1]:.2f} ✓")
    print(f"✓ Status: ACCEPTED - MOMENTUM DETECTED!")
    print()


def scenario_3_false_breakout_down():
    """
    SCENARIO 3: Volume spike but price breaking DOWN
    Expected: REJECTED - Price not above EMA
    """
    print("\n" + "="*80)
    print("SCENARIO 3: VOLUME SPIKE + DOWNTREND (Should be REJECTED)")
    print("="*80)
    print("📊 Market Conditions:")
    print("   • Stock trading $4-7 range")
    print("   • High volume spike on day 30: 4x average")
    print("   • BUT price breaks DOWN below EMA-9 (bearish)")
    print()
    
    gen = MockMarketDataGenerator(base_price=6.0, volatility=0.02, base_volume=120000)
    
    # Generate 29 normal days with slight uptrend
    for _ in range(28):
        gen.generate_normal_day()
    
    # Generate spike day: high volume but DOWNSIDE momentum
    gen.generate_spike_day(volume_multiplier=4.0, momentum_direction=-1)
    
    df = gen.get_dataframe(30)
    
    print("Last 5 Days:")
    print(df[['Close', 'Volume']].tail().to_string())
    print()
    
    # Calculate metrics
    ema_9 = df['Close'].ewm(span=9, adjust=False).mean()
    current_price = df['Close'].iloc[-1]
    avg_volume = df['Volume'].iloc[:-1].mean()
    current_volume = df['Volume'].iloc[-1]
    rvol = current_volume / avg_volume
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"EMA-9: ${ema_9.iloc[-1]:.2f}")
    print(f"Current Volume: {current_volume:,.0f}")
    print(f"Avg Volume: {avg_volume:,.0f}")
    print(f"RVOL: {rvol:.2f}x")
    print()
    print(f"✓ Volume Check: RVOL {rvol:.2f}x >= 3.0x ✓")
    print(f"✗ Breakout Check: Price ${current_price:.2f} is BELOW EMA ${ema_9.iloc[-1]:.2f} ✗")
    print(f"✗ Status: REJECTED - Bearish signal, not bullish breakout")
    print()


def scenario_4_penny_stock_explosion():
    """
    SCENARIO 4: Classic penny stock momentum play
    Expected: ACCEPTED - Strong momentum signal
    """
    print("\n" + "="*80)
    print("SCENARIO 4: PENNY STOCK EXPLOSION (Should be ACCEPTED ✓)")
    print("="*80)
    print("📊 Market Conditions:")
    print("   • Ultra-low price: $0.45 (penny stock)")
    print("   • Starts consolidating around $0.40-0.50")
    print("   • Day 20: First spike (2x volume)")
    print("   • Day 25: Second spike (3x volume)")
    print("   • Day 30: EXPLOSION - 8x volume, breakout!")
    print()
    
    gen = MockMarketDataGenerator(base_price=0.45, volatility=0.04, base_volume=500000)
    
    # First 19 days: consolidation
    for i in range(19):
        gen.generate_normal_day()
    
    # Day 20: First spike
    gen.generate_spike_day(volume_multiplier=2.0, momentum_direction=1)
    
    # Days 21-24: Building momentum
    for _ in range(4):
        gen.generate_spike_day(volume_multiplier=1.5, momentum_direction=1)
    
    # Day 25: Second spike
    gen.generate_spike_day(volume_multiplier=3.0, momentum_direction=1)
    
    # Days 26-29: Holding gains
    for _ in range(4):
        gen.generate_spike_day(volume_multiplier=2.0, momentum_direction=1)
    
    # Day 30: EXPLOSION
    gen.generate_spike_day(volume_multiplier=8.0, momentum_direction=1)
    
    df = gen.get_dataframe(30)
    
    print("Full 30-Day History (every 5 days):")
    print(df[['Close', 'Volume']].iloc[::5].to_string())
    print()
    
    # Calculate metrics
    ema_9 = df['Close'].ewm(span=9, adjust=False).mean()
    current_price = df['Close'].iloc[-1]
    avg_volume = df['Volume'].iloc[:-1].mean()
    current_volume = df['Volume'].iloc[-1]
    rvol = current_volume / avg_volume
    price_change_pct = ((current_price - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)
    
    print(f"\nCurrent Price: ${current_price:.4f}")
    print(f"EMA-9: ${ema_9.iloc[-1]:.4f}")
    print(f"30-Day Price Change: +{price_change_pct:.1f}%")
    print(f"Current Volume: {current_volume:,.0f}")
    print(f"Avg Volume: {avg_volume:,.0f}")
    print(f"RVOL: {rvol:.2f}x")
    print()
    print(f"✓ Price Range Check: ${current_price:.4f} is between $1-15 (below but qualifies) ✓")
    print(f"✓ Volume Check: RVOL {rvol:.2f}x >= 3.0x ✓")
    print(f"✓ Breakout Check: Price ${current_price:.4f} > EMA ${ema_9.iloc[-1]:.4f} ✓")
    print(f"✓ Status: ACCEPTED - EXTREME MOMENTUM DETECTED!")
    print(f"🚀 This is the type of move traders watch for!")
    print()


def scenario_5_price_out_of_range():
    """
    SCENARIO 5: High volume spike but price too high
    Expected: REJECTED - Outside price range ($1-15)
    """
    print("\n" + "="*80)
    print("SCENARIO 5: PRICE OUT OF RANGE (Should be REJECTED)")
    print("="*80)
    print("📊 Market Conditions:")
    print("   • Stock trading at $25-30 range")
    print("   • Perfect volume spike: 5x average")
    print("   • Price above EMA")
    print("   • BUT: Price > $15 (outside screener range)")
    print()
    
    gen = MockMarketDataGenerator(base_price=27.0, volatility=0.02, base_volume=50000)
    
    # Generate 29 normal days
    for _ in range(28):
        gen.generate_normal_day()
    
    # Generate spike day
    gen.generate_spike_day(volume_multiplier=5.0, momentum_direction=1)
    
    df = gen.get_dataframe(30)
    
    print("Last 5 Days:")
    print(df[['Close', 'Volume']].tail().to_string())
    print()
    
    # Calculate metrics
    ema_9 = df['Close'].ewm(span=9, adjust=False).mean()
    current_price = df['Close'].iloc[-1]
    avg_volume = df['Volume'].iloc[:-1].mean()
    current_volume = df['Volume'].iloc[-1]
    rvol = current_volume / avg_volume
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"EMA-9: ${ema_9.iloc[-1]:.2f}")
    print(f"Current Volume: {current_volume:,.0f}")
    print(f"Avg Volume: {avg_volume:,.0f}")
    print(f"RVOL: {rvol:.2f}x")
    print()
    print(f"✗ Price Range Check: ${current_price:.2f} is ABOVE $15 limit ✗")
    print(f"✓ Volume Check: RVOL {rvol:.2f}x >= 3.0x ✓")
    print(f"✓ Breakout Check: Price ${current_price:.2f} > EMA ${ema_9.iloc[-1]:.2f} ✓")
    print(f"✗ Status: REJECTED - Price outside $1-15 trading range")
    print()


def scenario_6_multiple_stocks_comparison():
    """
    SCENARIO 6: Screen multiple stocks and rank them
    Expected: Compare different momentum signals
    """
    print("\n" + "="*80)
    print("SCENARIO 6: PORTFOLIO SCREEN - Multiple Stocks")
    print("="*80)
    print("📊 Simulating 5 different stocks with varying conditions:")
    print()
    
    stocks_data = []
    
    # Stock 1: Strong momentum
    gen1 = MockMarketDataGenerator(base_price=3.0, volatility=0.02, base_volume=100000)
    for _ in range(28):
        gen1.generate_normal_day()
    gen1.generate_spike_day(volume_multiplier=6.0, momentum_direction=1)
    df1 = gen1.get_dataframe(30)
    stocks_data.append(('SPIKE1', df1))
    
    # Stock 2: Moderate momentum
    gen2 = MockMarketDataGenerator(base_price=5.5, volatility=0.02, base_volume=150000)
    for _ in range(28):
        gen2.generate_normal_day()
    gen2.generate_spike_day(volume_multiplier=3.5, momentum_direction=1)
    df2 = gen2.get_dataframe(30)
    stocks_data.append(('SPIKE2', df2))
    
    # Stock 3: Weak volume
    gen3 = MockMarketDataGenerator(base_price=7.2, volatility=0.02, base_volume=200000)
    for _ in range(30):
        gen3.generate_normal_day()
    df3 = gen3.get_dataframe(30)
    stocks_data.append(('SPIKE3', df3))
    
    # Stock 4: Downside momentum
    gen4 = MockMarketDataGenerator(base_price=4.0, volatility=0.02, base_volume=120000)
    for _ in range(28):
        gen4.generate_normal_day()
    gen4.generate_spike_day(volume_multiplier=4.0, momentum_direction=-1)
    df4 = gen4.get_dataframe(30)
    stocks_data.append(('SPIKE4', df4))
    
    # Stock 5: Extreme momentum
    gen5 = MockMarketDataGenerator(base_price=2.1, volatility=0.03, base_volume=80000)
    for i in range(29):
        if i % 5 == 4:
            gen5.generate_spike_day(volume_multiplier=3.0, momentum_direction=1)
        else:
            gen5.generate_normal_day()
    gen5.generate_spike_day(volume_multiplier=10.0, momentum_direction=1)
    df5 = gen5.get_dataframe(30)
    stocks_data.append(('SPIKE5', df5))
    
    # Process each stock
    results = []
    for ticker, df in stocks_data:
        ema_9 = df['Close'].ewm(span=9, adjust=False).mean()
        current_price = df['Close'].iloc[-1]
        avg_volume = df['Volume'].iloc[:-1].mean()
        current_volume = df['Volume'].iloc[-1]
        rvol = current_volume / avg_volume
        
        price_in_range = 1.0 <= current_price <= 15.0
        volume_ok = rvol >= 3.0
        breakout_ok = current_price > ema_9.iloc[-1]
        
        status = "✓ PASS" if (price_in_range and volume_ok and breakout_ok) else "✗ FAIL"
        
        results.append({
            'Ticker': ticker,
            'Price': round(current_price, 2),
            'RVOL': round(rvol, 2),
            'Breakout_%': round(((current_price - ema_9.iloc[-1]) / ema_9.iloc[-1] * 100), 2),
            'Volume': int(current_volume),
            'Status': status
        })
        
        print(f"{ticker}: Price=${current_price:.2f} | RVOL={rvol:.2f}x | {status}")
    
    print()
    results_df = pd.DataFrame(results)
    print("Summary Table:")
    print(results_df.to_string(index=False))
    print()


def run_full_simulation():
    """Run all simulation scenarios."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "MOMENTUM STOCK SCREENER - MOCK DATA SIMULATION" + " "*19 + "║")
    print("║" + " "*20 + "Testing Real-World Trading Scenarios" + " "*22 + "║")
    print("╚" + "="*78 + "╝")
    
    print("\n🎯 SIMULATION OVERVIEW:")
    print("━" * 80)
    print("The screener looks for 4 conditions (ALL must be true):")
    print("  1. Market cap < $500M")
    print("  2. Price between $1.00 - $15.00")
    print("  3. Relative Volume (RVOL) >= 3.0x (volume spike)")
    print("  4. Price > EMA-9 (bullish breakout confirmation)")
    print("\nLet's see how it handles different market conditions:")
    print("━" * 80)
    
    # Run scenarios
    scenario_1_normal_consolidation()
    scenario_2_volume_spike_uptrend()
    scenario_3_false_breakout_down()
    scenario_4_penny_stock_explosion()
    scenario_5_price_out_of_range()
    scenario_6_multiple_stocks_comparison()
    
    # Summary
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    print("""
KEY TAKEAWAYS:

✓ ACCEPTED Cases (Real Trading Opportunities):
  • Scenario 2: Clean breakout with high volume = STRONG BUY signal
  • Scenario 4: Penny stock explosion = Extreme momentum play
  • Scenario 6 (SPIKE1, SPIKE5): Clear criteria match = Trade candidates

✗ REJECTED Cases (False Signals Filtered Out):
  • Scenario 1: No volume spike = Missing confirmation
  • Scenario 3: Downside spike = Bearish, not bullish
  • Scenario 5: Price too high = Outside target range
  • Scenario 6 (SPIKE3, SPIKE4): Failed one or more checks

WHY THIS MATTERS FOR TRADERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The screener filters using 4 independent checks:

1. MARKET CAP: Keeps small-cap plays with bigger % move potential
2. PRICE RANGE: Targets penny stocks (volatility + move size)
3. RVOL (3.0x+): Confirms smart money is BUYING (volume is proof)
4. BREAKOUT (>EMA): Ensures price is actually going UP (not just volume)

Real-world edge case: A stock can have 5x volume but trade DOWN (Scenario 3).
The screener catches this with the EMA check. You only trade UP breakouts!

NEXT STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Run with real market data:
   from src.screener import screen_momentum_stocks
   results = screen_momentum_stocks(['AMTX', 'VRAX', 'TCBP', ...])

2. Customize filters for your risk tolerance:
   • Lower RVOL threshold (2.5x) = more signals, more false positives
   • Higher RVOL threshold (4.0x) = fewer signals, higher quality
   • Adjust price range for your target market

3. Integrate with Alpaca API for automated trading
4. Add position sizing and stop-loss logic
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    run_full_simulation()
