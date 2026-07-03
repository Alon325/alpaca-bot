import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def screen_momentum_stocks(ticker_list, market_cap_max=500_000_000, price_min=1.0, 
                           price_max=15.0, rvol_threshold=3.0, period_days=30, ema_span=9):
    """
    Screen a list of stocks for extreme momentum trading opportunities.
    
    This screener identifies low-cap, high-volume breakout candidates by filtering for:
    1. Low market cap (default: < $500M)
    2. Low share price (default: $1-$15)
    3. High relative volume spike (default: 3x average)
    4. Price above 9-period EMA (breakout confirmation)
    
    Args:
        ticker_list (list): List of ticker symbols to scan
        market_cap_max (int): Maximum market cap filter in dollars (default: 500M)
        price_min (float): Minimum price filter (default: 1.0)
        price_max (float): Maximum price filter (default: 15.0)
        rvol_threshold (float): Relative volume multiplier (default: 3.0x)
        period_days (int): Historical period in days for calculations (default: 30)
        ema_span (int): EMA span for trend confirmation (default: 9)
    
    Returns:
        pd.DataFrame: DataFrame containing qualifying stocks with metrics
    """
    selected_stocks = []
    total_processed = 0
    total_passed = 0
    
    logger.info(f"Starting momentum screen on {len(ticker_list)} tickers")
    logger.info(f"Filters: Market Cap < ${market_cap_max/1e6:.0f}M, Price ${price_min}-${price_max}, RVOL >= {rvol_threshold}x")
    
    for ticker_symbol in ticker_list:
        total_processed += 1
        try:
            # Fetch ticker data
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            # === Filter 1: Market Cap ===
            market_cap = info.get('marketCap', 0)
            if market_cap == 0:
                logger.debug(f"{ticker_symbol}: Skipped - No market cap data")
                continue
            
            if market_cap > market_cap_max:
                logger.debug(f"{ticker_symbol}: Skipped - Market cap ${market_cap/1e6:.1f}M exceeds limit")
                continue
            
            shares_float = info.get('floatShares', 0)
            
            # === Fetch Historical Data ===
            hist = ticker.history(period=f"{period_days}d")
            if len(hist) < 20:
                logger.debug(f"{ticker_symbol}: Skipped - Insufficient historical data ({len(hist)} days)")
                continue
            
            # === Calculate Volume Metrics ===
            avg_volume = hist['Volume'].iloc[:-1].mean()
            current_volume = hist['Volume'].iloc[-1]
            
            if avg_volume == 0:
                logger.debug(f"{ticker_symbol}: Skipped - No volume data")
                continue
            
            rvol = current_volume / avg_volume
            
            # === Get Current Price ===
            current_price = hist['Close'].iloc[-1]
            
            # === Filter 2: Price Range ===
            if not (price_min <= current_price <= price_max):
                logger.debug(f"{ticker_symbol}: Skipped - Price ${current_price:.2f} outside range ${price_min}-${price_max}")
                continue
            
            # === Calculate EMA ===
            hist['EMA'] = hist['Close'].ewm(span=ema_span, adjust=False).mean()
            current_ema = hist['EMA'].iloc[-1]
            
            # === Filter 3: Relative Volume ===
            if rvol < rvol_threshold:
                logger.debug(f"{ticker_symbol}: Skipped - RVOL {rvol:.2f}x below threshold {rvol_threshold}x")
                continue
            
            # === Filter 4: Price Above EMA (Breakout Confirmation) ===
            if current_price <= current_ema:
                logger.debug(f"{ticker_symbol}: Skipped - Price ${current_price:.2f} not above EMA ${current_ema:.2f}")
                continue
            
            # === PASSED ALL FILTERS ===
            total_passed += 1
            
            # Calculate additional metrics for trading decision
            price_change_pct = ((current_price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6] * 100) if len(hist) > 5 else 0
            volume_change_pct = ((current_volume - avg_volume) / avg_volume * 100)
            
            selected_stocks.append({
                'Ticker': ticker_symbol,
                'Price': round(current_price, 4),
                'EMA_9': round(current_ema, 4),
                'Breakout_Dist_%': round(((current_price - current_ema) / current_ema * 100), 2),
                'Market_Cap_$M': round(market_cap / 1_000_000, 2),
                'Float_M': round(shares_float / 1_000_000, 2) if shares_float else 'N/A',
                'Current_Volume': int(current_volume),
                'Avg_Volume': int(avg_volume),
                'RVOL': round(rvol, 2),
                'Volume_Change_%': round(volume_change_pct, 2),
                '5D_Change_%': round(price_change_pct, 2),
                'Scan_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            logger.info(f"✓ {ticker_symbol}: ${current_price:.2f} | RVOL: {rvol:.2f}x | Breakout: {((current_price - current_ema) / current_ema * 100):.1f}%")
            
        except Exception as e:
            logger.warning(f"{ticker_symbol}: Error - {str(e)}")
            continue
    
    # Create DataFrame
    results_df = pd.DataFrame(selected_stocks)
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SCAN COMPLETE: {total_passed}/{total_processed} stocks qualified ({(total_passed/total_processed*100):.1f}%)")
    logger.info(f"{'='*60}\n")
    
    return results_df


def rank_momentum_stocks(df, weight_rvol=0.4, weight_breakout=0.3, weight_volume=0.3):
    """
    Rank qualified momentum stocks by composite momentum score.
    
    Args:
        df (pd.DataFrame): DataFrame from screen_momentum_stocks()
        weight_rvol (float): Weight for RVOL metric (0-1)
        weight_breakout (float): Weight for breakout distance (0-1)
        weight_volume (float): Weight for volume change (0-1)
    
    Returns:
        pd.DataFrame: DataFrame sorted by momentum score (highest first)
    """
    if df.empty:
        logger.warning("No stocks to rank - DataFrame is empty")
        return df
    
    # Normalize metrics to 0-100 scale
    df['RVOL_Score'] = (df['RVOL'] - df['RVOL'].min()) / (df['RVOL'].max() - df['RVOL'].min() + 1) * 100
    df['Breakout_Score'] = (df['Breakout_Dist_%'] - df['Breakout_Dist_%'].min()) / (df['Breakout_Dist_%'].max() - df['Breakout_Dist_%'].min() + 1) * 100
    df['Volume_Score'] = (df['Volume_Change_%'] - df['Volume_Change_%'].min()) / (df['Volume_Change_%'].max() - df['Volume_Change_%'].min() + 1) * 100
    
    # Calculate composite score
    df['Momentum_Score'] = (
        df['RVOL_Score'] * weight_rvol +
        df['Breakout_Score'] * weight_breakout +
        df['Volume_Score'] * weight_volume
    )
    
    # Sort by momentum score descending
    return df.sort_values('Momentum_Score', ascending=False)


def export_results(df, output_format='csv', filename=None):
    """
    Export screening results to file.
    
    Args:
        df (pd.DataFrame): Results DataFrame
        output_format (str): 'csv' or 'excel'
        filename (str): Output filename (auto-generated if None)
    
    Returns:
        str: Path to exported file
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"momentum_scan_{timestamp}.{output_format}"
    
    if output_format == 'csv':
        df.to_csv(filename, index=False)
    elif output_format == 'excel':
        df.to_excel(filename, index=False, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported format: {output_format}")
    
    logger.info(f"Results exported to: {filename}")
    return filename


# Example usage
if __name__ == "__main__":
    # Example: Screen popular micro-cap and penny stocks
    sample_tickers = ["AMTX", "VRAX", "TCBP", "HOLO", "GNS", "MULN", "AI", "SNDL", "PROG"]
    
    # Run the screener
    results = screen_momentum_stocks(
        sample_tickers,
        market_cap_max=500_000_000,
        price_min=1.0,
        price_max=15.0,
        rvol_threshold=3.0,
        period_days=30
    )
    
    if not results.empty:
        # Rank the results
        ranked_results = rank_momentum_stocks(results)
        
        print("\n" + "="*100)
        print("EXTREME MOMENTUM STOCKS - RANKED BY COMPOSITE SCORE")
        print("="*100)
        print(ranked_results[['Ticker', 'Price', 'RVOL', 'Breakout_Dist_%', 'Volume_Change_%', 'Momentum_Score']].to_string(index=False))
        print("="*100 + "\n")
        
        # Export results
        export_results(ranked_results, output_format='csv')
    else:
        print("No momentum stocks qualified in this scan.")
