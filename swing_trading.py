import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

def analyze_stock(ticker, period='6mo'):
    """
    Complete swing trading analysis with all technical indicators.
    Input: ticker symbol (e.g., 'ALB')
    Output: Candlestick chart with all indicators overlaid
    """
    # Download data
    df = yf.download(ticker, period=period, progress=False)
    df = df.dropna()
    
    # Moving Averages
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    
    # Stochastic Oscillator (14,3,3)
    low_min = df['Low'].rolling(14).min()
    high_max = df['High'].rolling(14).max()
    df['%K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['%D'] = df['%K'].rolling(3).mean()
    
    # Fibonacci Retracement (recent swing high/low)
    high_indices = argrelextrema(df['High'].values, np.greater, order=5)[0]
    low_indices = argrelextrema(df['Low'].values, np.less, order=5)[0]
    
    if len(high_indices) > 0 and len(low_indices) > 0:
        high_idx = high_indices[-1]
        low_idx = low_indices[-1]
        swing_high = df['High'].iloc[high_idx]
        swing_low = df['Low'].iloc[low_idx]
    else:
        swing_high = df['High'].max()
        swing_low = df['Low'].min()
        high_idx = df['High'].idxmax()
        low_idx = df['Low'].idxmin()
    
    fib_range = swing_high - swing_low
    
    fib_levels = {
        '0%': swing_high,
        '23.6%': swing_high - 0.236 * fib_range,
        '38.2%': swing_high - 0.382 * fib_range,
        '50%': swing_high - 0.5 * fib_range,
        '61.8%': swing_high - 0.618 * fib_range,
        '100%': swing_low
    }
    
    # Fibonacci Projection (Extension beyond swing high)
    try:
        if high_idx > low_idx:
            retrace_low = df['Low'].iloc[low_idx:high_idx+1].min()
        else:
            retrace_low = swing_low
    except:
        retrace_low = swing_low
    
    proj_range = swing_high - retrace_low
    fib_proj = {
        '127.2%': swing_high + 0.272 * proj_range,
        '161.8%': swing_high + 0.618 * proj_range,
        '261.8%': swing_high + 1.618 * proj_range
    }
    
    # Support/Resistance (Pivot points from local extrema) - **COMPLETE FIX**
    support_idx = argrelextrema(df['Low'].values, np.less, order=10)[0]
    resistance_idx = argrelextrema(df['High'].values, np.greater, order=10)[0]
    
    support_levels = df['Low'].iloc[support_idx]
    resistance_levels = df['High'].iloc[resistance_idx]
    
    # Convert to numpy arrays to avoid any pandas issues
    top_supports = support_levels.sort_values(ascending=True).head(3).values if len(support_levels) > 0 else np.array([])
    top_resist = resistance_levels.sort_values(ascending=False).head(3).values if len(resistance_levels) > 0 else np.array([])
    
    # Trendline (linear regression on recent lows)
    recent_lows = df['Low'].tail(100)
    recent_lows_idx = argrelextrema(recent_lows.values, np.less, order=3)[0]
    trend_slope = np.nan
    
    if len(recent_lows_idx) >= 2:
        x_lows = recent_lows_idx
        y_lows = recent_lows.iloc[recent_lows_idx]
        coeffs = np.polyfit(x_lows, y_lows, 1)
        trend_slope = coeffs[0]
        trend_intercept = coeffs[1]
        df['Trendline'] = trend_slope * np.arange(len(df)) + trend_intercept
    else:
        df['Trendline'] = np.nan
    
    # Create addplots
    apds = []
    
    # Add MAs
    apds.append(mpf.make_addplot(df['MA50'], color='orange', width=1, label='MA50'))
    if len(df['MA200'].dropna()) > 0:
        apds.append(mpf.make_addplot(df['MA200'], color='red', width=1.5, label='MA200'))
    
    # Add Fibonacci levels as hlines (main panel)
    for level_name, price in fib_levels.items():
        apds.append(mpf.make_addplot([price]*len(df), color='purple', width=1, 
                                   linestyle='--', alpha=0.7, label=f'Fib {level_name}'))
    
    # Add Support/Resistance (top 3 levels) - **NOW SAFE**
    for s in top_supports:
        apds.append(mpf.make_addplot([s]*len(df), color='green', width=2, alpha=0.8, 
                                   linestyle='-', label='Support'))
    
    for r in top_resist:
        apds.append(mpf.make_addplot([r]*len(df), color='red', width=2, alpha=0.8, 
                                   linestyle='-', label='Resistance'))
    
    # Add Stochastic subplot (separate panel)
    stoch_df = pd.DataFrame({
        '%K': df['%K'].fillna(50),
        '%D': df['%D'].fillna(50)
    }, index=df.index)
    apds.append(mpf.make_addplot(stoch_df['%K'], panel=1, color='blue', width=1))
    apds.append(mpf.make_addplot(stoch_df['%D'], panel=1, color='orange', width=1))
    
    # Add Trendline
    if not df['Trendline'].isna().all():
        apds.append(mpf.make_addplot(df['Trendline'], color='cyan', width=2, alpha=0.8, label='Trendline'))
    
    # Plot
    mpf.plot(df, type='candle', style='yahoo', volume=True,
             addplot=apds, 
             title=f'{ticker} Swing Trading Analysis\n'
                   f'Swing High: ${swing_high:.2f} | Swing Low: ${swing_low:.2f}',
             figsize=(16,12), panel_ratios=(3,1))
    
    # Print key levels
    print(f"\n{ticker} Key Levels:")
    print("Fibonacci Retracement:", {k: f"${v:.2f}" for k,v in fib_levels.items()})
    print("Fibonacci Projection:", {k: f"${v:.2f}" for k,v in fib_proj.items()})
    print("Top Supports:", [f"${s:.2f}" for s in top_supports])
    print("Top Resistance:", [f"${r:.2f}" for r in top_resist])
    print("Trendline Slope:", f"{trend_slope:.4f}" if not np.isnan(trend_slope) else "N/A")

# Usage
if __name__ == "__main__":
    ticker = input("Enter ticker symbol (e.g., AAPL): ").upper()
    analyze_stock(ticker)
