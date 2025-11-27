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
    high_idx = argrelextrema(df['High'].values, np.greater, order=5)[0]
    low_idx = argrelextrema(df['Low'].values, np.less, order=5)[0]
    
    if len(high_idx) > 0 and len(low_idx) > 0:
        high_idx = high_idx[-1]
        low_idx = low_idx[-1]
        swing_high = df['High'].iloc[high_idx]
        swing_low = df['Low'].iloc[low_idx]
    else:
        swing_high = df['High'].max()
        swing_low = df['Low'].min()
    
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
    retrace_low = df['Low'].iloc[low_idx:high_idx+1].min() if len(high_idx) > 0 and len(low_idx) > 0 else swing_low
    proj_range = swing_high - retrace_low
    fib_proj = {
        '127.2%': swing_high + 0.272 * proj_range,
        '161.8%': swing_high + 0.618 * proj_range,
        '261.8%': swing_high + 1.618 * proj_range
    }
    
    # Support/Resistance (Pivot points from local extrema) - FIXED
    support_idx = argrelextrema(df['Low'].values, np.less, order=10)[0]
    resistance_idx = argrelextrema(df['High'].values, np.greater, order=10)[0]
    
    support_levels = df['Low'].iloc[support_idx] if len(support_idx) > 0 else pd.Series([])
    resistance_levels = df['High'].iloc[resistance_idx] if len(resistance_idx) > 0 else pd.Series([])
    
    # Trendline (linear regression on recent lows)
    recent_lows = df['Low'].tail(100)
    recent_lows_idx = argrelextrema(recent_lows.values, np.less, order=3)[0]
    trend_slope = np.nan
    
    if len(recent_lows_idx) >= 2:
        x_lows = recent_lows_idx
        y_lows = recent_lows.iloc[recent_lows_idx]
        trend_slope = np.polyfit(x_lows, y_lows, 1)[0]
        trend_intercept = np.polyfit(x_lows, y_lows, 1)[1]
        df['Trendline'] = trend_slope * np.arange(len(df)) + trend_intercept
    else:
        df['Trendline'] = np.nan
    
    # Create addplots
    apds = []
    
    # Add MAs
    apds.append(mpf.make_addplot(df['MA50'], color='orange', width=1, label='MA50'))
    if len(df['MA200'].dropna()) > 0:
        apds.append(mpf.make_addplot(df['MA200'], color='red', width=1.5, label='MA200'))
    
    # Add Fibonacci levels as hlines (panel 1)
    fib_panel = []
    for level_name, price in fib_levels.items():
        fib_panel.append(mpf.make_addplot([price]*len(df), color='purple', width=1, 
                                        linestyle='--', panel=1, label=f'Fib {level_name}'))
    apds.extend(fib_panel)
    
    # Add Support/Resistance (top 3 levels) - FIXED
    top_supports = support_levels.nlargest(3) if len(support_levels) >= 1 else pd.Series([])
    top_resist = resistance_levels.nsmallest(3) if len(resistance_levels) >= 1 else pd.Series([])
    
    for s in top_supports:
        apds.append(mpf.make_addplot([s]*len(df), color='green', width=1.5, alpha=0.7, 
                                   panel=1, label='Support'))
    for r in top_resist:
        apds.append(mpf.make_addplot([r]*len(df), color='red', width=1.5, alpha=0.7, 
                                   panel=1, label='Resistance'))
    
    # Add Stochastic subplot (panel 2)
    stoch_df = pd.DataFrame({
        '%K': df['%K'],
        '%D': df['%D']
    }, index=df.index).fillna(50)
    stoch_panel = mpf.make_addplot(stoch_df, panel=2, color=['blue', 'orange'], 
                                 ylabel='Stochastic', secondary_y=False)
    apds.append(stoch_panel)
    
    # Plot
    fig, axes = mpf.plot(df, type='candle', style='yahoo', volume=True,
                        addplot=apds, panel_ratios=(4,2,1),
                        title=f'{ticker} Swing Trading Analysis\n'
                              f'Fib High: ${swing_high:.2f} | Low: ${swing_low:.2f}',
                        figsize=(16,12), returnfig=True)
    
    plt.tight_layout()
    plt.show()
    
    # Print key levels
    print(f"\n{ticker} Key Levels:")
    print("Fibonacci Retracement:", {k: f"${v:.2f}" for k,v in fib_levels.items()})
    print("Fibonacci Projection:", {k: f"${v:.2f}" for k,v in fib_proj.items()})
    print("Top Supports:", [f"${s:.2f}" for s in top_supports] if len(top_supports)>0 else "N/A")
    print("Top Resistance:", [f"${r:.2f}" for r in top_resist] if len(top_resist)>0 else "N/A")
    print("Trendline Slope:", f"{trend_slope:.4f}" if not np.isnan(trend_slope) else "N/A")

# Usage
if __name__ == "__main__":
    ticker = input("Enter ticker symbol (e.g., AAPL): ").upper()
    analyze_stock(ticker)

