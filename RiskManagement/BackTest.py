import numpy as np
import pandas as pd

def backtest_strategy(price_series, strategy_fn):
    """
    price_series: pd.Series of prices (e.g., AAPL)
    strategy_fn: function that takes price_series and returns signal (1=long, -1=short, 0=flat)
    """
    signal = strategy_fn(price_series)
    returns = price_series.pct_change().fillna(0)
    strategy_returns = signal.shift(1) * returns  # shift to avoid lookahead bias
    equity_curve = (1 + strategy_returns).cumprod()

    stats = {
        "Total Return": equity_curve.iloc[-1] - 1,
        "Annualized Return": (equity_curve.iloc[-1])**(252 / len(equity_curve)) - 1,
        "Annualized Volatility": strategy_returns.std() * np.sqrt(252),
        "Sharpe Ratio": (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252),
        "Max Drawdown": max_drawdown(equity_curve),
        "Calmar Ratio": calmar_ratio(equity_curve),
        "Turnover": turnover(signal),
        "Win Rate": (strategy_returns > 0).mean()
    }

    return stats, equity_curve, strategy_returns

def max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    return drawdown.min()

def calmar_ratio(equity_curve):
    ann_return = (equity_curve.iloc[-1])**(252 / len(equity_curve)) - 1
    mdd = abs(max_drawdown(equity_curve))
    return ann_return / mdd if mdd != 0 else np.nan

def turnover(signal):
    return signal.diff().abs().sum() / len(signal)

# === Example Strategy ===
def moving_average_crossover(prices, short_window=10, long_window=50):
    short_ma = prices.rolling(short_window).mean()
    long_ma = prices.rolling(long_window).mean()
    signal = np.where(short_ma > long_ma, 1, -1)
    return pd.Series(signal, index=prices.index)

# === Example Usage ===
if __name__ == "__main__":
    import yfinance as yf
    data = yf.download("AAPL", start="2020-01-01", end="2024-12-31")["Adj Close"]
    stats, curve, rets = backtest_strategy(data, moving_average_crossover)
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")

    print('debug')
