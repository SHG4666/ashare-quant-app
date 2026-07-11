import pandas as pd


def add_moving_averages(df: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.DataFrame:
    """Return a copy with short/long simple moving average columns."""
    if short_window <= 0 or long_window <= 0:
        raise ValueError("moving average windows must be positive")
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")
    if "close" not in df.columns:
        raise KeyError("input dataframe must contain a 'close' column")

    result = df.copy()
    result["ma_short"] = result["close"].rolling(short_window).mean()
    result["ma_long"] = result["close"].rolling(long_window).mean()
    return result


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Return a copy with a relative strength index (RSI) column."""
    if window <= 0:
        raise ValueError("RSI window must be positive")
    if "close" not in df.columns:
        raise KeyError("input dataframe must contain a 'close' column")

    result = df.copy()
    delta = result["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    rs = avg_gain / avg_loss
    result["rsi"] = 100 - (100 / (1 + rs))
    result.loc[(avg_loss == 0) & avg_gain.notna(), "rsi"] = 100.0
    return result


def add_macd(
    df: pd.DataFrame,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> pd.DataFrame:
    """Return a copy with MACD DIF, DEA and histogram columns."""
    if fast_window <= 0 or slow_window <= 0 or signal_window <= 0:
        raise ValueError("MACD windows must be positive")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")
    if "close" not in df.columns:
        raise KeyError("input dataframe must contain a 'close' column")

    result = df.copy()
    ema_fast = result["close"].ewm(span=fast_window, adjust=False).mean()
    ema_slow = result["close"].ewm(span=slow_window, adjust=False).mean()
    result["macd_dif"] = ema_fast - ema_slow
    result["macd_dea"] = result["macd_dif"].ewm(span=signal_window, adjust=False).mean()
    result["macd_hist"] = 2 * (result["macd_dif"] - result["macd_dea"])
    return result


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Return a copy with Bollinger Band middle/upper/lower columns."""
    if window <= 0:
        raise ValueError("Bollinger window must be positive")
    if num_std <= 0:
        raise ValueError("Bollinger standard deviation multiplier must be positive")
    if "close" not in df.columns:
        raise KeyError("input dataframe must contain a 'close' column")

    result = df.copy()
    middle = result["close"].rolling(window).mean()
    std = result["close"].rolling(window).std()
    result["bb_middle"] = middle
    result["bb_upper"] = middle + num_std * std
    result["bb_lower"] = middle - num_std * std
    return result
