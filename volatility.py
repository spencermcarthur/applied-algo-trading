import numpy as np
import pandas as pd

from types import FunctionType


def close_close(df: pd.DataFrame, window: int, ewm: bool = False) -> pd.Series:
    '''
    Compute close-close volatility from standard candle data.

    Parameters
    ----------
    df : pd.DataFrame
        Candle data. Should have a pd.DatetimeIndex and columns open, high, low,
        close, and volume.
    window : int
        Sliding window size to use for volatility computation.
    ewm : bool, optional
        Flag to indicate whether to compute exponentially weighted volatility,
        by default False.

    Returns
    -------
    pd.Series
        Volatility (un-annualized).
    '''
    # close-close log returns
    r = df['close'].apply(np.log).diff()

    if ewm:
        vol = r.ewm(span=window, min_periods=window).std()
    else:
        vol = r.rolling(window=window, min_periods=window).std()

    return vol


def rogers_satchell(df: pd.DataFrame, window: int,
                    ewm: bool = False) -> pd.Series:
    '''
    Compute Rogers-Satchell volatility from standard candle data.

    Parameters
    ----------
    df : pd.DataFrame
        Candle data. Should have a pd.DatetimeIndex and columns open, high, low,
        close, and volume.
    window : int
        Sliding window size to use for volatility computation.
    ewm : bool, optional
        Flag to indicate whether to compute exponentially weighted volatility,
        by default False.

    Returns
    -------
    pd.Series
        Volatility (un-annualized).

    References
    ----------
    .. [1] https://portfolioslab.com/tools/rogers-satchell
    '''
    a = np.log(df['high'] / df['close']) * np.log(df['high'] / df['open'])
    b = np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])
    rs = a + b
    if ewm:
        var_rs = rs.ewm(span=window, min_periods=window).mean()
    else:
        var_rs = rs.rolling(window=window, min_periods=window).mean()

    vol = np.sqrt(var_rs)

    return vol


def yang_zhang(df: pd.DataFrame, window: int, ewm: bool = False) -> pd.Series:
    '''
    Compute Yang-Zhang volatility from standard candle data.

    Parameters
    ----------
    df : pd.DataFrame
        Candle data. Should have a pd.DatetimeIndex and columns open, high, low,
        close, and volume.
    window : int
        Sliding window size to use for volatility computation.
    ewm : bool, optional
        Flag to indicate whether to compute exponentially weighted volatility,
        by default False.

    Returns
    -------
    pd.Series
        Volatility (un-annualized).

    References
    ----------
    .. [1] https://portfolioslab.com/tools/yang-zhang
    '''
    o = np.log(df['open'] / df['close'].shift(1))  # adjusted open
    c = np.log(df['close'] / df['open'])  # adjusted close

    var_rs = rogers_satchell(df, window, ewm)**2  # Rogers-Satchell variance

    if ewm:
        var_o = o.ewm(span=window, min_periods=window).var()
        var_c = c.ewm(span=window, min_periods=window).var()
    else:
        var_o = o.rolling(window=window, min_periods=window).var()
        var_c = c.rolling(window=window, min_periods=window).var()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    vol = np.sqrt(var_o + k * var_c + (1 - k) * var_rs)

    return vol


def compute_vol_cone(df: pd.DataFrame, func: FunctionType, periods: list,
                     quantiles: list, ann_factors: list = None) -> pd.DataFrame:
    '''
    Computes the volatility cone.

    Parameters
    ----------
    df : pd.DataFrame
        Candle data for an asset.
    func : FunctionType
        Takes the candle and a period as input and returns a pandas Series
        object with the computed volatility.
    periods : list
        Periods over which to compute the volatility. Common examples are 30-,
        60-, and 120-day volatilities.
    quantiles : list
        Desired quantiles of the historical volatility.
    ann_factors : list, optional
        Annualization factors for each period length. Default none, returns
        un-annualized volatility.

    Returns
    -------
    pd.DataFrame
        Volatility cone. Index is periods and columns are quantiles.
    '''
    # sort input lists
    periods = sorted(periods)
    quantiles = sorted(quantiles)

    if ann_factors is not None:
        annualize = True
    else:
        annualize = False

    # pd.DataFrame for storing volatility cone
    cone = pd.DataFrame(index=quantiles, columns=periods)

    # loop through volatility periods
    for i in range(len(periods)):
        p = periods[i]
        if annualize:
            af = ann_factors[i]
        else:
            af = 1

        # compute volatility
        vol = func(df, p)

        # compute adjustment factor
        h = p  # subperiod length
        T = len(vol.dropna())  # number of observations
        n = T - h + 1  # number of distinct subseries of length h
        m = 1 / (1 - h / n + (h**2 - 1) / (3*n**2))  # adjustment factor

        # add volatility quantiles to cone
        vol *= np.sqrt(m)   # adjust
        vol *= af  # annualize
        cone[p] = vol.quantile(quantiles)  # add to cone

    # transpose cone so that index is periods and columns are quantiles
    cone = cone.transpose(copy=True)

    return cone
