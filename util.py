import pandas as pd


def resample_bars(df: pd.DataFrame, rule: str, **kwargs) -> pd.DataFrame:
    '''
    Resample candle data to a different frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Candle data.
    rule : str
        Rule for resampling data. See pd.DataFrame.resample.

    Returns
    -------
    pd.DataFrame
        Resampled candle data.
    '''
    df_new = pd.DataFrame()

    df_new['open'] = df['open'].resample(rule, **kwargs).first()
    df_new['high'] = df['high'].resample(rule, **kwargs).max()
    df_new['low'] = df['low'].resample(rule, **kwargs).min()
    df_new['close'] = df['close'].resample(rule, **kwargs).last()
    df_new['volume'] = df['volume'].resample(rule, **kwargs).sum()

    return df_new
