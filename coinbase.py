import numpy as np
import os
import pandas as pd
import requests

from datetime import datetime, timedelta
from dateutil import tz
from time import sleep


API_URL = 'https://api.exchange.coinbase.com'


def validate_response(resp: requests.Response) -> requests.Response:
    '''
    Validate HTTP response.

    Parameters
    ----------
    resp : requests.Response
        Response object to be validated.

    Returns
    -------
    requests.Response
        Unaltered request. Meant for nesting an request call inside of the
        validate function.
    '''
    assert resp.status_code == 200, resp.raise_for_status()
    return resp


def _get_candles(product_id: str,
                 start: datetime = None,
                 end: datetime = None,
                 granularity: int = None) -> pd.DataFrame:
    # get data
    params = {
        'granularity': granularity,
        'start': start.isoformat(),
        'end': end.isoformat()
    }
    headers = {'Connection': 'close'}

    # handle retry on error
    num_retries = 0
    while True:
        try:
            resp = requests.get(API_URL + f'/products/{product_id}/candles',
                                params=params, headers=headers)
            resp = validate_response(resp)
            break
        except:
            num_retries += 1
            if num_retries > 5:
                raise RuntimeError('failed 5 retries, aborting')
            sleep(3)

    # format response as DataFrame
    df = pd.DataFrame(resp.json())
    if df.empty:
        return df
    df.columns = ['time', 'low', 'high', 'open', 'close', 'volume']
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    return df.sort_index()


def get_candles(product_id: str,
                start: datetime = None,
                end: datetime = None,
                granularity: int = None) -> pd.DataFrame:
    '''
    Get price candles from Coinbase.

    Parameters
    ----------
    product_id : str
        Name of the currency pair. E.g. BTC-USD, ETH-BTC, etc.
    start : datetime, optional
        First candle in desired date range, by default now - 299 * granularity.
    end : datetime, optional
        Last candle in desired date range, by default now.
    granularity : int, optional
        Size of each candle in seconds, by default 60.

    Returns
    -------
    pd.DataFrame
        Candles in desired range. Columns are open, high, low, close, and
        volume. Index is a pd.DatetimeIndex converted to UTC time.
    '''
    # validate granularity
    granularity_choices = [60, 300, 900, 3600, 21600, 86400]
    if granularity is None:
        # default to 1 minute
        granularity = granularity_choices[0]
    elif not isinstance(granularity, int):
        raise ValueError('granularity must be int')
    elif granularity not in granularity_choices:
        raise ValueError(
            f'granularity must be one of the following: '
            f'{", ".join(map(lambda x: str(x), granularity_choices))}'
        )

    # validate datetimes
    if start is None or end is None:
        # set default values based on current time
        end = datetime.utcnow().replace(tzinfo=tz.tzutc())
        end -= timedelta(seconds=end.second, microseconds=end.microsecond)
        start = end - timedelta(seconds=granularity * 299)
    elif not isinstance(start, datetime):
        raise ValueError('start must be datetime')
    elif not isinstance(end, datetime):
        raise ValueError('end must be datetime')

    # convert start and end to UTC
    if start.tzinfo is None:
        start = start.replace(tzinfo=tz.tzlocal())
    if start.tzinfo != tz.tzutc():
        start = start.astimezone(tz.tzutc())
    if end.tzinfo is None:
        end = end.replace(tzinfo=tz.tzlocal())
    if end.tzinfo != tz.tzutc():
        end = end.astimezone(tz.tzutc())

    # return from single call if small enough (API limit is 300 candles)
    if (end - start) / timedelta(seconds=granularity) < 300:
        df = _get_candles(product_id, start, end, granularity)
    # otherwise, chunk into multiple calls
    else:
        df = pd.DataFrame()
        dates = pd.date_range(start, end + timedelta(seconds=300 * granularity),
                              freq=f'{int(300*granularity/60)}min')
        for i in range(len(dates) - 1):
            s = dates[i]
            e = dates[i+1] - timedelta(minutes=1)
            df_ = _get_candles(product_id, s, e, granularity)
            df = df.append(df_)

    # trim extra candles from end of range and sort by time
    df = df.loc[start:end]
    df.sort_index(inplace=True)

    # locate missing data, if any
    idx_diff = np.diff(df.index)
    min_diff = idx_diff[idx_diff > pd.Timedelta(0)].min()
    dr = pd.date_range(df.index.min(), df.index.max(), freq=min_diff)
    df = df.reindex(dr)
    idx = df[df.isna().all(1)].index

    # if any data is missing, forward fill (last close price, 0 volume)
    if len(idx) > 0:
        df.loc[idx, 'volume'] = 0
        df['close'] = df['close'].ffill()
        df.loc[idx] = df.loc[idx].bfill(axis=1)

    return df


def download_minute_bars(product_id: str,
                         start: datetime,
                         end: datetime,
                         fname: str = None):
    if os.path.exists(fname):
        return

    df = get_candles(product_id, start, end, 60)
    if fname is None:
        fname = f'{product_id}_1m_bars.pickle'
        print(f'saving candles to ./{fname}')
        df.to_pickle(fname)
    else:
        df.to_pickle(fname)
