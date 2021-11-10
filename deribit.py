import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import requests
import warnings

from datetime import datetime, timedelta
from dateutil.parser import parse
from requests.models import HTTPError

URL = 'https://deribit.com/api/v2'
RATE_LIMIT_PER_SEC = 20


def validate_response(resp: requests.Response):
    '''
    validate request response
    '''
    assert resp.status_code == 200, resp.raise_for_status()
    return resp


def get_instruments(currency, kind=None, expired=None) -> dict:
    '''
    get list of instruments by currency and kind
    '''
    # handle lists of currencies and kinds
    if isinstance(currency, list):
        results = []
        for ccy in currency:
            if isinstance(kind, list):
                for k in kind:
                    results += get_instruments(ccy, k, expired)
            else:
                results += get_instruments(ccy, kind, expired)
        return results

    # convert expired flag to string (API requirement)
    if isinstance(expired, bool):
        expired = str(expired).lower()

    # create parameters dict
    params = {
        'currency': currency,
        'kind': kind,
        'expired': expired
    }

    # send GET request and validate response
    resp = validate_response(
        requests.get(URL + '/public/get_instruments', params=params)
    )

    # get list of results and extract instrument names
    results = resp.json().get('result')
    symbols = list(map(lambda x: x.get('instrument_name'), results))

    return symbols


def get_candles(instrument_name, start_timestamp, end_timestamp, resolution):
    '''
    get price candles based on instrument name, start time, end time, and time resolution

    Parameters:
        instrument_name : str
            symbol of instrument as returned by "get_instruments"
        start_timestamp : str, datetime, int
            date string, datetime object, or integer timestamp of
            the start time of the time range
        end_timestamp : str, datetime, int
            date string, datetime object, or integer timestamp of
            the end time of the time range
        resolution : str, int

    '''
    if isinstance(start_timestamp, str):
        start_timestamp = round(parse(start_timestamp).timestamp() * 1000)
    elif isinstance(start_timestamp, datetime):
        start_timestamp = round(start_timestamp.timestamp() * 1000)

    if isinstance(end_timestamp, str):
        end_timestamp = round(parse(end_timestamp).timestamp() * 1000)
    elif isinstance(end_timestamp, datetime):
        end_timestamp = round(end_timestamp.timestamp() * 1000)

    if isinstance(resolution, int):
        resolution = str(resolution)

    resolution_choices = ['1', '3', '5', '10', '15',
                          '30', '60', '120', '180', '360', '720', '1D']
    if not resolution in resolution_choices:
        raise ValueError(
            f'resolution must be one of {", ".join(resolution_choices)}')

    params = {
        'instrument_name': instrument_name,
        'start_timestamp': start_timestamp,
        'end_timestamp': end_timestamp,
        'resolution': resolution
    }

    resp = validate_response(
        requests.get(URL + '/public/get_tradingview_chart_data', params=params)
    )

    status = resp.json().get('status')
    if status == 'no_data':
        warnings.warn(f'No data: {instrument_name}')
        return

    res = resp.json().get('result')
    res.pop('status')
    res.pop('cost')

    df = pd.DataFrame(res)
    df['time'] = df.ticks.apply(
        lambda x: datetime.fromtimestamp(round(x / 1000)))
    df.drop('ticks', axis=1, inplace=True)
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    return df


def daterange(start_date: datetime, end_date: datetime, delta: timedelta = timedelta(days=1)):
    for n in range(int((end_date - start_date) / delta)):
        yield start_date + delta * n


def map_contracts(ccy, dr):
    date_fmt = '%d%b%y'
    contract_map = {}

    for dt in dr:
        # get perp data
        perp = get_candles(f'{ccy}-PERPETUAL', dt, dt, '1D')

        # build symbol
        dstr = dt.strftime(date_fmt).lstrip('0').upper()
        tp = ((perp['high'] + perp['low'] + perp['close']) / 3).mean()
        strike = int(round(tp / 1000) * 1000)

        contract_found = False
        attempts = 0

        while not contract_found and attempts < 2:
            sym = f'{ccy}-{dstr}-{strike}-C'

            try:
                # check contract availability
                get_candles(sym, dt - timedelta(days=1), dt, '1D')

                # add to map
                contract_map.setdefault(ccy, {})
                contract_map.get(ccy).setdefault(dstr, [])
                contract_map.get(ccy).get(dstr).append(strike)

                contract_found = True
                print(sym, 'found')
            except HTTPError as exc:
                print(sym, exc.response.json().get(
                    'error').get('data').get('reason'))
                strike += 1000
                attempts += 1

    return contract_map


def main():
    map_contracts('BTC', daterange(
        datetime(2021, 10, 1), datetime(2021, 11, 1)))


if __name__ == '__main__':
    main()
