"""
WORK IN PROGRESS
"""

import black_scholes as bs
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import volatility as vola

from datetime import datetime, timedelta
from dateutil.parser import parse

DATA_PATH = './data'
OPT_SPREADS = {'BTC': 0.029, 'ETH': 0.032}  # Deribit
TICK_SIZES = {'BTC': 0.0005, 'ETH': 0.005}  # Deribit
MIN_LOT_SIZES = {'BTC': 0.1, 'ETH': 1}  # Deribit
MIN_LOT_SIZES_SPOT = {'BTC': 0.00000001, 'ETH': 0.00000001}  # Coinbase


class VolStrategy:
    def __init__(self, ccy: str, vol_pd: int, lookback_pd: int,
                 K0: float, q_spread: float, entry_exit_threshold: float,
                 bt_start_date: str, bt_end_date: str,
                 opt_trade_fee_rate: float, spot_trade_fee_rate: float,
                 max_position_size: float, rehedge_threshold_delta: float,
                 ewm=False):
        '''
        [summary]

        Parameters
        ----------
        ccy : str
            Underlying currency.
        vol_pd : int
            Period for calculating volatility, e.g. 30-, 60-day vol, etc.
        lookback_pd : int
            [description]
        K0 : float
            Initial strategy capital.
        q_spread : float
            Historical spread quantile for computing entry/exit signal.
        entry_exit_threshold : float
            Spread deviation for entry/exit.
        bt_start_date : str
            Backtest start date.
        bt_end_date : str
            Backtest end date.
        opt_trade_fee_rate : float
            Trading fee rate for option trades (fraction of notional value
            quote currency).
        spot_trade_fee_rate : float
            Trading fee rate for spot trades.
        max_position_size : float
            Max position size (fraction of total capital).
        rehedge_threshold_delta : float
            Total net delta above which rehedging should occur.
            E.g. if the total net delta of the portfolio is 0.02, rehedge
            when possible.
        ewm : bool, optional
            Flag to indicate if exponential weighting should be used in
            moving window computations, by default False.
        '''
        # store inputs
        self.ccy = ccy
        self.vol_pd = vol_pd
        self.agg_pd = lookback_pd
        self.capital = K0
        self.q_spread = q_spread
        self.entry_exit_threshold = entry_exit_threshold
        self.bt_start_date = bt_start_date
        self.bt_end_date = bt_end_date
        self.opt_trade_fee_rate = opt_trade_fee_rate
        self.spot_trade_fee_rate = spot_trade_fee_rate
        self.max_position_size = max_position_size
        self.rehedge_threshold_delta = rehedge_threshold_delta
        self.ewm = ewm

        # set long and short levels
        self.short_level = 1 + self.entry_exit_threshold
        self.long_level = 1 - self.entry_exit_threshold

        # initialize variables
        self.signals = None
        self.spot = None
        self.vol = None
        self.vol_spread = None
        self.opt_data = None
        self._pnl = None
        self.backtest_dates = None
        self.pos_starting_pfl_value = None

        self.open_positions = False
        self.stopped_out = False

        # initialize containers
        self._trades = pd.DataFrame()
        self._net_pfl_value = pd.Series(dtype=float)
        self.positions = {}
        self.prices_base = {}

        # find options data files for underlying currency
        files = list(filter(lambda x: re.search(
            self.ccy + '_(?P<exp>\d+\w{3}\d{2})_*', x), os.listdir(DATA_PATH))
        )
        self.option_files = sorted(files, key=lambda x: parse(
            x.split('_')[1], dayfirst=True
        ))

        # read option expirations
        self.expirations = np.array(
            sorted(map(lambda x: parse(x.split('_')
                   [1], dayfirst=True), self.option_files))
        )

        # store constants
        self.K0 = K0
        self.DATA_PATH = DATA_PATH
        self.OPT_SPREAD_RATE = OPT_SPREADS.get(self.ccy)
        self.SPOT_SPREAD_RATE = 0  # assume negligible spread for spot trade
        self.TICK_SIZE = TICK_SIZES.get(self.ccy)
        self.MIN_LOT_SIZE = MIN_LOT_SIZES.get(self.ccy)
        self.MIN_LOT_SIZE_SPOT = MIN_LOT_SIZES_SPOT.get(self.ccy)

        self.POSITION_STOP_LIMIT = -0.015  # TODO: parameterize this value

    @property
    def trades(self):
        return self._trades

    @property
    def net_pfl_value(self):
        return self._net_pfl_value

    @property
    def pnl(self):
        return self._pnl

    def backtest(self):
        self.load_spot_data()
        self.compute_vol()
        self.compute_vol_spread()
        self.compute_signals()
        self.compute_backtest_dates()

        self._net_pfl_value[self.signals.index[0]] = self.capital

        for dt in self.backtest_dates:
            diff = int(self.signals.loc[dt, 'diff'])
            signal = int(self.signals.loc[dt, 'signal'])

            if diff != 0:  # if entry/exit signal
                if not self.open_positions:  # if no positions open
                    if signal != 0:  # if long/short signal
                        self.enter_positions(dt, signal)
                else:  # if positions open
                    self.close_positions(dt)
            elif signal != 0:  # if long/short signal
                if self.stopped_out:  # if we're stopped out
                    continue
                elif not self.open_positions:  # if no positions open
                    self.enter_positions(dt, signal)
                else:  # if positions open
                    self.update_positions(dt)

        self.compute_pnl()

    def load_spot_data(self):
        if self.spot is not None:
            return

        self.spot = pd.read_pickle(
            self.DATA_PATH + f'/{self.ccy}_1m_bars.pickle')

    def find_option(self, d: datetime):
        i_exp = np.where(self.expirations > d + timedelta(days=90))[0].min()
        opt_file = self.option_files[i_exp]
        opt = self.load_option_file(opt_file)
        while not opt.index.min() < d:
            i_exp -= 1
            opt_file = self.option_files[i_exp]
            opt = self.load_option_file(opt_file)

        exp = opt_file.split('_')[1]
        self.opt_data = opt

        tp = self.spot.eval('(high + low + close) / 3').loc[d]
        tp_diff = (opt['strike'] - tp).abs().min()

        if (int(tp + tp_diff) == opt.loc[d, 'strike']).any():
            strike = int(tp + tp_diff)
        elif (int(tp - tp_diff) == opt.loc[d, 'strike']).any():
            strike = int(tp - tp_diff)

        opt = opt.loc[d]
        opt = opt[opt['strike'] == strike]

        return opt, exp

    def load_option_file(self, file) -> pd.DataFrame:
        fname = f'{self.DATA_PATH}/{file}'
        if not os.path.exists(fname):
            raise FileNotFoundError(fname)
        return pd.read_pickle(fname)

    # -------------------------- Compute Quantities --------------------------

    def compute_straddle_price(self, call_price: float, put_price: float):
        return call_price + put_price

    def compute_max_allowed_qty(self, trade_cost_base: float):
        '''
        Compute maximum allowed quantity for a trade based on trade cost and 
        risk limits.

        Parameters
        ----------
        trade_cost_base : float
            Cost of trade in base currency.

        Returns
        -------
        float
            Maximum allowed trade quantity based on risk limits.
        '''
        # max trade size in base currency
        max_trade_size_base = self.capital * self.max_position_size

        # compute max allowed quantity for trade
        max_allowed_qty = max_trade_size_base / trade_cost_base
        max_allowed_qty = np.floor(
            max_allowed_qty / self.MIN_LOT_SIZE) * self.MIN_LOT_SIZE

        return max_allowed_qty

    def compute_gross_fees(self, price, qty, option=True):
        # compute notional
        notional = np.abs(price * qty)

        if option:  # if option
            spread_fee = notional * self.OPT_SPREAD_RATE
            exchange_fee = notional * \
                np.minimum(self.opt_trade_fee_rate, price * 0.125)
        else:  # if spot
            spread_fee = notional * self.SPOT_SPREAD_RATE
            exchange_fee = notional * self.spot_trade_fee_rate

        return spread_fee + exchange_fee

    def compute_straddle_fees(self, call_price, put_price, qty):
        call_fee = self.compute_gross_fees(call_price, qty)
        put_fee = self.compute_gross_fees(put_price, qty)
        return call_fee + put_fee

    def compute_straddle_delta(self, call, call_price, put, put_price,
                               spot_price, tau):
        # compute call delta
        call_iv = call.iv(spot_price, tau, 0, call_price)
        call_delta = call.delta(spot_price, tau, 0, call_iv)

        # compute put delta
        put_iv = put.iv(spot_price, tau, 0, put_price)
        put_delta = put.delta(spot_price, tau, 0, put_iv)

        return call_delta + put_delta

    def compute_vol(self):
        if self.vol is not None:
            return
        if self.spot is None:
            self.load_spot_data()

        self.vol = vola.yang_zhang(
            self.spot, self.vol_pd, ewm=self.ewm).loc[
                self.bt_start_date:self.bt_end_date]

    def compute_vol_spread(self):
        if self.vol_spread is not None:
            return
        if self.vol is None:
            self.compute_vol()

        self.vol_spread = self.vol / \
            self.vol.rolling(self.agg_pd).quantile(self.q_spread)

    def compute_signals(self):
        if self.signals is not None:
            return
        if self.vol is None:
            self.compute_vol()
        if self.vol_spread is None:
            self.compute_vol_spread()

        self.signals = pd.DataFrame()
        self.signals['signal'] = -1 * (self.vol_spread > self.short_level)\
            .astype(int) + (self.vol_spread < self.long_level).astype(int)
        self.signals['diff'] = self.signals['signal'].diff()

    def compute_backtest_dates(self):
        signal_idx = self.signals[(self.signals['signal'] != 0)].index
        diff_idx = self.signals[
            (self.signals['diff'] != 0) & (~self.signals['diff'].isna())
        ].index
        self.backtest_dates = signal_idx.union(diff_idx).unique().sort_values()

    def compute_pfl_delta(self):
        # load spot price/qty
        spot_price = self.prices_base.get(self.ccy)
        spot_qty = self.positions.get(self.ccy)

        # compute total portfolio delta
        total_net_delta = spot_qty
        for opt in self.positions.keys():
            if not isinstance(opt, bs.Option):
                continue

            tau = opt.dte / 365  # time to exp (years)
            price = self.prices_base.get(opt)
            qty = self.positions.get(opt)
            iv = opt.iv(spot_price, tau, 0, price)
            delta = opt.delta(spot_price, tau, 0, iv)
            total_net_delta += delta * qty

        return total_net_delta

    def compute_pfl_gamma(self):
        # load spot price/qty
        spot_price = self.prices_base.get(self.ccy)

        # compute total portfolio delta
        total_net_gamma = 0
        for opt in self.positions.keys():
            if not isinstance(opt, bs.Option):
                continue

            tau = opt.dte / 365  # time to exp (years)
            price = self.prices_base.get(opt)
            qty = self.positions.get(opt)
            iv = opt.iv(spot_price, tau, 0, price)
            gamma = opt.gamma(spot_price, tau, 0, iv)
            total_net_gamma += gamma * qty

        return total_net_gamma

    # ------------------------------- Positions -------------------------------

    def enter_positions(self, dt: datetime, long_short_signal):
        # reset stop flag
        self.stopped_out = False

        # find liquid contract close to atm
        options, exp = self.find_option(dt)
        dte = options['dte'].iloc[0]
        strike = int(options['strike'].iloc[0])

        # get spot price
        spot_price = self.spot.loc[:dt, 'close']
        spot_price = spot_price.loc[spot_price.last_valid_index()]

        # get option prices
        atm_call = options[options['cp_flag'] == 'C']
        atm_put = options[options['cp_flag'] == 'P']
        if atm_call.empty or atm_put.empty:
            return

        C_quote = atm_call.eval('(open + high + low + close) / 4').iloc[0]
        P_quote = atm_put.eval('(open + high + low + close) / 4').iloc[0]

        C_base = C_quote * spot_price
        P_base = P_quote * spot_price

        # set position initial portfolio value
        self.pos_starting_pfl_value = self.capital

        # Option trade -------------------------------------------------------
        # compute trade price and allowed quantity - straddle
        trade_price_quote = self.compute_straddle_price(C_quote, P_quote)
        trade_price_base = trade_price_quote * spot_price
        max_allowed_qty = self.compute_max_allowed_qty(trade_price_base)

        # compute fees - straddle
        trade_fees_quote = self.compute_straddle_fees(
            C_quote, P_quote, max_allowed_qty)
        trade_fees_base = trade_fees_quote * spot_price

        # total trade cost: fees + trade cost
        total_trade_cost_base = trade_fees_base + trade_price_base \
            * max_allowed_qty

        # while total cost exceeds risk limit
        while (total_trade_cost_base / self.capital > self.max_position_size):
            # reduce position size
            max_allowed_qty -= self.MIN_LOT_SIZE

            # recompute fees and total cost
            trade_fees_quote = self.compute_straddle_fees(
                C_quote, P_quote, max_allowed_qty)
            trade_fees_base = trade_fees_quote * spot_price
            total_trade_cost_base = trade_fees_base + trade_price_base \
                * max_allowed_qty

        # record option positions and prices
        call = bs.Option(self.ccy, exp, strike, bs.CALL)
        put = bs.Option(self.ccy, exp, strike, bs.PUT)

        self.positions[call] = max_allowed_qty * long_short_signal
        self.positions[put] = max_allowed_qty * long_short_signal

        self.prices_base[call] = C_base
        self.prices_base[put] = P_base

        # assess fees and position change
        self.capital -= total_trade_cost_base

        # record straddle trade
        self.record_trade(trade_price_quote, max_allowed_qty,
                          spot_price, trade_fees_base, long_short_signal)

        # Hedge --------------------------------------------------------------
        # time to expiration in years
        tau = dte / 365

        # compute hedge quantity
        trade_net_delta = self.compute_straddle_delta(
            call, C_base, put, P_base, spot_price, tau)
        total_net_delta = trade_net_delta * max_allowed_qty
        hedge_qty = np.round(-total_net_delta /
                             self.MIN_LOT_SIZE_SPOT) * self.MIN_LOT_SIZE_SPOT

        # record hedge position and price
        self.positions[self.ccy] = hedge_qty
        self.prices_base[self.ccy] = spot_price

        # compute hedge fees
        hedge_fees_base = self.compute_gross_fees(spot_price, hedge_qty,
                                                  option=False)

        # assess hedge position changes
        self.capital -= hedge_fees_base
        self.capital -= spot_price * hedge_qty

        # record hedge trade
        self.record_trade(1, hedge_qty, spot_price, hedge_fees_base,
                          -np.sign(total_net_delta))

        # compute portfolio value after trades
        self.update_pfl_value(dt)

        self.open_positions = True

    def update_positions(self, dt: datetime):
        # update prices of holdings
        self.update_prices(dt)

        # rehedge if necessary
        self.check_rehedge()

        # update pfl value
        self.update_pfl_value(dt)

        # check drawdown
        self.check_drawdown(dt)

        return

    def close_positions(self, dt: datetime):
        # update prices and portfolio value
        self.update_prices(dt)
        self.update_pfl_value(dt)

        # close spot position ------------------------------------------------
        spot_price = self.prices_base.pop(self.ccy)
        spot_qty = self.positions.pop(self.ccy)

        # compute fee
        spot_fee_quote = self.compute_gross_fees(1, spot_qty, False)
        spot_fee_base = spot_fee_quote * spot_price

        # assess position changes and fee
        self.capital += spot_price * spot_qty
        self.capital -= spot_fee_base

        # record spot trade
        self.record_trade(1, np.abs(spot_qty), spot_price,
                          spot_fee_base, np.sign(spot_qty))

        # close option positions
        for opt in self.positions.copy().keys():
            if not isinstance(opt, bs.Option):
                continue

            # get price and qty
            opt_price_base = self.prices_base.pop(opt)
            opt_price_quote = opt_price_base / spot_price
            opt_qty = self.positions.pop(opt)

            # compute fee
            opt_fee_quote = self.compute_gross_fees(opt_price_quote, opt_qty)
            opt_fee_base = opt_fee_quote * spot_price

            # assess position change and fee
            self.capital += opt_price_base * opt_qty - opt_fee_base

            # record option trade
            self.record_trade(opt_price_quote, opt_qty, spot_price,
                              opt_fee_base, np.sign(opt_qty))

        # set positions flag off
        self.open_positions = False

    def update_prices(self, dt: datetime):
        # update spot price
        spot_price = self.spot.loc[:dt, 'close']
        spot_price = spot_price.loc[spot_price.last_valid_index()]

        self.prices_base[self.ccy] = spot_price

        # update option prices
        opt_prices = self.opt_data.loc[dt]
        for opt in self.positions.keys():
            if not isinstance(opt, bs.Option):
                continue

            k = opt.strike
            cp = 'C' if opt.type == bs.CALL else 'P'
            opt_price_quote = opt_prices[
                (opt_prices['strike'] == k) & (opt_prices['cp_flag'] == cp)
            ].loc[dt, 'close']
            opt.dte = opt_prices.iloc[0]['dte']
            opt_price_base = opt_price_quote * spot_price

            self.prices_base[opt] = opt_price_base

    def record_trade(self, trade_price_quote: float, trade_quantity: float,
                     spot_price: float, fee_base: float, direction: int):
        self._trades = self._trades.append({
            'trade_price_quote': trade_price_quote,
            'qty': trade_quantity,
            'spot_price': spot_price,
            'fee_base': fee_base,
            'direction': int(-np.sign(direction))
        }, ignore_index=True)

    def check_rehedge(self):
        # compute current portfolio delta
        pfl_delta = self.compute_pfl_delta()

        # rehedge if delta is outside bounds
        if np.abs(pfl_delta) > self.rehedge_threshold_delta:
            self.rehedge(pfl_delta)

    def rehedge(self, delta):
        # get spot price
        spot_price = self.prices_base.get(self.ccy)

        # compute net hedge trade quantity
        net_trade_qty = -delta  # opposite trade to offset delta
        net_trade_qty = np.round(
            net_trade_qty / self.MIN_LOT_SIZE_SPOT) * self.MIN_LOT_SIZE_SPOT

        # compute hedge trade fees
        fees_base = self.compute_gross_fees(
            spot_price, net_trade_qty, option=False)

        # assess and record trade
        self.positions[self.ccy] += net_trade_qty
        self.record_trade(1, np.abs(net_trade_qty), spot_price,
                          fees_base, np.sign(net_trade_qty))

    def update_pfl_value(self, dt: datetime):
        # sum net position value
        net_pos_value = 0
        for k in self.positions.keys():
            net_pos_value += self.positions.get(k) * self.prices_base.get(k)

        self._net_pfl_value[dt] = net_pos_value + self.capital

    def check_drawdown(self, dt: datetime):
        position_drawdown = self.net_pfl_value.iloc[-1] / \
            self.pos_starting_pfl_value - 1
        if position_drawdown < self.POSITION_STOP_LIMIT:
            self.close_positions(dt)
            self.stopped_out = True

    def compute_pnl(self):  # TODO: implement function
        if self._trades.empty:
            self._pnl = 0
            return

    # ------------------------------- Plot Data -------------------------------

    def plot_vol(self):
        if self.vol is None:
            self.compute_vol()
        self.vol.plot()
        plt.grid(True)
        plt.show(block=False)

    def plot_vol_spread(self):
        if self.vol_spread is None:
            self.compute_vol_spread()
        self.vol_spread.plot()
        plt.axhline(self.short_level, color='r')
        plt.axhline(self.long_level, color='r')
        plt.grid(True)
        plt.show()

    def plot_signals(self):
        if self.signals is None:
            self.compute_signals()
        self.signals.plot(subplots=True)
        plt.grid(True)
        plt.show()

    def plot_net_pfl_value(self, norm=False):
        if norm:
            net_pfl_value = self.net_pfl_value / self.K0
        else:
            net_pfl_value = self.net_pfl_value

        net_pfl_value.plot(marker='.', drawstyle='steps-post')
        plt.grid(True)
        plt.show()


def test_btc_strategy():
    vol_pd = 30 * 60 * 24  # n-day volatility
    lookback_pd = 5 * 60 * 24  # m-day aggregation period
    starting_capital = 100_000
    spread_quantile_entry_exit = 0.95
    spread_deviation_entry_exit_threshold = 0.1
    backtest_start_date = '2020-1-1'
    backtest_end_date = '2021-7-1'
    opt_trade_fee_rate = 0.0003
    spot_trade_fee_rate = 0.0008
    max_position_size = 0.02
    rehedge_threshold_delta = 0.1

    btc_vol_strat = VolStrategy('BTC',
                                vol_pd,
                                lookback_pd,
                                starting_capital,
                                spread_quantile_entry_exit,
                                spread_deviation_entry_exit_threshold,
                                backtest_start_date,
                                backtest_end_date,
                                opt_trade_fee_rate,
                                spot_trade_fee_rate,
                                max_position_size,
                                rehedge_threshold_delta)
    # btc_vol_strat.plot_vol()
    # btc_vol_strat.plot_vol_spread()
    btc_vol_strat.backtest()
    pnl = btc_vol_strat.compute_pnl()


if __name__ == '__main__':
    test_btc_strategy()
