import numpy as np
import warnings

from datetime import datetime
from dateutil.parser import parse
from scipy.optimize import fsolve
from scipy.stats import norm

np.seterr(divide='ignore')
warnings.filterwarnings('ignore')

CALL = 1
PUT = -1

cdf = norm.cdf
pdf = norm.pdf


class Option:
    '''
    Class to wrap option contract information and functions.
    '''

    def __init__(self, sym: str, exp: datetime, strike: int,
                 option_type: int):
        '''
        Parameters
        ----------
        sym : str
            Underlying symbol.
        exp : datetime
            Option expiration.
        strike : int
            Strike (exercise) price.
        option_type : int
            Use this module's CALL and PUT variables.
        '''
        assert isinstance(sym, str), 'sym must be string'
        assert isinstance(exp, (datetime, str)
                          ), 'exp must be datetime or string'
        assert isinstance(strike, (int, float)), 'strike must be int or float'
        assert option_type in [
            CALL, PUT], 'option_type must be 1 (call) or -1 (put)'

        self._symbol = sym
        if isinstance(exp, datetime):
            self._expiration = exp
            self._expiration_str = exp.strftime('%d%b%y').lstrip('0').upper()
        elif isinstance(exp, str):
            self._expiration = parse(exp, dayfirst=True)
            self._expiration_str = self._expiration.strftime(
                '%d%b%y').lstrip('0').upper()
        self._strike = strike
        self._type = option_type
        self._dte = None

    @property
    def symbol(self):
        return self._symbol

    @property
    def expiration(self):
        return self._expiration

    @property
    def expiration_str(self):
        return self._expiration_str

    @property
    def strike(self):
        return self._strike

    @property
    def type(self):
        return self._type

    @property
    def dte(self):
        return self._dte

    @dte.setter
    def dte(self, dte_):
        self._dte = dte_

    def price(self, S, tau, r, sigma):
        if self._type == CALL:
            return call(S, tau, r, sigma, self._strike)
        else:
            return put(S, tau, r, sigma, self._strike)

    def iv(self, S, tau, r, price):
        if self._type == CALL:
            return call_iv(S, tau, r, self._strike, price)
        else:
            return put_iv(S, tau, r, self._strike, price)

    def delta(self, S, tau, r, sigma):
        if self._type == CALL:
            return call_delta(S, tau, r, sigma, self._strike)
        else:
            return put_delta(S, tau, r, sigma, self._strike)

    def gamma(self, S, tau, r, sigma):
        return gamma(S, tau, r, sigma, self._strike)

    def theta(self, S, tau, r, sigma):
        if self.type == CALL:
            return call_theta(S, tau, r, sigma, self._strike)
        else:
            return put_theta(S, tau, r, sigma, self._strike)

    def vega(self, S, tau, r, sigma):
        return vega(S, tau, r, sigma, self._strike)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'Option({self._symbol}-{self._expiration_str}-' \
            f'{self._strike}-{"C" if self._type == CALL else "P"})'


def _d1(S, tau, r, sigma, K):
    return (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))


def _d2(S, tau, r, sigma, K):
    return _d1(S, tau, r, sigma, K) - sigma*np.sqrt(tau)


def call(S: float, tau: float, r: float, sigma: float, K: float) -> float:
    '''
    Compute call price under Black-Scholes model.

    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    tau : float
        Time to expiry (years).
    r : float
        Annualized risk-free rate.
    sigma : float
        Annualized volatility.
    K : float
        Strike (exercise) price.

    Returns
    -------
    float
        Black-Scholes price.
    '''
    C = cdf(_d1(S, tau, r, sigma, K))*S - \
        cdf(_d2(S, tau, r, sigma, K))*K*np.exp(-r*tau)
    return C


def put(S, tau, r, sigma, K):
    '''
    Compute put price under Black-Scholes model.

    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    tau : float
        Time to expiry (years).
    r : float
        Annualized risk-free rate.
    sigma : float
        Annualized volatility.
    K : float
        Strike (exercise) price.

    Returns
    -------
    float
        Black-Scholes price.
    '''
    P = -cdf(-_d1(S, tau, r, sigma, K))*S + \
        cdf(-_d2(S, tau, r, sigma, K))*K*np.exp(-r*tau)
    return P


def call_iv(S, tau, r, K, C):
    iv = fsolve(lambda x: call(S, tau, r, x, K) - C, 0.2)[0]
    return np.maximum(iv, 0)


def put_iv(S, tau, r, K, P):
    iv = fsolve(lambda x: P - put(S, tau, r, x, K), 0.2)[0]
    return np.maximum(iv, 0)


def call_delta(S, tau, r, sigma, K):
    '''
    Compute call delta under Black-Scholes model.

    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    tau : float
        Time to expiry (years).
    r : float
        Annualized risk-free rate.
    sigma : float
        Annualized volatility.
    K : float
        Strike (exercise) price.

    Returns
    -------
    float
        Black-Scholes delta.
    '''
    return cdf(_d1(S, tau, r, sigma, K))


def put_delta(S, tau, r, sigma, K):
    '''
    Compute put delta under Black-Scholes model.

    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    tau : float
        Time to expiry (years).
    r : float
        Annualized risk-free rate.
    sigma : float
        Annualized volatility.
    K : float
        Strike (exercise) price.

    Returns
    -------
    float
        Black-Scholes delta.
    '''
    return call_delta(S, tau, r, sigma, K) - 1


def gamma(S, tau, r, sigma, K):
    '''
    Compute option gamma under Black-Scholes model.

    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    tau : float
        Time to expiry (years).
    r : float
        Annualized risk-free rate.
    sigma : float
        Annualized volatility.
    K : float
        Strike (exercise) price.

    Returns
    -------
    float
        Black-Scholes gamma.
    '''
    return pdf(_d1(S, tau, r, sigma, K)) / (S * sigma * np.sqrt(tau))


def call_theta(S, tau, r, sigma, K):
    '''
    Compute call theta under Black-Scholes model.

    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    tau : float
        Time to expiry (years).
    r : float
        Annualized risk-free rate.
    sigma : float
        Annualized volatility.
    K : float
        Strike (exercise) price.

    Returns
    -------
    float
        Black-Scholes theta.
    '''
    a = -S * pdf(_d1(S, tau, r, sigma, K)) * sigma / (2 * np.sqrt(tau))
    b = -r * K * np.exp(-r*tau) * cdf(_d2(S, tau, r, sigma, K))
    return a + b


def put_theta(S, tau, r, sigma, K):
    '''
    Compute put theta under Black-Scholes model.

    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    tau : float
        Time to expiry (years).
    r : float
        Annualized risk-free rate.
    sigma : float
        Annualized volatility.
    K : float
        Strike (exercise) price.

    Returns
    -------
    float
        Black-Scholes theta.
    '''
    a = -S * pdf(_d1(S, tau, r, sigma, K)) * sigma / (2 * np.sqrt(tau))
    b = r * K * np.exp(-r*tau) * cdf(-_d2(S, tau, r, sigma, K))
    return a + b


def vega(S, tau, r, sigma, K):
    '''
    Compute option vega under Black-Scholes model.

    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    tau : float
        Time to expiry (years).
    r : float
        Annualized risk-free rate.
    sigma : float
        Annualized volatility.
    K : float
        Strike (exercise) price.

    Returns
    -------
    float
        Black-Scholes vega.
    '''
    return S * pdf(_d1(S, tau, r, sigma, K)) * np.sqrt(tau)
