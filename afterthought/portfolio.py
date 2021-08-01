import numpy as np
import pandas as pd
import datetime as dt
import pickle
import bz2
from .analyzer import summarize_returns


DATA_PATH = '../backtest/'

class Portfolio():
    """
    Portfolio is the core class for event-driven backtesting. It conducts the
    backtesting in the following order:
    1. Initialization:
        Set the capital base we invest and the securities we
        want to trade.
    2. Receive the price information with .receive_price(): 
        Insert the new price information of each securities so that the
        Portfolio class will calculated and updated the relevant status such
        as the portfolio value and position weights.
    3. Rebalance with .rebalance():
        Depending on the signal, we can choose to change the position
        on each securities.
    4. Keep position with .keep_position():
        If we don't rebalance the portfolio, we need to tell it to keep
        current position at the end of the market.
    Example
    -------
    see Vol_MA.ipynb, Vol_MA_test_robustness.ipynb
    
    Parameters
    ----------
    capital: numeric
        capital base we put into the porfolio
    inception: datetime.datetime
        the time when we start backtesting
    components: list of str
        tikers of securities to trade, such as ['AAPL', 'MSFT', 'AMZN]
    name: str
        name of the portfolio
    is_share_integer: boolean
        If true, the shares of securities will be rounded to integers.
    """
    def __init__(self, capital, inception, components, 
                 name='portfolio', is_share_integer=False):
        
        # -----------------------------------------------
        # initialize parameters
        # -----------------------------------------------
        self.capital = capital # initial money invested
        if isinstance(components, str):
            components = [components]    # should be list
        self.components = components  # equities in the portfolio
        # self.commission_rate = commission_rate
        self.inception = inception
        self.component_prices = pd.DataFrame(columns=self.components)
        self.name = name
        self.is_share_integer = is_share_integer
        # self.benchmark = benchmark
                
        # -----------------------------------------------
        # record portfolio status to series and dataFrames
        # -----------------------------------------------
        
        # temoprary values
        self._nav = pd.Series(capital,index=[inception])
        self._cash = pd.Series(capital,index=[inception]) 
        self._security = pd.Series(0,index=[inception]) 
        self._component_prices = pd.DataFrame(columns=self.components) # empty
        self._shares = pd.DataFrame(0, index=[inception], columns=self.components)
        self._positions = pd.DataFrame(0, index=[inception], columns=self.components)        
        self._weights = pd.DataFrame(0, index=[inception], columns=self.components)
        self._share_changes = pd.DataFrame(columns=self.components) # empty
        self._now = self.inception
        self._max_nav = pd.Series(capital,index=[inception])
        self._drawdown = pd.Series(0, index=[inception])
        self._relative_drawdown = pd.Series(0, index=[inception])
        
        # series
        self.nav_open = pd.Series()
        self.nav_close = pd.Series()
        self.cash_open = pd.Series()
        self.cash_close = pd.Series() 
        self.security_open = pd.Series()                         
        self.security_close =  pd.Series()
        self.max_nav = pd.Series()
        self.drawdown_open = pd.Series()
        self.drawdown_close = pd.Series()
        self.relative_drawdown_open = pd.Series()
        self.relative_drawdown_close = pd.Series()
        
        # dataframes
        self.shares_open = pd.DataFrame(columns=self.components)
        self.shares_close = pd.DataFrame(columns=self.components)
        self.positions_open = pd.DataFrame(columns=self.components)
        self.positions_close = pd.DataFrame(columns=self.components)        
        self.weights_open = pd.DataFrame(columns=self.components)
        self.weights_close = pd.DataFrame(columns=self.components)       
        
    def receive_price(self, prices):
        assert isinstance(prices,pd.Series) | isinstance(prices,pd.DataFrame)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame().T
        assert len(prices) == 1 # one line dataframe
        assert prices.index >= self._now # only receive new prices
        self._now = prices.index # record the current time, format:
                                 # pd.DatetimeIndex(['%Y-%m-%d'])
        self._component_prices = prices
        self.__update_portfolio()
        self.__record_open_values()
        
    def __update_portfolio(self):
        # new timestamp for _shares and _cash
        self._shares.set_index(self._now, inplace=True)
        self._cash.index = self._now
        self._max_nav.index = self._now
        # update positions, potfolio_value and weights
        self._positions = self._shares * self._component_prices
        self._security = self._positions.sum(axis=1)
        self._nav = self._cash[0] + self._security
        self._weights = self._positions / self._nav[0] # [0] to use scalar
        # compare maximum portfolio value and calculate drawdown
        self._max_nav = (self._max_nav
                           if self._max_nav[0] >= self._nav[0] 
                           else self._nav
                           )
        self._drawdown = self._max_nav - self._nav
        self._relative_drawdown = self._drawdown / self._max_nav
    
    def __record_open_values(self):
        # also record values not seperated into open value and close value
        self.component_prices \
            = self.component_prices.append(self._component_prices)
        self.max_nav = self.max_nav.append(self._max_nav)
        # open values
        self.cash_open = self.cash_open.append(self._cash)
        self.shares_open = self.shares_open.append(self._shares)
        self.positions_open = self.positions_open.append(self._positions)
        self.security_open = self.security_open.append(self._security)
        self.nav_open = self.nav_open.append(self._nav)
        self.weights_open = self.weights_open.append(self._weights)
        # drawdown
        self.drawdown_open = self.drawdown_open.append(self._drawdown)
        self.relative_drawdown_open \
            = self.relative_drawdown_open.append(self._relative_drawdown)
        
    def __record_close_values(self):
        self.cash_close = self.cash_close.append(self._cash)
        self.shares_close = self.shares_close.append(self._shares)
        self.positions_close = self.positions_close.append(self._positions)
        self.security_close = self.security_close.append(self._security)
        self.nav_close = self.nav_close.append(self._nav)
        self.weights_close = self.weights_close.append(self._weights) 
        self.drawdown_close = self.drawdown_close.append(self._drawdown)
        self.relative_drawdown_close \
            = self.relative_drawdown_close.append(self._relative_drawdown)        
        
    def keep_position(self):
        self.__record_close_values()

    def _process_order(self, kind, order):
        assert kind in ['to_shares','to_weights', 'to_values']
        assert isinstance(order,pd.Series) | isinstance(order,pd.DataFrame)
        if isinstance(order, pd.Series):
            order = order.to_frame().T
        assert len(order) == 1 # one line dataframe
        assert order.index >= self._now # cannot trade in the past

        order_full = pd.DataFrame(0, columns=self.components,
                                  index=order.index)
        order_full[order.columns] = order

        if kind == 'to_shares':
            target_shares = order_full
        elif kind == 'to_weights':
            target_shares = (order_full * self._nav[0]
                                / self._component_prices)
        elif kind == 'to_values':
            target_shares = order_full / self._component_prices

        # constraint for integer shares of securities
        target_shares = target_shares.fillna(0)
        if self.is_share_integer:
            target_shares = np.round(target_shares)

        return target_shares

    def rebalance(self, kind, order):
        target_shares = self._process_order(kind, order)
        # self._commissions = calCommissions(target_shares)
        self._share_changes = target_shares - self._shares
        self._cash = (self._cash 
                     - (self._share_changes
                        *self._component_prices).sum(axis=1)
                         # - self._commisions
                     )
        self._shares = target_shares
        
        self.__update_portfolio()
        self.__record_close_values()

    @property
    def is_backtested(self):
        return (self._now > self.inception)[0] # get value from array

    @property
    def returns(self):
        if self.is_backtested:
            returns = self.nav_close.pct_change().fillna(0)
            returns.name = self.name
            return returns
        else:
            return None
    
    @property
    def cum_returns(self):
        if self.is_backtested:
            cum_returns = self.nav_close / self.capital
            cum_returns.name = self.name
            return cum_returns
        else:
            return None

    def summarize(self, benchmark_returns=None, period='daily', brief=True):
        """
        Generate performance metrics of the portfolio including:
        [Total Return, Annual Reutrn, Annual Volatility, Max Drawdown, VaR,
        CVaR, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Omega Ratio, Skew,
        Kurtosis].
        If benchmark_return is asigned, it will generate extra metrics,
        including:
        [Alpha, Beta, Excess Sharpe, Capture Ratio, Up Capture, Down Captie]
        Parameters:
        benchmark_returns: Series
            index: datetime
            values: return
        """

        return summarize_returns(self.returns,
                                 benchmark_returns,
                                 name=self.name,
                                 period=period,
                                 brief=brief)

    def to_pickle(self, path=DATA_PATH, name=None):
        if name is None: name = self.name
        file = path + name + '.port'
        with bz2.BZ2File(file, 'wb') as f:
            pickle.dump(self, f)