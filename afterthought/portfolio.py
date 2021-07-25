import numpy as np
import pandas as pd
import datetime as dt

ANNUAL_DAYS = 252
ANNUAL_WEEKS = 52
ANNUAL_MONTHS = 12
ANNUAL_DICT = {'D': ANNUAL_DAYS,
               'W': ANNUAL_WEEKS,
               'M': ANNUAL_MONTHS}

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
    capital_base: numeric
        capital base we put into the porfolio
    date_start: datetime.datetime
        the time when we start backtesting
    components: list of str
        tikers of securities to trade, such as ['AAPL', 'MSFT', 'AMZN]
    portfolio_name: str
        name of the portfolio
    commission_rate: numeric
        (the feature is not added yet)
    int_position: boolean
        If true, the shares of securities will be rounded to integers.
    """
    def __init__(self, capital_base, date_start, components, 
                 portfolio_name='portfolio', commission_rate=None,
                 int_position=False, benchmark=None):
        
        # -----------------------------------------------
        # initialize parameters
        # -----------------------------------------------
        self.capital_base = capital_base # initial money invested
        if isinstance(components, str):
            components = [components]    # should be list
        self.components = components  # equities in the portfolio
        self.commission_rate = commission_rate
        self.date_start = date_start
        self.component_prices = pd.DataFrame(columns=self.components)
        self.name = portfolio_name
        self.int_position = int_position
        self.benchmark = benchmark
                
        # -----------------------------------------------
        # record portfolio status to series and dataFrames
        # -----------------------------------------------
        
        # temoprary values
        self._portfolio_value = pd.Series(capital_base,index=[date_start])
        self._cash = pd.Series(capital_base,index=[date_start]) 
        self._total_position_value = pd.Series(0,index=[date_start]) 
        self._component_prices = pd.DataFrame(columns=self.components) # empty
        self._positions = pd.DataFrame(0, index=[date_start], columns=self.components)
        self._position_values = pd.DataFrame(0, index=[date_start], columns=self.components)        
        self._weights = pd.DataFrame(0, index=[date_start], columns=self.components)
        self._position_change = pd.DataFrame(columns=self.components) # empty
        self._now = self.date_start
        self._max_value = pd.Series(capital_base,index=[date_start])
        self._drawdown = pd.Series(0, index=[date_start])
        self._relative_drawdown = pd.Series(0, index=[date_start])
        
        # series
        self.portfolio_value_start = pd.Series()
        self.portfolio_value_end = pd.Series()
        self.cash_start = pd.Series()
        self.cash_end = pd.Series() 
        self.total_position_value_start = pd.Series()                         
        self.total_position_value_end =  pd.Series()
        self.max_value = pd.Series()
        self.drawdown_start = pd.Series()
        self.drawdown_end = pd.Series()
        self.relative_drawdown_start = pd.Series()
        self.relative_drawdown_end = pd.Series()
        
        # dataframes
        self.positions_start = pd.DataFrame(columns=self.components)
        self.positions_end = pd.DataFrame(columns=self.components)
        self.position_values_start = pd.DataFrame(columns=self.components)
        self.position_values_end = pd.DataFrame(columns=self.components)        
        self.weights_start = pd.DataFrame(columns=self.components)
        self.weights_end = pd.DataFrame(columns=self.components)       
        
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
        self.__record_starting_values()
        
    def __update_portfolio(self):
        # new timestamp for _positions and _cash
        self._positions.set_index(self._now, inplace=True)
        self._cash.index = self._now
        self._max_value.index = self._now
        # update position_values, potfolio_value and weights
        self._position_values = self._positions * self._component_prices
        self._total_position_value = self._position_values.sum(axis=1)
        self._portfolio_value = self._cash[0] + self._total_position_value
        self._weights = self._position_values / self._portfolio_value[0] # [0] for using only scalar
        # compare maximum portfolio value and calculate drawdown
        self._max_value = (self._max_value if self._max_value[0] >= self._portfolio_value[0] 
                           else self._portfolio_value)
        self._drawdown = self._max_value - self._portfolio_value
        self._relative_drawdown = self._drawdown / self._max_value
    
    def __record_starting_values(self):
        # also record values not seperated into starting value and ending value
        self.component_prices = self.component_prices.append(self._component_prices)
        self.max_value = self.max_value.append(self._max_value)
        # starting values
        self.cash_start = self.cash_start.append(self._cash)
        self.positions_start = self.positions_start.append(self._positions)
        self.position_values_start = self.position_values_start.append(self._position_values)
        self.total_position_value_start = self.total_position_value_start.append(self._total_position_value)
        self.portfolio_value_start = self.portfolio_value_start.append(self._portfolio_value)
        self.weights_start = self.weights_start.append(self._weights)
        # drawdown
        self.drawdown_start = self.drawdown_start.append(self._drawdown)
        self.relative_drawdown_start = self.relative_drawdown_start.append(self._relative_drawdown)
        
    def __record_ending_values(self):
        self.cash_end = self.cash_end.append(self._cash)
        self.positions_end = self.positions_end.append(self._positions)
        self.position_values_end = self.position_values_end.append(self._position_values)
        self.total_position_value_end = self.total_position_value_end.append(self._total_position_value)
        self.portfolio_value_end = self.portfolio_value_end.append(self._portfolio_value)
        self.weights_end = self.weights_end.append(self._weights) 
        self.drawdown_end = self.drawdown_end.append(self._drawdown)
        self.relative_drawdown_end = self.relative_drawdown_end.append(self._relative_drawdown)        
        
    def keep_position(self):
        self.__record_ending_values()

    def _process_order(self, kind, order):
        assert kind in ['to_positions','to_weights', 'to_values']
        assert isinstance(order,pd.Series) | isinstance(order,pd.DataFrame)
        if isinstance(order, pd.Series):
            order = order.to_frame().T
        assert len(order) == 1 # one line dataframe
        assert order.index >= self._now # cannot trade in the past
        if kind == 'to_positions':
            target_positions = order
        elif kind == 'to_weights':
            target_positions = (order * self._portfolio_value[0]
                                / self._component_prices)
        elif kind == 'to_values':
            target_positions = order / self._component_prices

        # constraint for integer positions/shares of securities
        target_positions = target_positions.fillna(0)
        if self.int_position:
            target_positions = np.round(target_positions)

        return target_positions

    def rebalance(self, kind, order):
        target_positions = self._process_order(kind, order)
        # self._commissions = calCommissions(target_positions)
        self._position_change = target_positions - self._positions
        self._cash = self._cash - (self._position_change * self._component_prices).sum(axis=1) # - self._commisions
        self._positions = target_positions
        
        self.__update_portfolio()
        self.__record_ending_values()