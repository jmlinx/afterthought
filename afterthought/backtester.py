import numpy as np
import pandas as pd
import datetime as dt
import copy
from tqdm import tqdm
from .portfolio import Portfolio
from .selector import Selector
from .allocator import Allocator
from .trigger import * 

class Backtester():
    """
    Backtestor is used to perform backtesting algorithm. At the initialization,
    method, Trigger, Allocator must be sepecified. Initializating the
    Backtestor by classmethod is prefered, see the classmethod for the
    instruction. After initialization, it takes in an initilized Porfolio 
    class and a price dataframe as parameters.
    
    Parameters:
    -----------
    Portfolio: .Portfolio Object
    price_df: pandas.Dataframe
        the columns should match the Portfolio.component_stocks
        the time index should cover the Portfolio.inception
    """
    def __init__(self, method, window_lookback, Trigger, Allocator,
                 Selector=None, Controller=None, drawdown_tolerance=None,
                 **kwargs):

        self.method = method
        self.window_lookback = window_lookback
        self.drawdown_tolerance = drawdown_tolerance
        self.trigger = Trigger
        self.allocator = Allocator
        self.selector = Selector
        self.controller = Controller       
        self.__dict__.update(kwargs)
        self._set_functor()
    
    def _set_functor(self):
        self._functor_dict = {'ByPeriod': self._reblance_by_period,
                              'BuyAndHold': self._buy_and_hold,
                              }
        self._functor = self._functor_dict[self.method]
    
    def __call__(self, Portfolio, price_df):
        return self._functor(Portfolio, price_df)

    @classmethod
    def ByPeriod(cls, window_lookback, Scheduler, Allocator, Selector=None,
                   Controller=None, drawdown_tolerance=None):
        """
        Initialize the Backtestor to perform backtesting by schedule.
        ByPeriod algorithm will perform rebalancing at a certain date,
        which is determined by the Scheduler. At the rebalancing day, it will
        call the Selector (optional), Allocator and Controller (optional) in
        order. If a Selector is input, the Backtestor will first implement the 
        Selector on the price to pick up ideal securities. Then it will use
        Allocator to determine the weight allocated on those securities. If
        a Controller is input, it will reduce or eliminate the weight on
        underperforming securities decided by the Controller.
        Parameters:
        -----------
        window_lookback: int
            days to look back, i.e. how many days of historical data is used
            for analyse at the rebalancing date
        Scheduler: Scheduler object
            it triggers the Backtestor to rebalance
        Allocator: Allocator object
            it determines the weights to investe in the securities
        Selector: (Optional) Selector object
            it picks out certain number of the top securities among the
            security universe
        Controller: (Optional) Controller object
            it makes the final adjustment on the weights
        drawdown_tolerance: float between 0 and 1
            if the portfolio drawdown reaches the value of drawdown_tolerance,
            the portfolio will clear all the positions for this holding period.
            it might enter the market again after next rebalance date.
        """
        return cls('ByPeriod', window_lookback, Scheduler, Allocator,
                   Selector, Controller, drawdown_tolerance)
    
    def _reblance_by_period(self, Portfolio, price_df):
        # asure no NaN for certain stocks at some dates
        price_df = self._process_price(price_df)
        # backtest starts from the start time of the portfolio
        index_time = price_df[Portfolio.inception:].index
        _drawdown_tolerance = self.drawdown_tolerance
        for time in tqdm(index_time,
                         desc='Backtesting {}'.format(Portfolio.name)):
            
            # ----------------------------------------
            #  update price and timer
            # ----------------------------------------
            
            self.trigger.update_scheduler(time)
            price_now = price_df.loc[time]
            Portfolio.receive_price(price_now)

            # ----------------------------------------
            #  rebalance
            # ----------------------------------------
            
            # if is time to trade, use Selector -> Allocator -> Controller             
            if self.trigger.is_trade_time():
                price_slice = price_df.loc[:time].tail(self.window_lookback)

                if self.selector:
                     selected = self.selector(price_slice)
                     price_slice = price_slice.loc[:, selected]

                weight = self.allocator(price_slice)

                if self.controller:
                    weight = self.controller(weight, price_slice)
                    
                Portfolio.rebalance('to_weights', weight) 
                
                continue # end action today
            
            # ----------------------------------------
            #  control drawdown
            # ----------------------------------------
            if self.drawdown_tolerance:
                # reset drawdown tolerance level if RD recovers to zero
                if (Portfolio._relative_drawdown[0] == 0 and 
                    _drawdown_tolerance!= self.drawdown_tolerance):
                    _drawdown_tolerance = self.drawdown_tolerance

                # if RD reach the drawdown_tolerance, clear the positions and increase
                # the drawdown tolerance level a little for the next trade
                if Portfolio._relative_drawdown[0] >= _drawdown_tolerance:
                    weight = pd.Series(0, Portfolio.components, name=time)
                    Portfolio.rebalance('to_weights', weight)
                    _drawdown_tolerance /= (1-_drawdown_tolerance)
                    continue # end action today
            
            # ----------------------------------------
            #  keep position if neither rebalance nor control drawdown
            # ----------------------------------------
            Portfolio.keep_position()
            
        # reset Trigger to eliminate the modification by the backtest    
        self.trigger.reset_scheduler()
        
        return Portfolio

    @classmethod
    def BuyAndHold(cls):
        """
        Initialize the Backtestor to conduct buy and hold method. It is
        supposed to make benchmark portfolio. If there are multiple components,
        the Backtestor will assigned equal weight to them and then buy and
        hold.
        """
        return cls('BuyAndHold', None, None, None, None, None)
    def _buy_and_hold(self, Portfolio, price_df):
        # asure no NaN for certain stocks at some dates
        price_df = self._process_price(price_df)
        # backtest starts from the start time of the portfolio
        index_time = price_df[Portfolio.inception:].index
        for time in index_time:
            price_now = price_df.loc[time]
            Portfolio.receive_price(price_now)
            if time == index_time[0]:
                weight = Allocator.EW()(price_now)
                Portfolio.rebalance('to_weights', weight)
            else:
                Portfolio.keep_position()
        return Portfolio
    
    def _process_price(self, price_df):
        price_df = copy.deepcopy(price_df)
        # need to format the single price series passed as pd.Series
        if isinstance(price_df, pd.Series):
            price_df = price_df.to_frame()
        price_df = price_df.fillna(method='ffill')
        return price_df
    
    def _alert_stop_loss(self):
        if (self.drawdown_tolerance < 1 and not self.controller):
        
            print(''.join(
                ['WARNING: a Controller shall be employed to prevent ',
                'entering improper positions again right after stopping loss.'])
                )