import numpy as np
import pandas as pd
import copy
import empyrical as ep
from scipy import stats


SIMPLE_METRIC_FUNCS = {
'Total Return': ep.cum_returns_final,
# 'Annual Reutrn': ep.annual_return,
# 'Annual Volatility': ep.annual_volatility,
'Max Drawdown': ep.max_drawdown,
'VaR': ep.value_at_risk,
'CVaR': ep.conditional_value_at_risk,
# 'Sharpe Ratio': ep.sharpe_ratio,
# 'Sortino Ratio': ep.sortino_ratio,
# 'Calmar Ratio': ep.calmar_ratio,
# 'Omega Ratio': ep.omega_ratio,
'Skew': stats.skew,
'Kurtosis': stats.kurtosis,
}

# SIMPLE_METRIC_FUNCS_BRIEF = {
#     'Total Return': ep.cum_returns_final,
#     'Annual Reutrn': ep.annual_return,
#     'Annual Volatility': ep.annual_volatility,
#     'Max Drawdown': ep.max_drawdown,
#     'Sharpe Ratio': ep.sharpe_ratio,
# }

SIMPLE_METRIC_FUNCS_BRIEF = {
    'Total Return': ep.cum_returns_final,
#     'Annual Reutrn': ep.annual_return,
#     'Annual Volatility': ep.annual_volatility,
    'Max Drawdown': ep.max_drawdown,
#     'Sharpe Ratio': ep.sharpe_ratio,
}

ANNUAL_METRIC_FUNCS = {
    # 'Annual Return': ep.annual_return,
    'CAGR': ep.cagr,
    'Annual Volatility': ep.annual_volatility,
    'Sharpe Ratio': ep.sharpe_ratio,
    'Sortino Ratio': ep.sortino_ratio,
    'Calmar Ratio': ep.calmar_ratio,
    'Omega Ratio': ep.omega_ratio,
}

ANNUAL_METRIC_FUNCS_BRIEF = {
    # 'Annual Return': ep.annual_return,
    'CAGR': ep.cagr,
    'Annual Volatility': ep.annual_volatility,
    'Sharpe Ratio': ep.sharpe_ratio,
}

FACTOR_METRIC_FUNCS = {
'Alpha': ep.alpha,
'Beta': ep.beta,
# 'Excess Sharpe': ep.excess_sharpe,
'Capture Ratio': ep.capture,
'Up Capture': ep.up_capture,
'Down Capture': ep.down_capture
}

FACTOR_METRIC_FUNCS_BRIEF = {
'Alpha': ep.alpha,
'Beta': ep.beta,
}

def summarize_returns(returns, benchmark_returns=None,
                      period='daily', name=None, brief=True):
    """
    Generate performance metrics of the portfolio including:
    [Total Return, Annual Reutrn, Annual Volatility, Max Drawdown, VaR,
    CVaR, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Omega Ratio, Skew,
    Kurtosis].
    If benchmark_return is asigned, it will generate extra metrics,
    including:
    [Alpha, Beta, Excess Sharpe, Capture Ratio, Up Capture, Down Captie]
    Parameters:
    benchmark_return: Series or DataFrame
        index: datetime
        name / coloumns: name of the asset
        values: returns
    """

    sheet_ben = None
    if benchmark_returns is not None:
        # align index
        idx = returns.index
        benchmark_returns = benchmark_returns.reindex(idx).fillna(0)
        benchmark_returns = benchmark_returns.loc[idx[0]:idx[-1]]
        sheet_ben = _summarize_returns_single(benchmark_returns,
                                    benchmark_returns,
                                    period,
                                    benchmark_returns.name,
                                    brief)
        sheet_ben['Alpha', 'Beta'] = np.nan

    if isinstance(returns, pd.Series):
        sheet = _summarize_returns_single(returns, benchmark_returns, period, name, 
                                          brief)
    elif isinstance(returns, pd.DataFrame):
        sheet = _summarize_returns_multiple(returns, benchmark_returns, period, brief)
        
  
    sheet_all = pd.concat([sheet, sheet_ben], 1)
    
    return sheet_all
        

def _summarize_returns_single(returns,
                              benchmark_returns,
                              period,
                              name,
                              brief):
    assert isinstance(returns, pd.Series)
    sheet = {
        'Begin': returns.index[0].strftime('%Y-%m-%d'),
        'End': returns.index[-1].strftime('%Y-%m-%d'),
    }
    
    annual_metric_funcs = (ANNUAL_METRIC_FUNCS_BRIEF if brief
                           else ANNUAL_METRIC_FUNCS)
    simple_metric_funcs = (SIMPLE_METRIC_FUNCS_BRIEF if brief
                            else SIMPLE_METRIC_FUNCS)
    factor_metric_funcs = (FACTOR_METRIC_FUNCS_BRIEF if brief
                            else FACTOR_METRIC_FUNCS)

    returns = returns.dropna()
    
    annual_metrics = {k: f(returns, period=period)
                      for k, f in annual_metric_funcs.items()}
    simple_metrics = {k: f(returns)
                      for k,f in simple_metric_funcs.items()}
    
    sheet.update(annual_metrics)
    sheet.update(simple_metrics)

    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.reindex(returns.index).fillna(0)
        extra_metrics = {k: f(returns, benchmark_returns)
                             for k,f in factor_metric_funcs.items()}
        sheet.update(extra_metrics)

    return pd.Series(sheet, name=name)

def _summarize_returns_multiple(returns_df, benchmark_returns, period, brief):
    assert isinstance(returns_df, pd.DataFrame)
    sheet = pd.DataFrame({
        name: _summarize_returns_single(returns, benchmark_returns, period,
                                        name, brief)
        for name, returns in returns_df.items()
    })
    return sheet


def shift_inception(inception, shift, capital, price_df, backtestor):
    """
    Test the sensitivity of strategy to the entry date. It first creats a list
    of consecutive entry dates, conducts separated backtests that starts on
    each entry date, and return the backtested Portfolios in a dictionary.
    
    Parameters:
    inception: str '%Y-%m-%d'
        the inception date of the backtest. For example, '2013-06-06'
    shift: int
        number of consecutive business days as entry date for backtest.
        For example, if date='2013-06-06' and shift=5, then the function will
        backtest 5 portfolios with entry date '2013-06-06', '2013-06-07',
        '2013-06-10', '2013-06-10', and '2013-06-12'.
    capital: float
        initial investment to the portfolio
    price_df: pandas.DataFrame
        the dataframe of historical prices of stocks.
        the columns should match the Portfolio.component_stocks.
        the range of time index should cover the Portfolio.date_start.
    backtestor: Backtestor Object
        initialized Backtestor
    """
    from .portfolio import Portfolio
    date_list = price_df.loc[inception:].head(shift).index.strftime('%Y-%m-%d')
    port_dict = {}
    for d in date_list:
        port = Portfolio(capital, d, price_df.columns, d)
        port = backtestor(port, price_df)
        port_dict[d] = port
    
    return port_dict


def shift_window(window_grid, inception, capital, price_df, backtestor):
    """
    Test the strategy with different time windows. It receives a list of 
    rebalance windows and lookback windows, conducts separated backtests based 
    on each window, and return the backtested Portfolios in a dictionary.
    
    Parameters:
    window_grid: list of tuple
        [(rebalance window 0, lookback window 0),
         (rebalance window 1, lookback window 1) ... ]
    inception: str '%Y-%m-%d'
        the inception date of the backtest. For example, '2013-06-06'
    capital: float
        initial investment to the portfolio
    price_df: pandas.DataFrame
        the dataframe of historical price_dfs of stocks.
        the columns should match the Portfolio.component_stocks.
        the range of time index should cover the Portfolio.inception.
    backtestor: Backtestor Object
        initialized Backtestor
    """
    from .portfolio import Portfolio
    from .trigger import Timer
    from .backtestor import Backtestor
    port_dict = {}
    backtestor = copy.deepcopy(backtestor)
    for w in window_grid:
        timer = Timer(w[0])
        backtestor.trigger = timer
        backtestor.window_lookback = w[1]
        

        port = Portfolio(capital, inception, price_df.columns, w)
        port = backtestor(port, price_df)
        port_dict[w] = port
        
    return port_dict


class GridSearch():
    """
    Used for performing portfolio optimization to search for suitable papameters.
    
    Initialization:
    ---------------
    price_df: pandas.DataFrame
    backtestor: Backtestor object
    inception: '%Y-%m-%d'
        start date of portfolio
    capital: float
        initial value of each portfoli
    name: str
        name of portfolio        
    """
    def __init__(self, price_df, backtestor, inception, capital=1000000,
                 name='strategy'):
        self.price_df = price_df
        self.backtestor = backtestor
        self.inception = inception
        self.capital = capital
        
    def search_window(self, window_grid, criterion='Sharpe Ratio'):
        self.window_grid = window_grid
        self.portfolio_dict = shift_window(window_grid, self.inception,
                                           self.capital, self.price_df,
                                           self.backtestor)
        self.summary = pd.concat([p.summary()
                             for p in self.portfolio_dict.values()], 1)
        self.best_window = self.summary.loc[:,criterion].idxmax()
        
    def get_portfolio(self):
        port = self.portfolio_dict[self.best_window]
        return port


class Frontier():
    def __init__(self, mu, sig):
        """
        A class to calculate effective frontier.
        """
        self.mu = mu
        self.sig = sig
        self._cal_inverse_sig()
        self._cal_matrices()
        
    @classmethod
    def ByReturn(cls, r_df, period=252):
        mu = r_df.mean() * period
        sig = r_df.cov() * period
        return cls(mu, sig)
    
    def __call__(self, mu):
        return self.cal_std(mu)

    def _cal_inverse_matrix(self, df):
        df_inv = pd.DataFrame(np.linalg.pinv(df.values),
                              df.columns, df.index)
        return df_inv

    def _cal_inverse_sig(self):
        self.sig_inv = self._cal_inverse_matrix(self.sig)
        
    def _cal_matrices(self):
        mu, sig_inv = self.mu, self.sig_inv
        
        e = pd.Series(1, index=mu.index)
        w_gm = (sig_inv@e) / (e@sig_inv@e)
        mu_gm = (mu@sig_inv@e) / (e@sig_inv@e)

        var_gm = 1 / (e@sig_inv@e)
        std_gm = var_gm ** 0.5

        psi = (e@sig_inv@e) * (mu@sig_inv@mu) - (e@sig_inv@mu)**2
        
        self.w_gm = w_gm
        self.mu_gm = mu_gm
        self.var_gm = var_gm
        self.std_gm = std_gm
        self.psi = psi
    
    def cal_std(self, mu):
        var_gm, mu_gm, psi = self.var_gm, self.mu_gm, self.psi
        var = var_gm + (mu - mu_gm)**2 / (psi*var_gm)
        std = var ** 0.5
        return std

    def get_gm(self):
        """
        Get global minimum std portfolio
        """
        return {'w_gm': self.w_gm, 'mu_gm':self.mu_gm,
                'std_gm':self.std_gm}