import pandas as pd
import numpy as np

class Selector:
    """
    Selector is used to select securities for portfolio. Use classmethod to 
    initialize it, which will take in certain paremeters according to the
    specific method. After initialization, the Selector takes in the price
    dataframe, and set the prices of unchosen securities as np.nan

    parameters:
    -----------
    price_data: pandas.DataFrame
          
    """
    def __init__(self, method, **kwargs):

        self.method = method
        self.kwargs = kwargs
        self._set_functor()
    
    def _set_functor(self):
        self._functor_dict = {'Score': self._select_security_score,
                             }
        self._functor = self._functor_dict[self.method]
    
    def __call__(self, price_data):
        return self._functor(price_data)
    
    
    @classmethod
    def Score(cls, score_df, short_score=0, long_score=9):
        return cls('Score', score_df=score_df, short_score=short_score, long_score=long_score)
    
    def _select_security_score(self, price_df):
        score_df = self.kwargs['score_df']
        short_score = self.kwargs['short_score']
        long_score = self.kwargs['long_score']
        date = price_df.index[-1]
        score = score_df.loc[date, :]
        selected = (score <= short_score) + (score >= long_score)
        return selected