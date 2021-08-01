import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Allocator:
    """
    Allocator is implemented to decide the optimal investment proportion
    on different securities. Use classmethod to initialize the Allocator.
    At the initialization, it will choose and stick to one allocation method.
    When the Allocator object is called, it will
    excute the allocation method and return the calculated weight.
    Parameters:
    -----------
    method: str
        available methods in self._functor_dict
    **kwarg:
        parameters required by certain method
    """
    def __init__(self, method, **kwargs):

        self.method = method
        self.kwargs = kwargs
        self.functor_dict = {
            'EW': self._allocate_EW,
        }
        self._set_functor()

    def _set_functor(self):
        self._functor = self.functor_dict[self.method]

    def __call__(self, price_data):
        return self._functor(price_data)

    @classmethod
    def EW(cls, loading=1):
        """
        Initialize the Allocator to allocate equal weights.
        """
        return cls('EW', loading=loading)

    def _allocate_EW(self, price_df):
        """
        Return equal weight
        """
        loading = self.kwargs['loading']
        if isinstance(price_df, pd.DataFrame):
            price_df = price_df.iloc[-1]
        weight = price_df.notna() / price_df.notna().sum()
        weight = weight.fillna(0)
        weight = weight * loading
        return weight
