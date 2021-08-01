class Timer():
    """
    Timer is applied to set the rebalance frequency of Backtestor.BySchedule.
    It initialized a counter. Everytime when update_scheduler() is called, the
    Timer will add one tothe counter. When is_trade_time() is called, if the
    counter equals to the preset period, it will return True, and reset the
    counter to zero.
    Parameters:
    -----------
    period: int
        time interval to return True
    counter_init: int
        initial value of counter. Default as period - 1, so the is_trade_time()
        will immediately return True if update_scheduler() is call once.
    """
    def __init__(self, period, counter_init=None):
        self.period = period
        self.counter_init = counter_init
        self._init_counter()

    def _init_counter(self):
        if self.counter_init:
            self.counter = self.counter_init
        else:
            self.counter = self.period - 1

    def update_scheduler(self, *args):
        self._count_time()

    def is_trade_time(self):
        if self.counter == self.period:
            self._reset_counter()
            return True
        else:
            return False

    def _count_time(self):
        self.counter += 1

    def _reset_counter(self):
        self.counter = 0

    def reset_scheduler(self):
        self._init_counter()
        
class Scheduler():
    """
    Schedule the trading on certain dates.
    
    Parameters
    ----------
    schedule: list of time object
    """
    def __init__(self, schedule):
        self.schedule = schedule
        self.iter_schedule = iter(self.schedule)
        self.trade_time = next(self.iter_schedule)
        
    def update_scheduler(self, time):
        self.now = time
        
    def is_trade_time(self):
        if self.now >= self.trade_time:
            next_trade_time = next(self.iter_schedule, None)
            if next_trade_time is not None:
                self.trade_time = next_trade_time
#             print('Rebalance: ', self.now)
                return True

    def reset_scheduler(self):
        self.iter_schedule = iter(self.schedule)