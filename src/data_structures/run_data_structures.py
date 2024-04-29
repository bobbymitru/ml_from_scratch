"""
Run bars generation logic
"""

# Imports
import numpy as np
import pandas as pd

from collections import namedtuple

from src.util.extra_algorithms import ExtraAlgorithms as ea

# class RunBars:
#     """
#     Contains all of the logic to construct the run bars. This class shouldn't be used directly.
#     Use functions instead
#     """
        
def _get_updated_counters(cache, flag, exp_num_ticks_init):
    """
    Updates the counters by resetting them or making use of the cache to update them based on a previous batch.

    :param cache: (string) Contains information from the previous batch that is relevant in this batch
    :param flag: (int) A flag which signals to use the cache
    :param exp_num_ticks: (int) Expected number of ticks per bar
    :return: Updated counters - cum_ticks, cum_dollar_value, cum_volume, high_price, low_price, exp_num_ticks, imbalance_array
    """

    # Check flag
    if flag and cache:
        # Update variables based on cache
        cum_ticks = int(cache[-1].cum_ticks)
        cum_dollar_value = np.float(cache[-1].cum_dollar_value)
        cum_volume = cache[-1].cum_volume
        low_price = np.float(cache[-1].low)
        high_price = np.float(cache[-1].high)
        # cumulative buy and sell imbalances for a particular run calculation (theta_t in Prado book)
        cum_theta_buy = np.float(cache[-1].cum_theta_buy)
        cum_theta_sell = np.float(cache[-1].cum_theta_sell)
        # expected number of ticks extracted from prev bars
        exp_num_ticks = np.float(cache[-1].exp_num_ticks)
        # array of latest imbalances
        imbalance_array = cache[-1].imbalance_array
    else:
        # Reset counters
        cum_ticks, cum_dollar_value, cum_volume, cum_theta_buy, cum_theta_sell = 0, 0, 0, 0, 0
        high_price, low_price = -np.inf, np.inf
        exp_num_ticks, imbalance_array = exp_num_ticks_init, {'buy': [], 'sell': []}  # in run bars we need to track both buy and sell imbalance

    return cum_ticks, cum_dollar_value, cum_volume, cum_theta_buy, cum_theta_sell, high_price, low_price, exp_num_ticks, imbalance_array


def _extract_bars(data, metric, exp_num_ticks_init=10, num_prev_bars=3, num_ticks_ewma_window=10,
                  cache=None, flag=False, num_ticks_bar=None):
    """
    For loop which compiles the various imbalance bars: dollar, volume, or tick.

    :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume
    :param metric: (string) dollar_run, volume_run or tick_run
    :param exp_num_ticks_init: (int) Initial guess of number of ticks in imbalance bar
    :param num_prev_bars: (int) Number of previous bars used for EWMA window (window=num_prev_bars * bar length) for estimating expected imbalance (tick, volume or dollar)
    :param num_ticks_ewma_window: (int) EWMA window to estimate expected number of ticks in a bar based on previous bars
    :param cache: (array) contains information from the previous batch that is relevant in this batch
    :param flag: (int) A flag which signals to use the cache
    :param num_ticks_bar: (int) Expected number of ticks per bar used to estimate the next bar
    :param prev_tick_rule: (int) Previous tick rule (if price_diff == 0 => use previous tick rule)
    :return: The financial data structure with the cache of short term history
    """

    cache_tup = namedtuple('CacheData', 
                           ['date_time', 'price', 'high', 'low', 'tick_rule', 'cum_volume', 'cum_dollar_value',
                            'cum_ticks', 'cum_theta_buy', 'cum_theta_sell', 'exp_num_ticks', 'imbalance_array'])

    if cache is None:
        cache = []
        prev_tick_rule = 0  # set the first tick rule with 0
        num_ticks_bar = []  # array of number of ticks from previous bars

    cum_ticks, cum_dollar_value, cum_volume, cum_theta_buy, cum_theta_sell, high_price, low_price, exp_num_ticks, imbalance_array = _get_updated_counters(cache, flag, exp_num_ticks_init)

    # Iterate over rows
    for row in data.values:
        # Set variables
        date_time = row[0]
        price = np.float(row[1])
        volume = row[2]

        # Calculations
        cum_ticks += 1
        dollar_value = price * volume
        cum_dollar_value = cum_dollar_value + dollar_value
        cum_volume = cum_volume + volume

        # Imbalance calculations
        try:
            tick_diff = price - cache[-1].price
            prev_tick_rule = cache[-1].tick_rule
        except IndexError:
            tick_diff = 0

        tick_rule = np.sign(tick_diff) if tick_diff != 0 else prev_tick_rule

        if metric == 'tick_run':
            imbalance = tick_rule
        elif metric == 'volume_run':
            imbalance = tick_rule * volume
        elif metric == 'dollar_run':
            imbalance = tick_rule * volume * price

        if imbalance > 0:
            imbalance_array['buy'].append(imbalance)
            # set zero to keep buy and sell arrays synced
            imbalance_array['sell'].append(0)
            cum_theta_buy += imbalance
        elif imbalance < 0:
            imbalance_array['sell'].append(abs(imbalance))
            imbalance_array['buy'].append(0)
            cum_theta_sell += abs(imbalance)

        if len(imbalance_array['buy']) < exp_num_ticks:
            # waiting for array to fill for ewma
            exp_buy_proportion, exp_sell_proportion = np.nan, np.nan
        else:
            # expected imbalance per tick
            ewma_window = int(exp_num_ticks * num_prev_bars)
            buy_sample = np.array(
                imbalance_array['buy'][-ewma_window:], dtype=float)
            sell_sample = np.array(
                imbalance_array['sell'][-ewma_window:], dtype=float)
            buy_and_sell_imb = sum(buy_sample) + sum(sell_sample)
            exp_buy_proportion = ea.ewma(
                buy_sample, window=ewma_window)[-1] / buy_and_sell_imb
            exp_sell_proportion = ea.ewma(
                sell_sample, window=ewma_window)[-1] / buy_and_sell_imb

        # Check min max
        if price > high_price:
            high_price = price
        if price <= low_price:
            low_price = price

        # Update cache
        cache_data = cache_tup(date_time, price, high_price, low_price, tick_rule, cum_volume, cum_dollar_value,
                               cum_ticks, cum_theta_buy, cum_theta_sell, exp_num_ticks, imbalance_array)
        cache.append(cache_data)

        # Check expression for possible bar generation
        if max(cum_theta_buy, cum_theta_sell) > exp_num_ticks * max(exp_buy_proportion, exp_sell_proportion): # pylint: disable=eval-used
            # Create bars
            open_price = cache[0].price
            low_price = min(low_price, open_price)
            close_price = price
            num_ticks_bar.append(cum_ticks)
            expected_num_ticks_bar = ea.ewma(
                np.array(num_ticks_bar[-num_ticks_ewma_window:], dtype=float), num_ticks_ewma_window)[-1] # expected number of ticks based on formed bars
            # Update bars & Reset counters
            list_bars.append([date_time, open_price, high_price, low_price, close_price,
                              cum_volume, cum_dollar_value, cum_ticks])
            cum_ticks, cum_dollar_value, cum_volume, cum_theta_buy, cum_theta_sell = 0, 0, 0, 0, 0
            high_price, low_price = -np.inf, np.inf
            exp_num_ticks = expected_num_ticks_bar
            cache = []

        # Update cache after bar generation (exp_num_ticks was changed after bar generation)
        cache_data = cache_tup(date_time, price, high_price, low_price, tick_rule, cum_volume, cum_dollar_value,
                               cum_ticks, cum_theta_buy, cum_theta_sell, exp_num_ticks, imbalance_array)
        cache.append(cache_data)

    return list_bars, cache, num_ticks_bar

def _assert_dataframe(test_batch):
    """
    Tests that the data frame read has the format: date_time, price, & volume.
    If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

    :param test_batch: (pd.DataFrame) DataFrame which will be tested
    """

    assert test_batch.shape[1] == 3, 'Must have only 3 columns in data frame: date_time, price, & volume.'
    assert isinstance(test_batch.loc[0, 'price'], float), 'price column in data frame not float.'
    assert isinstance(test_batch.loc[0, 'volume'], float), 'volume column in data frame not float.'

    try:
        pd.to_datetime(test_batch.iloc[0, 0])
    except ValueError:
        print('data frame, column 0, not a date time format:', test_batch.iloc[0, 0])

def _batch_run(df_ticks, metric, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=2e7):
    """
    Reads a data frame in batches and then constructs the financial data structure in the form of a DataFrame.
    The data frame must has only 3 columns: date_time, price, & volume.

    :param df_ticks: (pd.DataFrame) Pandas Data Frame containing raw tick data in the format [timestamp, price, volume]
    :param metric: (string) tick_run, dollar_run or volume_run
    :param exp_num_ticks_init: (int) Initial expetected number of ticks per bar
    :param num_prev_bars: (int) Number of previous bars used for EWMA window (window=num_prev_bars * bar length) for estimating expected imbalance (tick, volume or dollar)
    :num_ticks_ewma_window: (int) EWMA window for expected number of ticks calculations
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size
    :return: (pd.DataFrame) Financial data structure
    """

    print('Reading data in batches:')

    # Variables
    count = 0
    flag = False  # The first flag is false since the first batch doesn't use the cache
    cache = None
    num_ticks_bar = None
    final_bars = []

    # Read in the first row & assert format
    _assert_dataframe(df_ticks)

    for batch in ():
        print('Batch number:', count)
        list_bars, cache, num_ticks_bar = _extract_bars(
            data=batch, metric=metric, exp_num_ticks_init=exp_num_ticks_init, num_prev_bars=num_prev_bars,
            num_ticks_ewma_window=num_ticks_ewma_window, cache=cache, flag=flag, num_ticks_bar=num_ticks_bar)

        # Append to bars list
        final_bars += list_bars
        count += 1

        # Set flag to True: notify function to use cache
        flag = True

    # Return a DataFrame
    cols = ['date_time', 'open', 'high', 'low', 'close', 'cum_vol', 'cum_dollar', 'cum_ticks']
    df_bars = pd.DataFrame(final_bars, columns=cols)
    print('Returning bars \n')

    return df_bars

def get_tick_run_bars(df_ticks, exp_num_ticks_init=10, num_prev_bars=3, num_ticks_ewma_window=10, batch_size=2e7):
    """
    Creates the tick run bars: date_time, open, high, low, close, cum_ticks
    :param df_ticks: (pd.DataFrame) Pandas Data Frame containing raw tick data in the format [timestamp, price, volume]
    :param exp_num_ticks_init: (int) Initial expetected number of ticks per bar
    :param num_prev_bars: (int) Number of previous bars used for EWMA window (window=num_prev_bars * bar length) for estimating expected tick imbalance
    :param num_ticks_ewma_window: (int) EWMA window to estimate expected number of ticks in a bar based on previous bars
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size
    :return: (pd.DataFrame) Dataframe of tick bars
    """

    return _batch_run(df_ticks, metric='tick_run', exp_num_ticks_init=exp_num_ticks_init,
                      num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window, batch_size=batch_size)

def get_volume_run_bars(df_ticks, exp_num_ticks_init=10, num_prev_bars=3, num_ticks_ewma_window=10, batch_size=2e7):
    """
    Creates the volume run bars: date_time, open, high, low, close, cum_vol
    :param df_ticks: (pd.DataFrame) Pandas Data Frame containing raw tick data in the format [timestamp, price, volume]
    :param exp_num_ticks_init: (int) Initial expetected number of ticks per bar
    :param num_prev_bars: (int) Number of previous bars used for EWMA window (window=num_prev_bars * bar length) for estimating expected volume imbalance
    :param num_ticks_ewma_window: (int) EWMA window to estimate expected number of ticks in a bar based on previous bars
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size
    :return: (pd.DataFrame) Dataframe of volume bars
    """

    return _batch_run(df_ticks, metric='volume_run', exp_num_ticks_init=exp_num_ticks_init,
                      num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window, batch_size=batch_size)

def get_dollar_run_bars(df_ticks, exp_num_ticks_init=10, num_prev_bars=3, num_ticks_ewma_window=10, batch_size=2e7):
    """
    Creates the dollar run bars: date_time, open, high, low, close, cum_dollar
    :param df_ticks: (pd.DataFrame) Pandas Data Frame containing raw tick data in the format [timestamp, price, volume]
    :param exp_num_ticks_init: (int) Initial expetected number of ticks per bar
    :param num_prev_bars: (int) Number of previous bars used for EWMA window (window=num_prev_bars * bar length) for estimating expected dollar imbalance
    :param num_ticks_ewma_window: (int) EWMA window to estimate expected number of ticks in a bar based on previous bars
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size
    :return: (pd.DataFrame) Dataframe of dollar bars
    """

    return _batch_run(df_ticks, metric='dollar_run', exp_num_ticks_init=exp_num_ticks_init,
                      num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window, batch_size=batch_size)