"""
Imbalance bars generation logic
"""

# Imports
import numpy as np
import pandas as pd

from src.util.extra_algorithms import ExtraAlgorithms as ea

class ImbalanceBars:
    """
    Contains all of the logic to construct the imbalance bars. This class shouldn't be used directly.
    Use functions instead
    """
    
    def get_tick_imbalance_bars(df_ticks, num_prev_bars=3, expected_num_ticks_init=10):
        """
        Creates Tick Imbalance Bars: date_time, open, high, low, close, volume

        :param df_ticks: (pd.DataFrame) Pandas Data Frame containing raw tick data in the format [timestamp, price, volume]
        :param num_prev_bars: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation)
        :param expected_num_ticks_init: (int) Initial expected number of ticks per bar
        :return: (pd.DataFrame) Dataframe of time bars
        """

        # Copy the data frame in order not to change the source object
        df = df_ticks.copy()

        # Convert timestamp column to datetime
        df['date_time'] = pd.to_datetime(df['timestamp'])

        # Initialize variables for grouping
        prev_price = df['price'][0]
        prev_tick_rule = 0
        cum_tick_imbalance = 0
        num_ticks = 0
        
        expected_imbalance = None
        expected_num_ticks = expected_num_ticks_init

        current_group = []
        grouped_rows = []
        imbalance_array = []
        bar_lenght_array = []

        # Iterate over rows
        for index, row in df.iterrows():
            num_ticks += 1
            
            # ========== TICK RULE ==========
            # Calculate the price difference
            price_delta = row['price'] - prev_price

            # Calculate the tick rule
            if price_delta == 0:
                curr_tick_rule = prev_tick_rule
                prev_tick_rule = curr_tick_rule
            else:
                curr_tick_rule = abs(price_delta) / price_delta
                prev_tick_rule = curr_tick_rule

            # Calculate tick imbalance and cumulative tick imbalance
            tick_imbalance = curr_tick_rule
            imbalance_array.append(tick_imbalance)
            cum_tick_imbalance += tick_imbalance

            current_group.append(row)

            # ========== EXPECTED IMBALANCE FOR THE FIRST BAR ==========
            if (expected_imbalance is None) and (len(imbalance_array) == expected_num_ticks_init):
                expected_imbalance = ea.ewma(np.array(imbalance_array), window=expected_num_ticks_init)[-1]

            # ========== IMBALANCE BAR ==========
            if (expected_imbalance is not None) and (abs(cum_tick_imbalance) >= (expected_num_ticks * abs(expected_imbalance))):
                # Aggregate data within the group
                group_data = {
                    'date_time': current_group[0]['date_time'],
                    'open': current_group[0]['price'],
                    'high': max([row['price'] for row in current_group]),
                    'low': min([row['price'] for row in current_group]),
                    'close': current_group[-1]['price'],
                    'volume': sum([row['volume'] for row in current_group])
                }
                grouped_rows.append(group_data)
                bar_lenght_array.append(num_ticks)

                # ========== EXPECTED NUMBER OF TICKS ==========
                expected_num_ticks = ea.ewma(np.array(bar_lenght_array), window=num_prev_bars)[-1]
                
                # ========== EXPECTED IMBALANCE ==========
                expected_imbalance = ea.ewma(np.array(imbalance_array), window = num_prev_bars * expected_num_ticks)[-1]

                # Reset variables for the next group
                cum_tick_imbalance = 0
                num_ticks = 0
                current_group = []

        # Create DataFrame from grouped data
        df_grouped = pd.DataFrame(grouped_rows)

        return df_grouped
    
    def get_volume_imbalance_bars(df_ticks, num_prev_bars=3, expected_num_ticks_init=10):
        """
        Creates Volume Imbalance Bars: date_time, open, high, low, close, volume
        
        :param df: (pd.DataFrame) Pandas Data Frame containing raw tick data in the format[date_time, price, volume]
        :param num_prev_bars: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation)
        :param expected_num_ticks_init: (int) Initial expected number of ticks per bar
        :return: (pd.DataFrame) Dataframe of volume bars
        """

        # Copy the data frame in order not to change the source object
        df = df_ticks.copy()

        # Convert timestamp column to datetime
        df['date_time'] = pd.to_datetime(df['timestamp'])

        # Initialize variables for grouping
        prev_price = df['price'][0]
        prev_tick_rule = 0
        cum_volume_imbalance = 0
        num_ticks = 0
        
        expected_imbalance = None
        expected_num_ticks = expected_num_ticks_init

        current_group = []
        grouped_rows = []
        imbalance_array = []
        bar_lenght_array = []

        # Iterate over rows
        for index, row in df.iterrows():
            num_ticks += 1
            
            # ========== TICK RULE ==========
            # Calculate the price difference
            price_delta = row['price'] - prev_price

            # Calculate the tick rule
            if price_delta == 0:
                curr_tick_rule = prev_tick_rule
                prev_tick_rule = curr_tick_rule
            else:
                curr_tick_rule = abs(price_delta) / price_delta
                prev_tick_rule = curr_tick_rule

            # Calculate volume imbalance and cumulative volume imbalance
            volume_imbalance = curr_tick_rule * row['volume']
            imbalance_array.append(volume_imbalance)
            cum_volume_imbalance += volume_imbalance

            current_group.append(row)

            # ========== EXPECTED IMBALANCE FOR THE FIRST BAR ==========
            if (expected_imbalance is None) and (len(imbalance_array) == expected_num_ticks_init):
                expected_imbalance = ea.ewma(np.array(imbalance_array), window=expected_num_ticks_init)[-1]

            # ========== IMBALANCE BAR ==========
            if (expected_imbalance is not None) and (abs(cum_volume_imbalance) >= (expected_num_ticks * abs(expected_imbalance))):
                # Aggregate data within the group
                group_data = {
                    'date_time': current_group[0]['date_time'],
                    'open': current_group[0]['price'],
                    'high': max([row['price'] for row in current_group]),
                    'low': min([row['price'] for row in current_group]),
                    'close': current_group[-1]['price'],
                    'volume': sum([row['volume'] for row in current_group])
                }
                grouped_rows.append(group_data)
                bar_lenght_array.append(num_ticks)

                # ========== EXPECTED NUMBER OF TICKS ==========
                expected_num_ticks = ea.ewma(np.array(bar_lenght_array), window=num_prev_bars)[-1]
                
                # ========== EXPECTED IMBALANCE ==========
                expected_imbalance = ea.ewma(np.array(imbalance_array), window = num_prev_bars * expected_num_ticks)[-1]

                # Reset variables for the next group
                cum_volume_imbalance = 0
                num_ticks = 0
                current_group = []

        # Create DataFrame from grouped data
        df_grouped = pd.DataFrame(grouped_rows)

        return df_grouped
    
    def get_dollar_imbalance_bars(df_ticks, num_prev_bars=3, expected_num_ticks_init=10):
        """
        Creates Dollar Imbalance Bars: date_time, open, high, low, close, volume
        
        :param df: (pd.DataFrame) Pandas Data Frame containing raw tick data in the format[date_time, price, volume]
        :param num_prev_bars: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation)
        :param expected_num_ticks_init: (int) Initial expected number of ticks per bar
        :return: (pd.DataFrame) Dataframe of dollar bars
        """

        # Copy the data frame in order not to change the source object
        df = df_ticks.copy()

        # Convert timestamp column to datetime
        df['date_time'] = pd.to_datetime(df['timestamp'])

        # Initialize variables for grouping
        prev_price = df['price'][0]
        prev_tick_rule = 0
        cum_dollar_imbalance = 0
        num_ticks = 0
        
        expected_imbalance = None
        expected_num_ticks = expected_num_ticks_init

        current_group = []
        grouped_rows = []
        imbalance_array = []
        bar_lenght_array = []

        # Iterate over rows
        for index, row in df.iterrows():
            num_ticks += 1
            
            # ========== TICK RULE ==========
            # Calculate the price difference
            price_delta = row['price'] - prev_price

            # Calculate the tick rule
            if price_delta == 0:
                curr_tick_rule = prev_tick_rule
                prev_tick_rule = curr_tick_rule
            else:
                curr_tick_rule = abs(price_delta) / price_delta
                prev_tick_rule = curr_tick_rule

            # Calculate dollar imbalance and cumulative dollar imbalance
            dollar_imbalance = curr_tick_rule * row['volume'] * row['price']
            imbalance_array.append(dollar_imbalance)
            cum_dollar_imbalance += dollar_imbalance

            current_group.append(row)

            # ========== EXPECTED IMBALANCE FOR THE FIRST BAR ==========
            if (expected_imbalance is None) and (len(imbalance_array) == expected_num_ticks_init):
                expected_imbalance = ea.ewma(np.array(imbalance_array), window=expected_num_ticks_init)[-1]

            # ========== IMBALANCE BAR ==========
            if (expected_imbalance is not None) and (abs(cum_dollar_imbalance) >= (expected_num_ticks * abs(expected_imbalance))):
                # Aggregate data within the group
                group_data = {
                    'date_time': current_group[0]['date_time'],
                    'open': current_group[0]['price'],
                    'high': max([row['price'] for row in current_group]),
                    'low': min([row['price'] for row in current_group]),
                    'close': current_group[-1]['price'],
                    'volume': sum([row['volume'] for row in current_group])
                }
                grouped_rows.append(group_data)
                bar_lenght_array.append(num_ticks)

                # ========== EXPECTED NUMBER OF TICKS ==========
                expected_num_ticks = ea.ewma(np.array(bar_lenght_array), window=num_prev_bars)[-1]
                
                # ========== EXPECTED IMBALANCE ==========
                expected_imbalance = ea.ewma(np.array(imbalance_array), window = num_prev_bars * expected_num_ticks)[-1]

                # Reset variables for the next group
                cum_dollar_imbalance = 0
                num_ticks = 0
                current_group = []

        # Create DataFrame from grouped data
        df_grouped = pd.DataFrame(grouped_rows)

        return df_grouped