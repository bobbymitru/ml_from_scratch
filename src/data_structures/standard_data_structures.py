"""
Standard bars generation logic
"""

# Imports
import numpy as np
import pandas as pd

class StandardBars:
    """
    Contains all of the logic to construct the standard bars. This class shouldn't be used directly.
    Use functions instead
    """

    def get_time_bars(df_ticks, resolution='H', num_units=1):
        """
        Creates Time Bars: date_time, open, high, low, close, volume, vwap
        
        :param df_ticks: (pd.DataFrame) Pandas Data Frame containing raw tick data in the format[date_time, price, volume]
        :param resolution: (str) Resolution type ('min' or 'T' -> minute, 'H' -> hour, 'D' -> day, 'B' -> business day, 'W' -> week, 'M' -> month end, 'Q' -> quarter end, 'Y' -> year end)
        :param num_units: (int) Number of resolution units (3 days for example, 2 hours)
        :return: (pd.DataFrame) Dataframe of time bars
        """
        
        # Copy the data frame in order not to change the source object
        df = df_ticks.copy()

        # Convert timestamp column to datetime
        df['date_time'] = pd.to_datetime(df['timestamp'])

        # Set timestamp as index
        df = df.set_index('date_time')

        # Resample the data based on the specified resolution and num_units
        df_grouped = df.resample(f'{num_units}{resolution}').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })

        # Rename columns
        df_grouped.columns = ['open', 'high', 'low', 'close', 'volume']

        # Calculate average price
        df_grouped['average_price'] = (df_grouped['high'] + df_grouped['low'] + df_grouped['close']) / 3

        # Calculate cumulative price * volume and cumulative volume
        df_grouped['cumulative_price_volume'] = (df_grouped['average_price'] * df_grouped['volume']).cumsum()
        df_grouped['cumulative_volume'] = df_grouped['volume'].cumsum()

        # Calculate VWAP
        df_grouped['vwap'] = df_grouped['cumulative_price_volume'] / df_grouped['cumulative_volume']

        # Reset index to get 'date_time' column
        df_grouped = df_grouped.reset_index()
        
        # Select columns
        df_grouped = df_grouped.loc[:, ['date_time', 'open', 'high', 'low', 'close', 'volume', 'vwap']]

        return df_grouped
    
    def get_tick_bars(df_ticks, threshold=10):
        """
        Creates Tick Bars: date_time, open, high, low, close, volume, vwap
        
        :param df_ticks: (pd.DataFrame) Pandas Data Frame containing raw tick data in the format[date_time, price, volume]
        :param threshold: (float) A cumulative value above this threshold triggers a sample to be taken.
        :return: (pd.DataFrame) Dataframe of tick bars
        """
        
        # Copy the data frame in order not to change the source object
        df = df_ticks.copy()

        # Sort values by date column in acsending order
        df = df.sort_values(by='timestamp')
        
        # Convert timestamp column to datetime
        df['date_time'] = pd.to_datetime(df['timestamp'])
        
        # Initialize variables for grouping
        grouped_rows = []
        current_group = []
        total_volume = 0
        
        # Iterate over rows
        for index, row in df.iterrows():
            current_group.append(row)
            total_volume += row['volume']
            
            # Check if total volume in the current group exceeds threshold
            if len(current_group) == threshold:
                # Aggregate data within the group
                group_data = {
                    'date_time': current_group[0]['date_time'],
                    'open': current_group[0]['price'],
                    'high': max([row['price'] for row in current_group]),
                    'low': min([row['price'] for row in current_group]),
                    'close': current_group[-1]['price'],
                    'volume': total_volume
                }
                grouped_rows.append(group_data)
                
                # Reset variables for the next group
                current_group = []
                total_volume = 0
        
        # If there are remaining rows after iteration, create a final group
        if current_group:
            group_data = {
                'date_time': current_group[0]['timestamp'],
                'open': current_group[0]['price'],
                'high': max([row['price'] for row in current_group]),
                'low': min([row['price'] for row in current_group]),
                'close': current_group[-1]['price'],
                'volume': total_volume
            }
            grouped_rows.append(group_data)
            
        # Create DataFrame from grouped data
        df_grouped = pd.DataFrame(grouped_rows)
        
        # Calculate average price
        df_grouped['average_price'] = (df_grouped['high'] + df_grouped['low'] + df_grouped['close']) / 3

        # Calculate cumulative price * volume and cumulative volume
        df_grouped['cumulative_price_volume'] = (df_grouped['average_price'] * df_grouped['volume']).cumsum()
        df_grouped['cumulative_volume'] = df_grouped['volume'].cumsum()

        # Calculate VWAP
        df_grouped['vwap'] = df_grouped['cumulative_price_volume'] / df_grouped['cumulative_volume']
    
        # Select columns
        df_grouped = df_grouped.loc[:, ['date_time', 'open', 'high', 'low', 'close', 'volume', 'vwap']]

        return df_grouped
    
    def get_volume_bars(df_ticks, threshold=10):
        """
        Creates Volume Bars: date_time, open, high, low, close, volume, vwap
        
        Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al, it is suggested that using 1/50 of the average daily volume, would result in more desirable statistical properties.
        
        :param df_ticks: (pd.DataFrame) Pandas Data Frame containing raw tick data in the format[date_time, price, volume]
        :param threshold: (float) A cumulative value above this threshold triggers a sample to be taken.
        :return: (pd.DataFrame) Dataframe of volume bars
        """
        # Copy the data frame in order not to change the source object
        df = df_ticks.copy()
        
        # Sort values by date column in acsending order
        df = df.sort_values(by='timestamp')
        
        # Convert timestamp column to datetime
        df['date_time'] = pd.to_datetime(df['timestamp'])
        
        # Initialize variables for grouping
        grouped_rows = []
        current_group = []
        total_volume = 0
        
        # Iterate over rows
        for index, row in df.iterrows():
            current_group.append(row)
            total_volume += row['volume']
            
            # Check if total volume in the current group exceeds threshold
            if total_volume >= threshold:
                # Aggregate data within the group
                group_data = {
                    'date_time': current_group[0]['date_time'],
                    'open': current_group[0]['price'],
                    'high': max([row['price'] for row in current_group]),
                    'low': min([row['price'] for row in current_group]),
                    'close': current_group[-1]['price'],
                    'volume': total_volume
                }
                grouped_rows.append(group_data)
                
                # Reset variables for the next group
                current_group = []
                total_volume = 0
        
        # If there are remaining rows after iteration, create a final group
        if current_group:
            group_data = {
                'date_time': current_group[0]['timestamp'],
                'open': current_group[0]['price'],
                'high': max([row['price'] for row in current_group]),
                'low': min([row['price'] for row in current_group]),
                'close': current_group[-1]['price'],
                'volume': total_volume
            }
            grouped_rows.append(group_data)
            
        # Create DataFrame from grouped data
        df_grouped = pd.DataFrame(grouped_rows)
        
        # Calculate average price
        df_grouped['average_price'] = (df_grouped['high'] + df_grouped['low'] + df_grouped['close']) / 3

        # Calculate cumulative price * volume and cumulative volume
        df_grouped['cumulative_price_volume'] = (df_grouped['average_price'] * df_grouped['volume']).cumsum()
        df_grouped['cumulative_volume'] = df_grouped['volume'].cumsum()

        # Calculate VWAP
        df_grouped['vwap'] = df_grouped['cumulative_price_volume'] / df_grouped['cumulative_volume']
    
        # Select columns
        df_grouped = df_grouped.loc[:, ['date_time', 'open', 'high', 'low', 'close', 'volume', 'vwap']]

        return df_grouped
    
    def get_dollar_bars(df_ticks, threshold=1000):
        """
        Creates Dollar Bars: date_time, open, high, low, close, volume, vwap
        
        :return: (pd.DataFrame) Dataframe of dollar bars
        """
        
        # Copy the data frame in order not to change the source object
        df = df_ticks.copy()

        # Sort values by date column in acsending order
        df = df.sort_values(by='timestamp')
        
        # Convert timestamp column to datetime
        df['date_time'] = pd.to_datetime(df['timestamp'])
        
        # Initialize variables for grouping
        grouped_rows = []
        current_group = []
        total_volume = 0
        total_dollar = 0
        
        # Iterate over rows
        for index, row in df.iterrows():
            current_group.append(row)
            total_volume += row['volume']
            total_dollar += row['volume'] * row['price']
            
            # Check if total volume in the current group exceeds threshold
            if total_dollar >= threshold:
                # Aggregate data within the group
                group_data = {
                    'date_time': current_group[0]['date_time'],
                    'open': current_group[0]['price'],
                    'high': max([row['price'] for row in current_group]),
                    'low': min([row['price'] for row in current_group]),
                    'close': current_group[-1]['price'],
                    'volume': total_volume
                }
                grouped_rows.append(group_data)
                
                # Reset variables for the next group
                current_group = []
                total_volume = 0
                total_dollar = 0
        
        # If there are remaining rows after iteration, create a final group
        if current_group:
            group_data = {
                'date_time': current_group[0]['timestamp'],
                'open': current_group[0]['price'],
                'high': max([row['price'] for row in current_group]),
                'low': min([row['price'] for row in current_group]),
                'close': current_group[-1]['price'],
                'volume': total_volume
            }
            grouped_rows.append(group_data)
            
        # Create DataFrame from grouped data
        df_grouped = pd.DataFrame(grouped_rows)
        
        # Calculate average price
        df_grouped['average_price'] = (df_grouped['high'] + df_grouped['low'] + df_grouped['close']) / 3

        # Calculate cumulative price * volume and cumulative volume
        df_grouped['cumulative_price_volume'] = (df_grouped['average_price'] * df_grouped['volume']).cumsum()
        df_grouped['cumulative_volume'] = df_grouped['volume'].cumsum()

        # Calculate VWAP
        df_grouped['vwap'] = df_grouped['cumulative_price_volume'] / df_grouped['cumulative_volume']
    
        # Select columns
        df_grouped = df_grouped.loc[:, ['date_time', 'open', 'high', 'low', 'close', 'volume', 'vwap']]

        return df_grouped