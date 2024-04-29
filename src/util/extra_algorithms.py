"""
Imbalance bars generation logic
"""

# Imports
import numpy as np
import pandas as pd

class ExtraAlgorithms():
    """
    Contains all of the logic for additional algorithms. This class shouldn't be used directly.
    Use functions instead
    """
    
    def ewma(arr_in, window):
        """
        Exponentialy weighted moving average specified by a decay 'window' to provide better adjustments for small windows via y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) / (1 + (1-a) + (1-a)^2 + ... + (1-a)^n)

        :param arr_in : (np.ndarray, float64) A single dimenisional numpy array
        :param window : (int) The decay window, or 'span'
        :return: (np.ndarray) The EWMA vector, same length / shape as 'arr_in'
        """
        arr_length = arr_in.shape[0]
        ewma_arr = np.empty(arr_length, dtype='float64')
        alpha = 2 / (window + 1)
        weight = 1
        ewma_old = arr_in[0]
        ewma_arr[0] = ewma_old
        for i in range(1, arr_length):
            weight += (1 - alpha)**i
            ewma_old = ewma_old * (1 - alpha) + arr_in[i]
            ewma_arr[i] = ewma_old / weight

        return ewma_arr
    
    def generate_cov_matrix(row):
        """
        Forms covariance matrix from current data frame row using 'rolling_cov', 'rolling_spx_var' and 'rolling_eur_var' column values
        """
        
        cov = row['rolling_cov']
        spx_var = row['rolling_spx_var']
        euro_var = row['rolling_euro_var']
        
        return np.matrix([[spx_var, cov], [cov, euro_var]])
    
    def pca_weights(cov, risk_dist=None, risk_target=1.):
        """
        Calculates hedging weights using covariance matrix (cov), risk distribution (risk_dist) and risk_target
        """
        
        # Following the risk_alloc distribution, match risk_target
        e_val, e_vec = np.linalg.eigh(cov) # must be Hermitian
        indices = e_val.argsort()[::-1] # arguments for sorting e_val desc
        e_val, e_vec = e_val[indices], e_vec[:, indices]
        if risk_dist is None:
            risk_dist = np.zeros(cov.shape[0])
            risk_dist[-1] = 1.
        loads = risk_target * (risk_dist / e_val) ** .5
        weights = np.dot(e_vec, np.reshape(loads, (-1, 1)))
        # ctr = (loads / risk_target) ** 2 * e_val # verify risk_dist
        
        return weights
    
    def get_daily_vol(close, span0=100):
        """
        Compute the daily volatility of price returns.

        :param close : (pd.Series) Close prices of the asset
        :param span0 : (int, optional) Lookback period for computing volatility.
        
        :return : (pd.Series) Daily volatility of price returns
        """
        
        # daily vo, reindexed to close
        df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
        df0 = df0[df0 > 0]
        df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1 # daily returns
        df0 = df0.ewm(span=span0).std()
        
#         # Calculate daily returns
#         daily_returns = price.pct_change()

#         # Compute rolling standard deviation with lookback period
#         volatility = daily_returns.rolling(window=lookback).std()

        return df0
    
    def get_t_events(g_raw, h):
        """
        Detects a shift in the mean value of measured quantity away from a target value
        
        :param g_raw : (int) The raw time series
        :param h : (int) The threshold
        :return : (pd.DatetimeIndex) The index of T events
        """
        
        t_events, s_pos, s_neg = [], 0, 0
        diff = g_raw.diff()
        for i in diff.index[1:]:
            s_pos, s_neg = max(0, s_pos + diff.loc[i]), min(0, s_neg + diff.loc[i])
            if s_neg < -h:
                s_neg = 0
                t_events.append(i)
            elif s_pos > h:
                s_pos = 0
                t_events.append(i)
        
        return pd.DatetimeIndex(t_events)
    
    def mp_pandas_obj(func, pd_obj, num_threads=24, mp_batches=1, lin_mols=True, **kargs):
        """
        Advances in Financial Machine Learning, Snippet 20.7, page 310.

        The mpPandasObj, used at various points in the book

        Parallelize jobs, return a dataframe or series.
        Example: df1=mp_pandas_obj(func,('molecule',df0.index),24,**kwds)

        First, atoms are grouped into molecules, using linParts (equal number of atoms per molecule)
        or nestedParts (atoms distributed in a lower-triangular structure). When mpBatches is greater
        than 1, there will be more molecules than cores. Suppose that we divide a task into 10 molecules,
        where molecule 1 takes twice as long as the rest. If we run this process in 10 cores, 9 of the
        cores will be idle half of the runtime, waiting for the first core to process molecule 1.
        Alternatively, we could set mpBatches=10 so as to divide that task in 100 molecules. In doing so,
        every core will receive equal workload, even though the first 10 molecules take as much time as the
        next 20 molecules. In this example, the run with mpBatches=10 will take half of the time consumed by
        mpBatches=1.

        Second, we form a list of jobs. A job is a dictionary containing all the information needed to process
        a molecule, that is, the callback function, its keyword arguments, and the subset of atoms that form
        the molecule.

        Third, we will process the jobs sequentially if numThreads==1 (see Snippet 20.8), and in parallel
        otherwise (see Section 20.5.2). The reason that we want the option to run jobs sequentially is for
        debugging purposes. It is not easy to catch a bug when programs are run in multiple processors.
        Once the code is debugged, we will want to use numThreads>1.

        Fourth, we stitch together the output from every molecule into a single list, series, or dataframe.

        :param func: (function) A callback function, which will be executed in parallel
        :param pd_obj: (tuple) Element 0: The name of the argument used to pass molecules to the callback function
                        Element 1: A list of indivisible tasks (atoms), which will be grouped into molecules
        :param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
        :param mp_batches: (int) Number of parallel batches (jobs per core)
        :param lin_mols: (bool) Tells if the method should use linear or nested partitioning
        :param kargs: (var args) Keyword arguments needed by func
        :return: (pd.DataFrame) of results
        """

        if lin_mols:
            parts = lin_parts(len(pd_obj[1]), num_threads * mp_batches)
        else:
            parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches)

        jobs = []
        for i in range(1, len(parts)):
            job = {pd_obj[0]: pd_obj[1][parts[i - 1]:parts[i]], 'func': func}
            job.update(kargs)
            jobs.append(job)

        if num_threads == 1:
            out = process_jobs_(jobs)
        else:
            out = process_jobs(jobs, num_threads=num_threads)

        if isinstance(out[0], pd.DataFrame):
            df0 = pd.DataFrame()
        elif isinstance(out[0], pd.Series):
            df0 = pd.Series()
        else:
            return out

        for i in out:
            df0 = df0.append(i)

        df0 = df0.sort_index()
        return df0