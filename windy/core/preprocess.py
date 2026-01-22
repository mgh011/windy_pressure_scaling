#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 09:07:43 2026

@author: mauro_ghirardelli
"""

# windy/core/preprocess.py

import warnings
import numpy as np
import xarray as xr
from scipy import signal


def detrend(ds, window):
    """
    Remove a linear trend within each resampled time block and return fluctuations.

    Requirements
    ------------
    - `ds` must have a `time` dimension.
    - Works with any other dimensions (e.g., height, tower) untouched.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    window : str
        Resampling window (e.g., "10min", "15min").

    Returns
    -------
    xarray.Dataset
        Dataset of detrended (fluctuation) time series for each block.
    """
    groups = ds.resample(time=window)
    out = []

    for _, g in groups:
        # apply scipy.signal.detrend along time axis for every variable
        g_dt = g.map(signal.detrend, args=[0])
        out.append(g_dt)

    return xr.concat(out, dim="time")


def get_fluctuations(ds, config):
    """
    Compute fluctuations for each field in the dataset.

    Fluctuations are defined as:
      - 'block'   : x - <x>_block  (block mean mapped back to original time)
      - 'detrend' : linear detrend performed independently in each block (returns fluctuations)

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with a `time` dimension.
    config : dict
        Must contain:
          - 'avg_method' : 'block' or 'detrend'
          - 'window'     : e.g. "10min"

    Returns
    -------
    xarray.Dataset
        Fluctuation dataset with same variables/dims as input.
    """
    method = config.get("avg_method", "detrend")
    window = config["window"]

    if method == "block":
        # block mean at window resolution, then broadcast back to original time
        ds_mean = ds.resample(time=window).mean()
        ds_mean = ds_mean.reindex(time=ds.time).ffill(dim="time")
        return ds - ds_mean

    if method == "detrend":
        return detrend(ds, window)

    warnings.warn(f"avg_method '{method}' not recognized, using 'detrend'.")
    return detrend(ds, window)
