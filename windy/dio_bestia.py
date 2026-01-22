#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:21:57 2026

@author: mauro_ghirardelli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import xarray as xr
import sys

# local path (come fai tu)
sys.path.append('/Users/mauro_ghirardelli/Documents/windy_pressure_scaling/windy/core/')

from layout import check_dataset_structure
from gapfill import fill_gaps
from rotation import double_rotation
from stats import fluxes_calc
from stationarity import stationarity
from spectra import spectra_eps, bin_spectra_log
from microbarom import get_microbarom
from autocorr import autocorrelation


def load_config_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_single_hf_file(cfg, filename):
    hf_dir = cfg["paths"]["hf_dir"]
    full_path = os.path.join(hf_dir, filename)

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")

    # includi tc perché c’è sempre (e serve a stats/stationarity)
    wanted_vars = tuple(cfg.get("fields", ["P", "u", "v", "w", "tc"]))

    ds = xr.open_dataset(
        full_path,
        decode_times=True,
        chunks=cfg.get("chunks", None),
    )

    present = [v for v in wanted_vars if v in ds.data_vars]
    ds = ds[present]

    if "heights" in ds.dims and "height" not in ds.dims:
        ds = ds.rename({"heights": "height"})
    # normalize time dim
    if "time_10hz" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"time_10hz": "time"})
    if "time_20hz" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"time_20hz": "time"})
    if "time_60hz" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"time_60hz": "time"})


    check_dataset_structure(ds)
    return ds


def process_one_file(cfg_path, filename):
    """
    Process a single high-frequency NetCDF file and compute statistical
    and spectral products.

    The function is designed to operate on ONE file only.
    File iteration / job arrays (e.g. SLURM) are intentionally handled elsewhere.

    Processing is organized in clearly separated BLOCKS, following the
    original M2HATS post-processing philosophy.

    Parameters
    ----------
    cfg_path : str
        Path to the JSON configuration file.
    filename : str
        Name of the HF NetCDF file to process (located inside cfg["paths"]["hf_dir"]).

    Returns
    -------
    xr.Dataset
        Dataset containing block-averaged statistics and, if enabled,
        spectral-derived quantities (epsilon, slopes).
    """
    
    results = {}

    # ============================================================
    # BLOCK -1: Configuration & data loading
    # ============================================================
    cfg = load_config_json(cfg_path)
    ds = load_single_hf_file(cfg, filename)

    # ============================================================
    # BLOCK 0: Data preparation
    # ============================================================
    ds, qc = fill_gaps(ds, cfg, count_nans=True)
    ds, rot = double_rotation(ds, cfg, return_rotation=True)

    # ============================================================
    # BLOCK 1: Turbulence statistics
    # ============================================================
    flux = fluxes_calc(ds, cfg)
    stat = stationarity(ds, cfg)

    stats = xr.merge([flux, stat, rot, qc])

    if "meanU" in stats:
        stats = stats.where(stats["meanU"] > 0)

    # ============================================================
    # BLOCK 2: Spectral analysis (optional)
    # ============================================================
    spectra_smoothed = None
    
    if cfg.get("products", {}).get("spectra", False):
    
        # --- 2a) compute Welch spectra (HEAVY, temporary)
        spectra_welch, epsilon, slopes = spectra_eps(
            ds, cfg,
            Umean=stats["meanU"],
            welch_segments=3,
            welch_overlap=0.5,
        )
        stats = xr.merge([stats, epsilon, slopes])
    
        # ============================================================
        # BLOCK 2b: Microbarom diagnostics (optional) [computed on WELCH]
        # ============================================================
        if cfg.get("products", {}).get("microbarom", False):
            mb = get_microbarom(spectra_welch, stats)
            stats = stats.assign(
                MB_fit=mb["area_fit"],
                MB_peak=mb["area_peak"],
                MB_peak_abs=mb["area_peak_abs"],
            )
    
        # --- 2c) final spectra product to SAVE (LIGHT): log-binned 80 bins
        # choose ONE binning routine and stick to it
        spectra_smoothed = bin_spectra_log(spectra_welch, N_bin=100)
    
        # --- drop heavy variable explicitly (optional, helps memory)
        del spectra_welch


    # ============================================================
    # BLOCK 2c: Autocorrelation (optional)
    # ============================================================
    autocorr = None
    if cfg.get("products", {}).get("autocorr", False):
        autocorr, intlen = autocorrelation(ds, cfg, stats["meanU"])
        stats = xr.merge([stats, intlen])   # keep only integral scales in stats
    
    # ============================================================
    # BLOCK 3: Output
    # ============================================================
    return spectra_smoothed, stats, autocorr
#%%
import numpy as np
def print_spectral_diagnostics(spectra, label):
    print(f"\n=== SPECTRAL DIAGNOSTICS: {label} ===")

    if spectra is None:
        print("No spectra returned.")
        return

    freq = spectra["freq"]

    print("freq min [Hz]:", float(freq.min().values))
    print("freq max [Hz]:", float(freq.max().values))
    print("n_freq bins  :", freq.size)

    # controllo monotonia
    df = np.diff(freq.values)
    print("freq monotonic increasing:", np.all(df > 0))

    # stampa qualche valore di controllo
    print("first 5 freq:", freq.values[:5])
    print("last  5 freq:", freq.values[-5:])

    # guarda una variabile spettrale se c'è
    for v in ["sp", "su", "sw"]:
        if v in spectra:
            x = spectra[v].isel(freq=slice(-5, None)).values
            print(f"{v} last bins (sample):", x.reshape(-1)[:5])
            break
#%%
if __name__ == "__main__":
    CFG = "/Users/mauro_ghirardelli/Documents/windy_pressure_scaling/windy/conf/M2HATS_configuration.txt"
    FNAME = "isfs_m2hats_qc_geo_tiltcor_hr_20230731_13_hf20.nc"

    spectra_smoothed, stats, autocorr = process_one_file(CFG, FNAME)
    print(stats)
    print_spectral_diagnostics(spectra_smoothed, "M2HATS hf20")

#%%
if __name__ == "__main__":
    CFG = "/Users/mauro_ghirardelli/Documents/windy_pressure_scaling/windy/conf/SCP_configuration.txt"
    FNAME = "scp_tc_20121129_02_hf10.nc"

    spectra_smoothed, stats, autocorr = process_one_file(CFG, FNAME)
    print(stats)
    print_spectral_diagnostics(spectra_smoothed, "SCP hf10")
#%%
import xarray as xr
import numpy as np

def inspect_hf_dataset(path, label):
    print("\n" + "="*60)
    print(f"INSPECT DATASET: {label}")
    print("="*60)

    ds = xr.open_dataset(path, decode_times=True)

    print("\n--- DATASET SUMMARY ---")
    print(ds)

    # ---- time diagnostics
    if "time" in ds.coords:
        t = ds["time"]
        dt = t.diff("time")
        dt_s = (dt / np.timedelta64(1, "s")).values

        print("\n--- TIME ---")
        print("time start:", t.values[0])
        print("time end  :", t.values[-1])
        print("dt [s] median:", np.nanmedian(dt_s))
        print("dt [s] min/max:", np.nanmin(dt_s), np.nanmax(dt_s))
        print("n time steps:", t.size)

    # ---- variable diagnostics (focus on P)
    if "P" in ds.data_vars:
        P = ds["P"]

        print("\n--- PRESSURE P ---")
        print("dims:", P.dims)
        print("dtype:", P.dtype)

        Pvals = P.values

        print("P min:", np.nanmin(Pvals))
        print("P max:", np.nanmax(Pvals))
        print("P mean:", np.nanmean(Pvals))
        print("P std :", np.nanstd(Pvals))

        print("n NaN :", np.isnan(Pvals).sum())
        print("n zero:", np.sum(Pvals == 0))

        # controllo varianza temporale (importantissimo)
        if "time" in P.dims:
            P_std_time = P.std("time", skipna=True)
            print("min std over time:", float(P_std_time.min().values))
            print("max std over time:", float(P_std_time.max().values))

    else:
        print("\n--- NO PRESSURE VARIABLE 'P' FOUND ---")

    return ds


DS_M2 = inspect_hf_dataset(
    "/Users/mauro_ghirardelli/Documents/M2HATS/hourly/"
    "isfs_m2hats_qc_geo_tiltcor_hr_20230731_13_hf20.nc",
    label="M2HATS hf20"
)

#%%
DS_SCP = inspect_hf_dataset(
    "/Users/mauro_ghirardelli/Documents/SCP/hourly/"
    "scp_tc_20121129_02_hf10.nc",
    label="SCP hf10"
)
