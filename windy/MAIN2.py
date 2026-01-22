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
        spectra_smoothed = bin_spectra_log(spectra_welch, N_bin=80)
    
        # --- drop heavy variable explicitly (optional, helps memory)
        del spectra_welch


    # ============================================================
    # BLOCK 2c: Autocorrelation (optional)
    # ============================================================
    #autocorr = None
    #if cfg.get("products", {}).get("autocorr", False):
    #    autocorr, intlen = autocorrelation(ds, cfg, stats["meanU"])
    #    stats = xr.merge([stats, intlen])   # keep only integral scales in stats
    
    # ============================================================
    # BLOCK 3: Output
    # ============================================================
    return spectra_smoothed, stats#, autocorr


if __name__ == "__main__":
    CFG = "/Users/mauro_ghirardelli/Documents/windy_pressure_scaling/windy/conf/SCP_configuration.txt"
    FNAME = "scp_tc_20121110_18_hf10.nc"

    spectra_smoothed_SCP, stats = process_one_file(CFG, FNAME)
    
    print(stats)
#%%
if __name__ == "__main__":
    CFG = "/Users/mauro_ghirardelli/Documents/windy_pressure_scaling/windy/conf/TEAMx_configuration.txt"
    FNAME = "s1_hf.nc"

    spectra_smoothed_TEAMx, stats = process_one_file(CFG, FNAME)
    
    print(spectra_smoothed)
    #%%

#%%
if __name__ == "__main__":
    CFG = "/Users/mauro_ghirardelli/Documents/windy_pressure_scaling/windy/conf/M2HATS_configuration.txt"
    FNAME = "isfs_m2hats_qc_geo_tiltcor_hr_20230723_11_hf20.nc"

    spectra_smoothed_M2HATS, stats = process_one_file(CFG, FNAME)
    
    print(spectra_smoothed)
#%%
print(spectra_smoothed)
#%%

import numpy as np
import matplotlib.pyplot as plt

import math
import matplotlib.pyplot as plt


def plot_spectra_grid_towers(
    spectra,
    var="su",
    height=None,
    ncols=4,
    premultiply_f=True,
):
    """
    Grid of towers: each panel overlays all time windows for that tower.

    If premultiply_f=True, plots f * S(f) (or f * |C(f)| for cospectra).
    """
    if var not in spectra:
        raise ValueError(f"Variable '{var}' not in spectra.")

    if height is None:
        height = float(spectra.height.values[0])

    towers = list(spectra.tower.values)
    n = len(towers)
    nrows = math.ceil(n / ncols)

    f = spectra["freq"].values

    fig = plt.figure(figsize=(3.2 * ncols, 4 * nrows))
    for i, tow in enumerate(towers, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        da = spectra[var].sel(tower=tow, height=height)

        for t in da.time.values:
            y = da.sel(time=t).values

            # mask: positive freq, finite, positive spectra if autospectrum
            m = np.isfinite(f) & np.isfinite(y) & (f > 0)
            if var.startswith("s"):
                m &= (y > 0)

            if m.sum() < 5:
                continue

            yy = np.abs(y[m]) if var.startswith("c") else y[m]

            if premultiply_f:
                yy = f[m] * yy

            ax.loglog(f[m], yy, alpha=0.5)

        ax.set_title(str(tow))
        ax.grid(True, which="both", ls="--", lw=0.3)

        if i > (n - ncols):
            ax.set_xlabel("f [Hz]")
        if (i - 1) % ncols == 0:
            ylabel = f"f·{var}" if premultiply_f else var
            ax.set_ylabel(ylabel)

    title = f"{var} | height={height} m | all 30-min windows"
    if premultiply_f:
        title = "f · " + title

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    plt.show()

plot_spectra_grid_towers(
    spectra_smoothed_SCP,
    var="sp",
    height=1,
    premultiply_f=True
)
#%%

import xarray as xr
import numpy as np

FNAME = "/Users/mauro_ghirardelli/Documents/SCP/hourly/scp_tc_20121128_00_hf10.nc"
ds = xr.open_dataset(FNAME)

print(ds)                       # struttura generale
print("\nVars:", list(ds.data_vars))

# scegli cosa stampare
tow = "M21"
h = 5.0
var = "P"   # oppure "u","v","w","tc","diagbits" ecc.

da = ds[var].sel(tower=tow, height=h)

# se il tempo è time_10hz:
tname = "time_10hz" if "time_10hz" in da.dims else "time"
print("\nSelected:", var, "tower=", tow, "height=", h, "| dim time:", tname)

# stampa prime N e ultime N righe (tempo + valore)
N = 15
t = da[tname].values
y = da.values

print(f"\n--- first {N} samples ---")
for i in range(min(N, y.shape[0])):
    print(str(t[i]), float(y[i]) if np.isfinite(y[i]) else y[i])

print(f"\n--- last {N} samples ---")
for i in range(max(0, y.shape[0]-N), y.shape[0]):
    print(str(t[i]), float(y[i]) if np.isfinite(y[i]) else y[i])

#%%
import xarray as xr
import matplotlib.pyplot as plt

FNAME = "/Users/mauro_ghirardelli/Documents/SCP/hourly/scp_tc_20121128_00_hf10.nc"
ds = xr.open_dataset(FNAME)

# --- scelta ---
towers = ["A5", "A10", "M21"]
height = 1.0        # cambia in 5.0 se vuoi
var = "P"

plt.figure(figsize=(12,4))

for tow in towers:
    if tow not in ds.tower.values:
        print(f"Tower {tow} not in dataset, skipping")
        continue

    da = ds[var].sel(tower=tow, height=height)

    t = da["time_10hz"].values
    y = da.values

    plt.plot(t, y, lw=1, label=tow)

plt.xlabel("Time")
plt.ylabel(f"P [{ds[var].attrs.get('units','')}]")
plt.title(f"Pressure time series at {height} m")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
#%%
print(
    spectra_smoothed_SCP.sp
        .sel(tower="A10", height=5.0)
)

#%%
daP = ds_fast["P"].sel(tower="A10", height=5.0)
print(daP.notnull().any().item(), daP.notnull().mean().item())
#%%
import numpy as np

# 1) spettri A10 a 5m: sono tutti NaN?
s = spectra_smoothed_SCP["sp"].sel(tower="A10", height=5.0)
print("SPEC A10 5m all-NaN?", bool(s.isnull().all()))
print("SPEC A10 5m any finite?", bool(np.isfinite(s.values).any()))

# 2) se sono finiti, sono IDENTICI a M21 5m? (broadcast)
sA = spectra_smoothed_SCP["sp"].sel(tower="A10", height=5.0)
sM = spectra_smoothed_SCP["sp"].sel(tower="M21", height=5.0)

# confronto robusto
diff = (sA - sM).values
print("max|A10-M21|:", np.nanmax(np.abs(diff)))
