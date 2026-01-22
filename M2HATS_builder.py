#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
M2HATS builder: HF (20 Hz) + LF (1 Hz)

Outputs
-------
HF dataset (20 Hz):
    dims: (time_hf_20, height, tower)
    includes:
      - horizontal array towers (t2..t47) at 4 m
      - vertical profile tower (t0) at all available heights (or subset if you choose)
      - 60 Hz variables are anti-alias filtered and decimated to 20 Hz

LF dataset (1 Hz):
    dims: (height, time, tower)
    includes:
      - horizontal array slow vars at 4 m
      - t0 slow vars at multiple heights
"""

import os
import re
from collections import defaultdict
import sys

import numpy as np
import xarray as xr
from scipy import signal


# ---------------------------------------------------------------------
# CONFIG DEFAULTS
# ---------------------------------------------------------------------
SITES_ARRAY_DEFAULT = [
    "t2", "t5", "t8", "t11", "t14", "t17", "t20",
    "t23", "t26", "t29", "t32", "t35", "t38", "t41", "t44", "t47"
]
SITE_PROFILE_DEFAULT = ["t0"]

COMMON_VARS = [
    "P", "Pirga", "RH", "T", "Tirga",
    "diagbits", "irgadiag", "ldiag",
    "tc", "u", "v", "w"
]

HEIGHT_TARGET_ARRAY = 4.0

PATTERN = re.compile(
    r"^(?P<var>[A-Za-z]+)_(?P<height>\d+(?:_\d+)?)m_t(?P<tower>\d+)$"
)


# =============================================================================
# Burst -> continuous time (EXACT VERSION REQUESTED)
# =============================================================================
def burst_to_continuous_time(
    ds: xr.Dataset,
    fs: float,
    sample_dim: str,
    time_dim: str = "time",
    out_dim: str = "time_hf",
) -> xr.Dataset:
    t_center = ds[time_dim].astype("datetime64[ns]")
    t_start = t_center - np.timedelta64(500, "ms")

    n = ds.sizes[sample_dim]
    k = xr.DataArray(np.arange(n, dtype=np.int64), dims=(sample_dim,))
    offsets_ns = np.rint(k.values * 1e9 / fs).astype(np.int64)
    offsets = xr.DataArray(offsets_ns.astype("timedelta64[ns]"), dims=(sample_dim,))

    time_2d = t_start + offsets
    out = ds.stack({out_dim: (time_dim, sample_dim)})
    out = out.assign_coords({out_dim: time_2d.stack({out_dim: (time_dim, sample_dim)})})
    return out


# =============================================================================
# FIR anti-aliasing + decimation (conservative NaN policy)
# =============================================================================
def _filtfilt_1d(x, b, axis=-1):
    if np.isnan(x).any():
        return x

    padlen = min(3 * (max(len(b), 1) - 1), x.shape[axis] - 1) if x.shape[axis] > 1 else 0
    if padlen <= 0:
        return x

    return signal.filtfilt(b, [1.0], x, axis=axis, padlen=padlen)


def lowpass_fir_xr(ds, dim, fs, fc, numtaps=101):
    if fc >= fs / 2:
        raise ValueError("fc must be smaller than the Nyquist frequency")

    b = signal.firwin(numtaps, fc, fs=fs)

    def _apply(da):
        return xr.apply_ufunc(
            _filtfilt_1d,
            da,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            kwargs={"b": b, "axis": -1},   # <-- QUESTO È IL FIX
            vectorize=True,
            dask="parallelized",
            output_dtypes=[da.dtype],
        )


    out_vars = {}
    for v in ds.data_vars:
        out_vars[v] = _apply(ds[v]) if dim in ds[v].dims else ds[v]

    return xr.Dataset(out_vars, coords=ds.coords, attrs=ds.attrs)


def anti_alias_then_decimate(ds, dim, fs, factor, fc=None, numtaps=101):
    new_nyq = fs / (2 * factor)
    if fc is None:
        fc = 0.8 * new_nyq

    ds_f = lowpass_fir_xr(ds, dim=dim, fs=fs, fc=fc, numtaps=numtaps)
    return ds_f.isel({dim: slice(None, None, factor)})


# =============================================================================
# Helpers
# =============================================================================
def vars_to_tower_dim(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert variables named VAR_tXX into variables VAR with a tower dimension.
    """
    groups = defaultdict(dict)
    for name, da in ds.data_vars.items():
        var, tower = name.split("_", 1)
        groups[var][tower] = da

    out = []
    for var, tower_map in groups.items():
        towers = sorted(tower_map.keys())
        da = xr.concat([tower_map[t] for t in towers], dim="tower")
        da = da.assign_coords(tower=towers)
        da.name = var
        out.append(da)

    return xr.merge(out)


def split_fast_slow_generic(ds: xr.Dataset):
    sample_dims = {"sample", "sample_20", "sample_30", "sample_50"}
    fast_vars, slow_vars = [], []
    for v, da in ds.data_vars.items():
        if any(d in da.dims for d in sample_dims):
            fast_vars.append(v)
        else:
            slow_vars.append(v)
    return ds[sorted(fast_vars)], ds[sorted(slow_vars)]


# =============================================================================
# Build array subset at 4 m
# =============================================================================
def build_horizontal_array_subset(ds0: xr.Dataset, sites_array) -> xr.Dataset:
    vars_keep, rename_map = [], {}

    for name, da in ds0.data_vars.items():
        m = PATTERN.match(name)
        if not m:
            continue
        var = m.group("var")
        height = float(m.group("height").replace("_", "."))
        tower = f"t{m.group('tower')}"

        if var not in COMMON_VARS:
            continue
        if tower not in sites_array:
            continue
        if height != HEIGHT_TARGET_ARRAY:
            continue

        vars_keep.append(name)
        rename_map[name] = f"{var}_{tower}"

    return ds0[vars_keep].rename(rename_map).expand_dims(height=[HEIGHT_TARGET_ARRAY])


def build_array_slow_and_fast(ds0: xr.Dataset, sites_array):
    ds_array = build_horizontal_array_subset(ds0, sites_array)
    ds_array_fast, ds_array_slow = split_fast_slow_generic(ds_array)

    # LF (slow)
    ds_slow_array = vars_to_tower_dim(ds_array_slow)

    # HF (fast): split by sample_20 vs sample
    vars_20 = [v for v, da in ds_array_fast.data_vars.items() if "sample_20" in da.dims]
    vars_60 = [v for v, da in ds_array_fast.data_vars.items() if "sample" in da.dims]

    ds_fast_20_burst = ds_array_fast[vars_20]
    ds_fast_60_burst = ds_array_fast[vars_60]

    # burst -> continuous
    ds_20_cont = burst_to_continuous_time(ds_fast_20_burst, fs=20.0, sample_dim="sample_20", out_dim="time_hf_20")
    ds_60_cont = burst_to_continuous_time(ds_fast_60_burst, fs=60.0, sample_dim="sample", out_dim="time_hf_60")

    # tower dimension
    ds_20 = vars_to_tower_dim(ds_20_cont)
    ds_60 = vars_to_tower_dim(ds_60_cont)

    # decimate 60 -> 20
    ds_60_to_20 = anti_alias_then_decimate(ds_60, dim="time_hf_60", fs=60.0, factor=3).rename(
        {"time_hf_60": "time_hf_20"}
    )

    ds_fast_array = xr.merge([ds_20, ds_60_to_20], join="inner")
    return ds_slow_array, ds_fast_array


# =============================================================================
# Build t0 profile (multi-height)
# =============================================================================
def build_t0_grouped_datasets(ds0: xr.Dataset):
    groups = defaultdict(dict)

    for name, da in ds0.data_vars.items():
        m = PATTERN.match(name)
        if not m:
            continue

        var = m.group("var")
        if var not in COMMON_VARS:
            continue

        tower = f"t{m.group('tower')}"
        if tower != "t0":
            continue

        h = float(m.group("height").replace("_", "."))
        dims = da.dims

        if dims == ("time",):
            kind = "slow"
        elif dims == ("time", "sample"):
            kind = "sample"
        elif dims == ("time", "sample_20"):
            kind = "sample_20"
        else:
            kind = "other"

        groups[(var, kind)][h] = da

    def _build(kind: str) -> xr.Dataset:
        out = {}
        for (var, k), hmap in groups.items():
            if k != kind:
                continue
            hs = sorted(hmap.keys())
            da_cat = xr.concat([hmap[h] for h in hs], dim="height").assign_coords(height=hs)
            da_cat.name = var
            out[var] = da_cat
        return xr.Dataset(out, attrs=ds0.attrs).sortby("height") if out else xr.Dataset()

    return _build("slow"), _build("sample_20"), _build("sample")


def build_t0_slow_and_fast(ds0: xr.Dataset):
    ds_t0_slow, ds_t0_fast20_burst, ds_t0_fast60_burst = build_t0_grouped_datasets(ds0)

    # LF: add tower dimension
    ds_slow_t0 = ds_t0_slow.expand_dims(tower=["t0"]) if len(ds_t0_slow.data_vars) else xr.Dataset()

    # HF: 20 Hz branch
    if len(ds_t0_fast20_burst.data_vars):
        ds_t0_20_cont = burst_to_continuous_time(ds_t0_fast20_burst, fs=20.0, sample_dim="sample_20", out_dim="time_hf_20")
        ds_t0_20_cont = ds_t0_20_cont.expand_dims(tower=["t0"])
    else:
        ds_t0_20_cont = xr.Dataset()

    # HF: 60 Hz branch -> decimate to 20 Hz
    if len(ds_t0_fast60_burst.data_vars):
        ds_t0_60_cont = burst_to_continuous_time(ds_t0_fast60_burst, fs=60.0, sample_dim="sample", out_dim="time_hf_60")
        ds_t0_60_cont = ds_t0_60_cont.expand_dims(tower=["t0"])

        ds_t0_60_to_20 = anti_alias_then_decimate(ds_t0_60_cont, dim="time_hf_60", fs=60.0, factor=3).rename(
            {"time_hf_60": "time_hf_20"}
        )
    else:
        ds_t0_60_to_20 = xr.Dataset()

    # merge t0 HF on 20 Hz axis (time intersection)
    ds_fast_t0 = xr.merge([ds_t0_20_cont, ds_t0_60_to_20], join="inner") if (
        len(ds_t0_20_cont.data_vars) or len(ds_t0_60_to_20.data_vars)
    ) else xr.Dataset()

    return ds_slow_t0, ds_fast_t0


# =============================================================================
# Public builder API
# =============================================================================
def build_hf_and_lf_datasets(
    file_path: str,
    sites_keep=None,
):
    """
    Build HF (20 Hz) and LF (1 Hz) datasets for M2HATS.

    Parameters
    ----------
    file_path : str
        Path to raw ISFS netCDF file.
    sites_keep : list[str] or None
        Horizontal array towers to keep (t2..t47). If None, uses defaults.

    Returns
    -------
    ds_hf : xr.Dataset
        20 Hz dataset with dims (time_hf_20, height, tower).
    ds_lf : xr.Dataset
        1 Hz dataset with dims (height, time, tower).
    """
    if sites_keep is None:
        sites_keep = SITES_ARRAY_DEFAULT
    sites_keep = list(sites_keep)

    ds0 = xr.open_dataset(file_path)

    # array products
    ds_slow_array, ds_fast_array = build_array_slow_and_fast(ds0, sites_keep)

    # t0 products
    ds_slow_t0, ds_fast_t0 = build_t0_slow_and_fast(ds0)

    # LF merge (union of heights/towers)
    ds_lf = xr.merge([ds_slow_array, ds_slow_t0], join="outer")

    # HF merge: MUST be outer on tower, otherwise tower intersection can be empty
    ds_hf = xr.merge([ds_fast_array, ds_fast_t0], join="outer")

    # Nice ordering
    if "tower" in ds_lf.dims:
        torder = ["t0"] + [t for t in ds_slow_array["tower"].values if t != "t0"]
        torder = [t for t in torder if t in ds_lf["tower"].values]
        ds_lf = ds_lf.sel(tower=torder)

    if "tower" in ds_hf.dims:
        torder = ["t0"] + [t for t in ds_fast_array["tower"].values if t != "t0"]
        torder = [t for t in torder if t in ds_hf["tower"].values]
        ds_hf = ds_hf.sel(tower=torder)

    return ds_hf, ds_lf


# =============================================================================
# Run
# =============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


DATASET_PATH = "/Users/mauro_ghirardelli/Documents/M2HATS/raw/"
OUTDIR = "/Users/mauro_ghirardelli/Documents/M2HATS/hourly/"

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    # prendi tutti i netcdf nella cartella (se vuoi solo hourly: aggiungi "_hr_" nel filtro)
    files = sorted(
        f for f in os.listdir(DATASET_PATH)
        if f.endswith(".nc")
        # and "_hr_" in f          # <- scommenta se vuoi SOLO file hourly
        # and f.startswith("isfs_m2hats")  # <- opzionale
    )

    for FILE_NAME in files:
        try:
            print("\n" + "=" * 60)
            print("PROCESSING:", FILE_NAME)
            print("=" * 60)
    
            file_path = os.path.join(DATASET_PATH, FILE_NAME)
    
            ds_hf, ds_lf = build_hf_and_lf_datasets(file_path, sites_keep=SITES_ARRAY_DEFAULT)
            ds_hf = ds_hf.rename({"time_hf_20": "time"})
    
            print("HF dataset (20 Hz):")
            print(ds_hf)
    
            print("\nLF dataset (1 Hz):")
            print(ds_lf)
        except Exception as e:
            print("\n" + "!" * 60)
            print("FAILED:", FILE_NAME)
            print("Reason:", repr(e))
            print("!" * 60 + "\n")
            continue

        # -------------------------------------------------------------
        # Save
        # -------------------------------------------------------------
        stem = os.path.splitext(FILE_NAME)[0]
        hf_out = os.path.join(OUTDIR, f"{stem}_hf20.nc")
        lf_out = os.path.join(OUTDIR, f"{stem}_lf1.nc")

        hf_encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_hf.data_vars}
        lf_encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_lf.data_vars}

        print("\nSaving HF...")
        ds_hf.to_netcdf(hf_out, encoding=hf_encoding)

        print("Saving LF...")
        ds_lf.to_netcdf(lf_out, encoding=lf_encoding)

        print("\nSaved:")
        print(hf_out)
        print(lf_out)


"""
DATASET_PATH = "/Users/mauro_ghirardelli/Documents/M2HATS/raw/"
FILE_NAME = "isfs_m2hats_qc_geo_tiltcor_hr_20230730_00.nc"

if __name__ == "__main__":
    file_path = os.path.join(DATASET_PATH, FILE_NAME)

    ds_hf, ds_lf = build_hf_and_lf_datasets(file_path, sites_keep=SITES_ARRAY_DEFAULT)
    ds_hf = ds_hf.rename({"time_hf_20": "time"})

    print("HF dataset (20 Hz):")
    print(ds_hf)

    print("\nLF dataset (1 Hz):")
    print(ds_lf)

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    OUTDIR = "/Users/mauro_ghirardelli/Documents/M2HATS/hourly/"
    os.makedirs(OUTDIR, exist_ok=True)

    hf_out = os.path.join(OUTDIR, "m2hats_20230730_00_hf20.nc")
    lf_out = os.path.join(OUTDIR, "m2hats_20230730_00_lf1.nc")

    # Simple and safe encoding
    hf_encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_hf.data_vars}
    lf_encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_lf.data_vars}

    print("\nSaving HF...")
    ds_hf.to_netcdf(hf_out, encoding=hf_encoding)

    print("Saving LF...")
    ds_lf.to_netcdf(lf_out, encoding=lf_encoding)

    print("\nSaved:")
    print(hf_out)
    print(lf_out)
    print(ds_hf)
"""