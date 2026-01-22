#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCP / ISFS – build analysis-ready FAST (10 Hz) and SLOW (1 Hz) datasets
=====================================================================

Author
------
mauro_ghirardelli

Overview
--------
This script loads one SCP ISFS NetCDF file and produces two datasets:

1) FAST dataset (10 Hz):
   - dims: (tower, time_10hz, height)
   - variables:
       u, v, w, P   (height-resolved for tower M; 1 m only for A5/A10 and (if present) M)
       tc, diagbits (1 m only, where available)

2) SLOW dataset (1 Hz):
   - dims: (time, tower, height)
   - variables: RH, T
   - height union across towers (NaNs where a tower has no measurement)

Notes
-----
- The ISFS record time is centered within each second (e.g. ...:00.500).
  Burst samples are reconstructed using start = time - 0.5 s.
- 20 Hz -> 10 Hz is pure decimation (keep one sample every 2).
- 13 Hz -> 10 Hz is linear interpolation onto the 10 Hz sonic grid.
- M fast pressure P_5m has one incomplete second at the beginning; it is dropped.

Usage
-----
python scp_build_fast_slow.py <infile.nc> <outdir>

Outputs:
  <stem>_fast10.nc
  <stem>_slow1.nc
"""

from __future__ import annotations

import os
import re
import sys
import numpy as np
import xarray as xr

from scipy import signal


# =============================================================================
# Configuration
# =============================================================================
TOWERS_KEEP_EXACT = ("A5", "A10")
TOWERS_KEEP_PREFIX = ("M",)

KILL_PREFIXES = ("Ifan_", "U_", "V_", "kh2o", "kh2oV")
KILL_EXACT = {"Wetness", "altitude", "latitude", "base_time", "longitude"}


# =============================================================================
# Helpers
# =============================================================================
def clean_tower_coord(ds: xr.Dataset, dim_old: str = "station", dim_new: str = "tower") -> xr.Dataset:
    if dim_old in ds.dims or dim_old in ds.coords:
        ds = ds.rename({dim_old: dim_new})
    if dim_new in ds.coords:
        ds = ds.assign_coords({dim_new: ds[dim_new].astype(str)})
    return ds


def keep_selected_towers(ds: xr.Dataset, dim: str = "tower") -> xr.Dataset:
    towers = ds[dim].values.astype(str)
    keep = [t for t in towers if (t in TOWERS_KEEP_EXACT) or any(t.startswith(p) for p in TOWERS_KEEP_PREFIX)]
    return ds.sel({dim: keep})


def drop_unwanted_vars(ds: xr.Dataset) -> xr.Dataset:
    drop_vars: list[str] = []

    for v in ds.variables:  # includes coords too
        if v in ("tower", "time"):
            continue

        if v.endswith("_C"):
            drop_vars.append(v)
            continue

        m = re.search(r"_A(\d+)$", v)
        if m:
            a_num = int(m.group(1))
            if a_num not in (5, 10):
                drop_vars.append(v)

    for v in ds.variables:
        if v in ("tower", "time"):
            continue
        if v in KILL_EXACT or v.startswith(KILL_PREFIXES):
            drop_vars.append(v)

    drop_vars = sorted(set(drop_vars))
    return ds.drop_vars(drop_vars, errors="ignore")


def split_by_suffix(ds: xr.Dataset, suffix: str) -> tuple[xr.Dataset, xr.Dataset]:
    v_yes = [v for v in ds.data_vars if v.endswith(suffix)]
    v_no = [v for v in ds.data_vars if not v.endswith(suffix)]
    return ds[v_yes], ds[v_no]


def split_by_sampling_dims(ds: xr.Dataset, sample_dims=("sample", "sample_13")) -> tuple[xr.Dataset, xr.Dataset]:
    sample_dims = set(sample_dims)
    fast, slow = [], []
    for v in ds.data_vars:
        if set(ds[v].dims) & sample_dims:
            fast.append(v)
        else:
            slow.append(v)
    return ds[fast], ds[slow]


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


def decimate(ds: xr.Dataset, dim: str, factor: int) -> xr.Dataset:
    return ds.isel({dim: slice(None, None, factor)})


def interp_to_time10(ds_hf: xr.Dataset, time_dim_in: str, time10: xr.DataArray) -> xr.Dataset:
    tmp = ds_hf.interp({time_dim_in: time10})
    return (
        tmp.assign_coords(time_10hz=(time_dim_in, time10.data))
           .swap_dims({time_dim_in: "time_10hz"})
           .drop_vars(time_dim_in)
    )


def drop_and_strip_M(ds_M: xr.Dataset) -> xr.Dataset:
    ds_M = ds_M.drop_vars("P_20m_M", errors="ignore")
    rename_map = {v: v[:-2] for v in ds_M.data_vars if v.endswith("_M")}
    if len(set(rename_map.values())) != len(rename_map.values()):
        raise ValueError("Name collision detected when stripping _M suffix.")
    return ds_M.rename(rename_map)


def split_M_by_sample(ds_M: xr.Dataset, sample_dim: str = "sample") -> tuple[xr.Dataset, xr.Dataset]:
    fast, slow = [], []
    for v in ds_M.data_vars:
        if sample_dim in ds_M[v].dims:
            fast.append(v)
        else:
            slow.append(v)
    return ds_M[fast], ds_M[slow]


def parse_height_m(name: str) -> float:
    m = re.search(r"_(\d+(?:_\d+)?)m$", name)
    if m is None:
        raise ValueError(f"Cannot parse height from variable name: {name}")
    return float(m.group(1).replace("_", "."))


def stack_by_height(ds: xr.Dataset, prefix: str, out_name: str, time_dim: str) -> xr.DataArray:
    vars_ = [v for v in ds.data_vars if v.startswith(prefix + "_")]
    if not vars_:
        raise KeyError(f"No variables found for prefix '{prefix}_'")
    heights = [parse_height_m(v) for v in vars_]
    order = sorted(range(len(vars_)), key=lambda i: heights[i])
    vars_sorted = [vars_[i] for i in order]
    heights_sorted = [heights[i] for i in order]

    da = xr.concat([ds[v] for v in vars_sorted], dim="height").assign_coords(height=heights_sorted)
    da.name = out_name
    return da.transpose(time_dim, "height")


def build_slow_base(ds_other: xr.Dataset) -> xr.Dataset:
    needed = ["RH_0_5m", "RH_2m", "T_0_5m", "T_2m"]
    missing = [v for v in needed if v not in ds_other.data_vars]
    if missing:
        raise KeyError(f"Missing expected slow variables in ds_other: {missing}")

    RH = xr.concat([ds_other["RH_0_5m"], ds_other["RH_2m"]], dim="height").assign_coords(height=[0.5, 2.0])
    T  = xr.concat([ds_other["T_0_5m"],  ds_other["T_2m"]],  dim="height").assign_coords(height=[0.5, 2.0])

    return xr.Dataset(
        {"RH": RH, "T": T},
        coords={"time": ds_other.time, "tower": ds_other.tower, "height": RH.height},
        attrs=ds_other.attrs,
    ).transpose("time", "tower", "height")


def merge_slow(ds_slow_base: xr.Dataset, ds_M_slow_stacked: xr.Dataset, tower_M: str) -> xr.Dataset:
    ds_M = ds_M_slow_stacked.expand_dims({"tower": [tower_M]}).transpose("time", "tower", "height")
    h_all = np.unique(np.concatenate([ds_slow_base.height.values, ds_M.height.values]))

    a = ds_slow_base.reindex(height=h_all)
    b = ds_M.reindex(height=h_all)

    out = b.combine_first(a)
    return out


def make_fast_1m_height(ds_10hz: xr.Dataset) -> xr.Dataset:
    map_1m = {
        "u_1m": "u",
        "v_1m": "v",
        "w_1m": "w",
        "P_1m": "P",
        "tc_1m": "tc",
        "diagbits_1m": "diagbits",
    }
    keep = [k for k in map_1m if k in ds_10hz.data_vars]
    ds1 = ds_10hz[keep].rename(map_1m)
    return ds1.expand_dims({"height": [1.0]}).transpose("tower", "time_10hz", "height")


# =============================================================================
# FIR-based anti aliasing filtering
# =============================================================================


def _filtfilt_1d(x, b, axis=-1):
    """
    Robust 1D zero-phase FIR filtering along a given axis.

    This helper applies scipy.signal.filtfilt using FIR coefficients `b`,
    with conservative handling of missing data.

    Behavior with NaNs:
    -------------------
    - If NaNs are present anywhere in the input array, the function returns
      the original signal unchanged.
    - This avoids introducing artifacts due to filtering across data gaps.

    Padding strategy:
    -----------------
    - A safe padding length is computed based on the filter length and signal size.
    - If padding is not possible (too few samples), the signal is returned unchanged.

    Parameters
    ----------
    x : array-like
        Input signal.
    b : array-like
        FIR filter coefficients.
    axis : int, optional
        Axis along which filtering is applied (default: last axis).

    Returns
    -------
    array-like
        Zero-phase filtered signal, or the original signal if filtering
        cannot be safely applied.
    """
    if np.isnan(x).any():
        return x

    padlen = min(
        3 * (max(len(b), 1) - 1),
        x.shape[axis] - 1
    ) if x.shape[axis] > 1 else 0

    if padlen <= 0:
        return x

    return signal.filtfilt(b, [1.0], x, axis=axis, padlen=padlen)


def lowpass_fir_xr(ds, dim, fs, fc, numtaps=101):
    """
    Apply a zero-phase low-pass FIR filter to all variables in an xarray Dataset
    that depend on a given dimension.
    """
    if fc >= fs / 2:
        raise ValueError("fc must be smaller than the Nyquist frequency")

    b = signal.firwin(numtaps, fc, fs=fs)

    def _apply(da):
        axis = da.get_axis_num(dim)
        return xr.apply_ufunc(
            _filtfilt_1d,
            da,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            kwargs={"b": b, "axis": axis},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[da.dtype],
        )

    out_vars = {}
    for v in ds.data_vars:
        if dim in ds[v].dims:
            out_vars[v] = _apply(ds[v])
        else:
            out_vars[v] = ds[v]

    return xr.Dataset(out_vars, coords=ds.coords, attrs=ds.attrs)


def anti_alias_then_decimate(ds, dim, fs, factor, fc=None, numtaps=101):
    """
    Perform anti-alias filtering followed by decimation along a given dimension.

    A low-pass FIR filter is applied prior to downsampling in order to suppress
    spectral content above the new Nyquist frequency and prevent aliasing.
    """
    new_nyq = fs / (2 * factor)

    if fc is None:
        fc = 0.8 * new_nyq  # e.g. 20 → 10 Hz: new Nyquist = 5 Hz → fc = 4 Hz

    ds_f = lowpass_fir_xr(ds, dim=dim, fs=fs, fc=fc, numtaps=numtaps)
    return ds_f.isel({dim: slice(None, None, factor)})


# =============================================================================
# Build FAST and SLOW
# =============================================================================
def build_scp_fast_and_slow(infile: str) -> tuple[xr.Dataset, xr.Dataset]:
    ds = xr.open_dataset(infile)
    ds = clean_tower_coord(ds)
    ds = keep_selected_towers(ds)
    ds = drop_unwanted_vars(ds)

    # split M vars vs others
    ds_M_raw, ds_other = split_by_suffix(ds, "_M")

    # -------------------------
    # SLOW (1 Hz) RH/T
    # -------------------------
    ds_slow_base = build_slow_base(ds_other)

    # identify tower_M label from coord (e.g. 'M21')
    tower_M = [t for t in ds.tower.values.astype(str) if t.startswith("M")][0]

    ds_M = drop_and_strip_M(ds_M_raw)
    ds_M_fast_raw, ds_M_slow_raw = split_M_by_sample(ds_M, sample_dim="sample")

    RH_M = stack_by_height(ds_M_slow_raw, prefix="RH", out_name="RH", time_dim="time")
    T_M  = stack_by_height(ds_M_slow_raw, prefix="T",  out_name="T",  time_dim="time")

    ds_M_slow_stacked = xr.Dataset(
        {"RH": RH_M, "T": T_M},
        coords={"time": ds_M_slow_raw.time, "height": RH_M.height},
        attrs=ds_M_slow_raw.attrs,
    )

    ds_slow = merge_slow(ds_slow_base, ds_M_slow_stacked, tower_M=tower_M)

    # -------------------------
    # FAST base (A5/A10/(maybe M) at 1m) -> 10 Hz
    # -------------------------
    ds_other_fast, _ = split_by_sampling_dims(ds_other, sample_dims=("sample", "sample_13"))

    ds_fast_sample = ds_other_fast[
        [v for v in ds_other_fast.data_vars if "sample" in ds_other_fast[v].dims]
        ]
        
    ds_fast_sample13 = ds_other_fast[
            [v for v in ds_other_fast.data_vars if "sample_13" in ds_other_fast[v].dims]
        ]


    ds_fast_20hz = burst_to_continuous_time(ds_fast_sample, fs=20, sample_dim="sample", out_dim="time_20hz")
    ds_fast_13hz = burst_to_continuous_time(ds_fast_sample13, fs=13, sample_dim="sample_13", out_dim="time_13hz")

    #ds_sonic_10hz = decimate(ds_fast_20hz, dim="time_20hz", factor=2).rename({"time_20hz": "time_10hz"})
    ds_sonic_10hz = anti_alias_then_decimate(
    ds_fast_20hz, dim="time_20hz", fs=20.0, factor=2, fc=4.0, numtaps=121
    ).rename({"time_20hz": "time_10hz"})

    time10 = ds_sonic_10hz["time_10hz"]
    #ds_press_10hz = interp_to_time10(ds_fast_13hz, time_dim_in="time_13hz", time10=time10)
    ds_fast_13hz_f = lowpass_fir_xr(ds_fast_13hz, dim="time_13hz", fs=13.0, fc=4.0, numtaps=121)
    ds_press_10hz = interp_to_time10(ds_fast_13hz_f, time_dim_in="time_13hz", time10=time10)


    ds_10hz = xr.merge([ds_sonic_10hz, ds_press_10hz], compat="override")

    # convert to height=1.0 + clean var names
    ds_fast_1m = make_fast_1m_height(ds_10hz)

    # -------------------------
    # FAST M multi-height -> 10 Hz -> stack height
    # -------------------------
    # P_5m sometimes has an incomplete first-second burst; drop those seconds
    ds_M_fast_p = ds_M_fast_raw[["P_5m"]]
    ds_M_fast_uvwt = ds_M_fast_raw.drop_vars(["P_5m"], errors="ignore")

    n = ds_M_fast_raw.sizes["sample"]
    n_valid = ds_M_fast_p["P_5m"].notnull().sum("sample")
    good_time = n_valid == n

    ds_M_fast_p = ds_M_fast_p.sel(time=good_time)
    ds_M_fast_uvwt = ds_M_fast_uvwt.sel(time=good_time)

    ds_M_uvwt_20hz = burst_to_continuous_time(ds_M_fast_uvwt, fs=20, sample_dim="sample", out_dim="time_20hz_M")
    ds_M_P5_20hz = burst_to_continuous_time(ds_M_fast_p, fs=20, sample_dim="sample", out_dim="time_20hz_M")

    #ds_M_uvwt_10hz = decimate(ds_M_uvwt_20hz, dim="time_20hz_M", factor=2).rename({"time_20hz_M": "time_10hz"})
    #ds_M_P5_10hz = decimate(ds_M_P5_20hz, dim="time_20hz_M", factor=2).rename({"time_20hz_M": "time_10hz"})
    
    ds_M_uvwt_10hz = anti_alias_then_decimate(
    ds_M_uvwt_20hz, dim="time_20hz_M", fs=20.0, factor=2, fc=4.0, numtaps=121
    ).rename({"time_20hz_M": "time_10hz"})
    
    
    #
    #PROVA DEI 15Hz
    #
    
    p = ds_M_fast_raw["P_5m"]
    N = p.sizes["sample"]
    t = p["time"].values.astype("datetime64[ns]")
    
    dt = np.diff(t).astype("timedelta64[ns]").astype(np.int64) / 1e9  # seconds
    dt_med = np.nanmedian(dt)
    dt_p95 = np.nanpercentile(dt, 95)
    dt_p05 = np.nanpercentile(dt, 5)
    
    fs_eff = N / dt_med

    fsP = fs_eff 
    ds_M_P5_hf = burst_to_continuous_time(ds_M_fast_p, fs=fsP, sample_dim="sample", out_dim="time_P")
    
    ds_M_P5_f = lowpass_fir_xr(ds_M_P5_hf, dim="time_P", fs=fsP, fc=4.0, numtaps=121)
    
    ds_M_P5_10hz = interp_to_time10(ds_M_P5_f, time_dim_in="time_P", time10=time10)

    """
    ds_M_P5_10hz = anti_alias_then_decimate(
        ds_M_P5_20hz, dim="time_20hz_M", fs=15.0, factor=2, fc=4.0, numtaps=121
    ).rename({"time_20hz_M": "time_10hz"})
    """
    ds_M_fast_10hz = xr.merge([ds_M_uvwt_10hz, ds_M_P5_10hz], compat="override")
    
    
    uM = stack_by_height(ds_M_fast_10hz, prefix="u", out_name="u", time_dim="time_10hz")
    vM = stack_by_height(ds_M_fast_10hz, prefix="v", out_name="v", time_dim="time_10hz")
    wM = stack_by_height(ds_M_fast_10hz, prefix="w", out_name="w", time_dim="time_10hz")
    PM = stack_by_height(ds_M_fast_10hz, prefix="P", out_name="P", time_dim="time_10hz")

    ds_M_fast_clean = xr.Dataset(
        {"u": uM, "v": vM, "w": wM, "P": PM},
        coords={"time_10hz": ds_M_fast_10hz.time_10hz, "height": uM.height},
        attrs=ds_M_fast_10hz.attrs,
    ).expand_dims({"tower": [tower_M]}).transpose("tower", "time_10hz", "height")

    
    # -------------------------
    # Merge FAST (union heights)
    # -------------------------
    h_all = np.unique(np.concatenate([ds_fast_1m.height.values, ds_M_fast_clean.height.values]))
    a = ds_fast_1m.reindex(height=h_all)
    b = ds_M_fast_clean.reindex(height=h_all)

    ds_fast = b.combine_first(a)

    # -------------------------
    # Sanity checks
    # -------------------------
    assert set(ds_slow.data_vars) == {"RH", "T"}
    assert {"time", "tower", "height"}.issubset(ds_slow.dims)

    assert {"tower", "time_10hz", "height"}.issubset(ds_fast.dims)
    assert (ds_fast.time_10hz.values == ds_fast_1m.time_10hz.values).all()

    return ds_fast, ds_slow

"""
if __name__ == "__main__":
    infile = "/Users/mauro_ghirardelli/Documents/SCP/documents-export-2026-01-15/scp_tc_20121110_00.nc"
    outdir = "/Users/mauro_ghirardelli/Documents/SCP/hourly"

    ds_fast, ds_slow = build_scp_fast_and_slow(infile)

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(infile))[0]

    fast_path = os.path.join(outdir, f"{stem}_fast10.nc")
    slow_path = os.path.join(outdir, f"{stem}_slow1.nc")

    ds_fast.to_netcdf(fast_path)
    ds_slow.to_netcdf(slow_path)

    print("\nWrote:")
    print(" ", fast_path)
    print(" ", slow_path)
    print(ds_fast)
    print(ds_slow)
"""
# =============================================================================
# CLI
# =============================================================================
def main():
    infile = sys.argv[1]
    outdir = sys.argv[2]

    ds_fast, ds_slow = build_scp_fast_and_slow(infile)

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(infile))[0]

    fast_path = os.path.join(outdir, f"{stem}_hf10.nc")
    slow_path = os.path.join(outdir, f"{stem}_lf1.nc")

    ds_fast.to_netcdf(fast_path)
    ds_slow.to_netcdf(slow_path)

    print("\nWrote:")
    print(" ", fast_path)
    print(" ", slow_path)


if __name__ == "__main__":
    main()

