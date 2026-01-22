#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 10:06:16 2026

@author: mauro_ghirardelli
"""

# windy/core/spectra.py

import numpy as np
import xarray as xr
from scipy import signal
import sys

sys.path.append('/Users/mauro_ghirardelli/Documents/windy_pressure_scaling/windy/core/')
from preprocess import get_fluctuations


def _infer_dt_seconds(da_time: xr.DataArray):
    dt = da_time.diff("time")
    dt_s = (dt / np.timedelta64(1, "s")).median(skipna=True)
    return float(dt_s.values)



def _resolve_welch_params(n, segments=None, overlap=0.0):
    if not segments:
        return n, 0
    ov = float(overlap)
    ov = max(0.0, min(0.95, ov))
    denom = (1.0 + (segments - 1) * (1.0 - ov))
    nperseg = max(8, int(np.floor(n / denom)))
    noverlap = int(np.floor(ov * nperseg))
    return nperseg, noverlap


def _welch_2d(x2d, fs, *, segments=None, overlap=0.0, window="hann", detrend="constant", scaling="density"):
    """
    x2d: shape (n_series, n_time)
    returns freq (nf,), psd (n_series, nf) without f=0
    """
    n = x2d.shape[-1]
    nperseg, noverlap = _resolve_welch_params(n, segments=segments, overlap=overlap)
    f, Pxx = signal.welch(
        x2d, fs=fs, axis=-1,
        window=window, detrend=detrend,
        nperseg=nperseg, noverlap=noverlap,
        scaling=scaling
    )
    return f[1:], np.real(Pxx[..., 1:])


def _csd_2d(x2d, y2d, fs, *, segments=None, overlap=0.0, window="hann", detrend="constant", scaling="density"):
    """
    x2d,y2d: shape (n_series, n_time)
    returns freq (nf,), Cxy (n_series, nf) without f=0
    """
    n = x2d.shape[-1]
    nperseg, noverlap = _resolve_welch_params(n, segments=segments, overlap=overlap)
    f, Pxy = signal.csd(
        x2d, y2d, fs=fs, axis=-1,
        window=window, detrend=detrend,
        nperseg=nperseg, noverlap=noverlap,
        scaling=scaling
    )
    return f[1:], Pxy[..., 1:]


def spectra_welch(ds_fluct, window, *, segments=3, overlap=0.5, window_type="hann", detrend="constant"):
    """
    Welch autospectra + selected cospectra, block-by-block.

    Returns xr.Dataset with dims: time, tower, height, freq
    (order may be time,tower,height,freq).
    """
    # normalize time dim name (we assume caller already renamed time_10hz -> time)
    if "time" not in ds_fluct.dims:
        raise ValueError("spectra_welch expects dimension 'time'.")

    # required vars
    for v in ("u", "v", "w", "tc"):
        if v not in ds_fluct.data_vars:
            raise ValueError(f"spectra_welch requires '{v}' in ds_fluct.")
    has_p = "P" in ds_fluct.data_vars

    dt = _infer_dt_seconds(ds_fluct["time"])
    fs = 1.0 / dt

    groups = ds_fluct.resample(time=window)
    out_blocks = []

    for label, g in groups:
        # bring to (tower, height, time) for fast reshape
        # if a dim is missing (unlikely), this will error loudly (good)
        g3 = g.transpose("tower", "height", "time")

        u = np.asarray(g3["u"].values)
        v = np.asarray(g3["v"].values)
        w = np.asarray(g3["w"].values)
        tc = np.asarray(g3["tc"].values)
        P = np.asarray(g3["P"].values) if has_p else None

        # reshape to (n_series, n_time)
        nt = u.shape[-1]
        n_series = u.shape[0] * u.shape[1]
        u2 = u.reshape(n_series, nt)
        v2 = v.reshape(n_series, nt)
        w2 = w.reshape(n_series, nt)
        t2 = tc.reshape(n_series, nt)
        p2 = P.reshape(n_series, nt) if has_p else None

        # autospectra
        freq, su = _welch_2d(u2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
        _, sv = _welch_2d(v2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
        _, sw = _welch_2d(w2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
        _, sT = _welch_2d(t2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)

        # cospectra (main) - complex cross spectra
        _, cuw = _csd_2d(u2, w2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
        _, cvw = _csd_2d(v2, w2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
        _, cuv = _csd_2d(u2, v2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
        _, cwT = _csd_2d(w2, t2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)

        data = {
            # autospectra
            "su": su, "sv": sv, "sw": sw, "sT": sT,

            # keep REAL cospectra as before (backward compatible)
            "cuw": cuw.real, "cvw": cvw.real, "cuv": cuv.real, "cwT": cwT.real,

            # NEW: imaginary (quadrature) parts
            "cuw_im": cuw.imag, "cvw_im": cvw.imag, "cuv_im": cuv.imag, "cwT_im": cwT.imag,
        }

        if has_p:
            _, sp = _welch_2d(p2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
            _, cwp = _csd_2d(w2, p2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)

            data["sp"] = sp

            # keep REAL cospectrum as before
            data["cwp"] = cwp.real

            # NEW: imaginary (quadrature) part
            data["cwp_im"] = cwp.imag


        # back to (tower,height,freq)
        ntow = g3.sizes["tower"]
        nh = g3.sizes["height"]
        nf = len(freq)

        ds_block = xr.Dataset(
            coords=dict(
                time=label,
                tower=g3["tower"].values,
                height=g3["height"].values,
                freq=freq,
            )
        )

        for name, arr in data.items():
            ds_block[name] = (("tower", "height", "freq"), arr.reshape(ntow, nh, nf))

        out_blocks.append(ds_block)

    return xr.concat(out_blocks, dim="time")


def spectral_slopes_epsilon(spectra, Umean):
    """
    Compute epsilon + slopes from autospectra su,sv,sw,sT using a cutoff f_c = U/(2π z).

    spectra: Dataset with dims (time,tower,height,freq) and vars su,sv,sw,sT
    Umean: DataArray with dims (time,tower,height) (or broadcastable)
    """
    if "height" not in spectra.coords:
        raise ValueError("spectral_slopes_epsilon expects coord 'height'.")
    z = spectra["height"]

    # cutoff broadcast: (time,tower,height)
    cutoff = Umean / (2.0 * np.pi * z)

    S = spectra[["su", "sv", "sw", "sT"]]
    S = S.where(S > 0)   # elimina zeri e negativi


    # high freq range: f > cutoff AND avoid last bins
    fmax = spectra["freq"].isel(freq=-8)
    Sh = S.where((spectra.freq > cutoff) & (spectra.freq < fmax))

    # push left limit to first maximum (per component)
    # (keeps the original idea, but simplified: use su peak)
    f_peak = Sh["su"].idxmax(dim="freq")
    Sh = Sh.where(spectra.freq > f_peak)

    Sl = S.where(spectra.freq < cutoff)

    # constants
    cu = 18 / 55 * 1.5
    cvw = cu * 4 / 3
    cT = 0.8

    # epsilon (median over freq)
    epsU = (2*np.pi/Umean * (Sh.freq**(5/3) * Sh.su / cu)**(3/2)).median("freq").rename("epsU")
    epsV = (2*np.pi/Umean * (Sh.freq**(5/3) * Sh.sv / cvw)**(3/2)).median("freq").rename("epsV")
    epsW = (2*np.pi/Umean * (Sh.freq**(5/3) * Sh.sw / cvw)**(3/2)).median("freq").rename("epsW")
    epsT = (((2*np.pi/Umean)**(2/3)) * (Sh.freq**(5/3)) * Sh.sT * (epsU**(1/3)) / cT).median("freq").rename("epsT")

    epsilon = xr.merge([epsU, epsV, epsW, epsT])

    # slopes: fit log10(S) vs log10(f)
    Sh_log = np.log10(Sh).assign_coords(freq=np.log10(Sh.freq))
    Sl_log = np.log10(Sl).assign_coords(freq=np.log10(Sl.freq))

    slopes_h = Sh_log.polyfit("freq", deg=1).sel(degree=1).drop_vars("degree").rename(
        dict(
            su_polyfit_coefficients="slopeHU",
            sv_polyfit_coefficients="slopeHV",
            sw_polyfit_coefficients="slopeHW",
            sT_polyfit_coefficients="slopeHT",
        )
    )

    slopes_l = Sl_log.polyfit("freq", deg=1).sel(degree=1).drop_vars("degree").rename(
        dict(
            su_polyfit_coefficients="slopeLU",
            sv_polyfit_coefficients="slopeLV",
            sw_polyfit_coefficients="slopeLW",
            sT_polyfit_coefficients="slopeLT",
        )
    )

    slopes = xr.merge([slopes_h, slopes_l])
    return slopes, epsilon


def spectra_eps(ds, config, Umean, *, welch_segments=3, welch_overlap=0.5):
    """
    Entry point: computes Welch spectra + epsilon/slopes.
    """
    window = config["window"]

    # fluctuations (uses avg_method)
    ds_fluct = get_fluctuations(ds, config)

    welch = spectra_welch(
        ds_fluct,
        window,
        segments=welch_segments,
        overlap=welch_overlap,
        window_type="hann",
        detrend="constant",
    )

    slopes, epsilon = spectral_slopes_epsilon(welch, Umean)

    return welch, epsilon, slopes


import numpy as np
import xarray as xr


def bin_spectra_log(ds, N_bin=80, freq_name="freq", out_freq_name="freq_bin"):
    """
    Log-bin all spectral variables in a spectra Dataset along frequency.

    Input:  vars with dim `freq_name` (e.g. 'freq')
    Output: same vars but with dim `out_freq_name` (e.g. 'freq_bin') of length N_bin.
            At the end we rename out_freq_name -> freq_name so the output keeps 'freq'.
    """
    if freq_name not in ds.coords:
        raise ValueError(f"bin_spectra_log requires coordinate '{freq_name}'.")

    freq0 = np.asarray(ds[freq_name].values)
    mask_f = np.isfinite(freq0) & (freq0 > 0)
    freq = freq0[mask_f]

    if freq.size < 2:
        raise ValueError("Not enough positive finite frequencies to bin.")

    edges = np.logspace(np.log10(freq.min()), np.log10(freq.max()), N_bin + 1)
    freq_bin = 0.5 * (edges[:-1] + edges[1:])

    def _bin_1d(spec_1d):
        spec_1d = np.asarray(spec_1d)[mask_f]

        out = np.full(N_bin, np.nan, dtype=float)
        valid = np.isfinite(spec_1d)
        if valid.sum() < 2:
            return out

        f = freq[valid]
        s = spec_1d[valid]

        idx = np.digitize(f, edges) - 1
        ok = (idx >= 0) & (idx < N_bin)
        idx = idx[ok]
        s = s[ok]

        for b in range(N_bin):
            m = (idx == b)
            if np.any(m):
                out[b] = np.mean(s[m])

        return out

    out_vars = {}
    for name, da in ds.data_vars.items():
        if freq_name not in da.dims:
            out_vars[name] = da
            continue

        binned = xr.apply_ufunc(
            _bin_1d,
            da,
            input_core_dims=[[freq_name]],
            output_core_dims=[[out_freq_name]],
            vectorize=True,
            dask="allowed",
            output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {out_freq_name: N_bin}},
        )

        binned = binned.assign_coords({out_freq_name: freq_bin})
        out_vars[name] = binned

    out = xr.Dataset(data_vars=out_vars, coords=dict(ds.coords))
    out = out.drop_vars(freq_name, errors="ignore")
    out = out.assign_coords({out_freq_name: freq_bin})

    # keep using the usual coordinate name 'freq'
    out = out.rename({out_freq_name: freq_name})

    return out

