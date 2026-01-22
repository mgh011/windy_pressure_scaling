#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Windpy batch launcher
====================

Iterates over all HF files found in cfg["paths"]["hf_dir"] and calls
windpy_process_one.py for each file.

Input directory  -> cfg["paths"]["hf_dir"]
Output directory -> cfg["paths"]["out_dir"]

Skips files for which *_30min_stats.nc already exists.
"""

import os
import json
import subprocess


def load_config_json(path):
    with open(path, "r") as f:
        return json.load(f)


def run_windpy_batch(
    cfg_path,
    processor_script,
    pattern=".nc",
    python_executable="python",
    skip_existing=True,
):
    """
    Run windpy_process_one.py on all HF files listed in cfg["paths"]["hf_dir"].
    """
    cfg = load_config_json(cfg_path)

    input_dir = cfg["paths"]["hf_dir"]
    output_dir = cfg["paths"]["out_dir"]

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(pattern) and ("hf" in f.lower())
    )


    processed = []
    skipped = []
    failed = []

    for fname in files:
        stem = os.path.splitext(fname)[0]
        out_stats = os.path.join(output_dir, f"{stem}_30min_stats.nc")

        if skip_existing and os.path.exists(out_stats):
            print(f"[SKIP] {fname}")
            skipped.append(fname)
            continue

        cmd = [python_executable, processor_script, cfg_path, fname, output_dir]

        print(f"[RUN] {fname}")
        try:
            subprocess.run(cmd, check=True)
            processed.append(fname)
        except subprocess.CalledProcessError:
            print(f"[FAIL] {fname}")
            failed.append(fname)

    return processed, skipped, failed


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":

    CONFIG_PATH = "/Users/mauro_ghirardelli/Documents/windy_pressure_scaling/windy/conf/SCP_configuration.txt"
    PROCESSOR_SCRIPT = "/Users/mauro_ghirardelli/Documents/windy_pressure_scaling/windy/windpy.py"

    processed, skipped, failed = run_windpy_batch(
        cfg_path=CONFIG_PATH,
        processor_script=PROCESSOR_SCRIPT,
        pattern=".nc",
        python_executable="python",
        skip_existing=True,
    )

    print("\n=== SUMMARY ===")
    print(f"Processed: {len(processed)}")
    print(f"Skipped:   {len(skipped)}")
    print(f"Failed:    {len(failed)}")

