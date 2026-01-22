#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch launcher for hourly ISFS NetCDF preprocessing
==================================================

Purpose
-------
Iterate over all hourly ISFS NetCDF files in an input directory and call an
external preprocessing script (the "single-file processor") on each file.

The single-file processor is assumed to:
- take two command-line arguments:
    1) input NetCDF file path
    2) output directory
- write two outputs to the output directory:
    <stem>_hf20.nc   (high-frequency, 20 Hz)
    <stem>_lf1.nc    (low-frequency, 1 Hz)

This launcher does not perform any data processing itself.
It only orchestrates which files are processed and optionally skips files
that have already been processed.

Design notes
------------
- Keeping orchestration separate from scientific processing makes it easier to:
  * run locally for debugging
  * run on HPC (e.g. SLURM) by swapping the launcher with a job array
  * re-run only missing outputs
"""
import os
import subprocess

def run_preprocessing_batch(
    input_dir,
    output_dir,
    processor_script,
    pattern=".nc",
    skip_existing=True,
    python_executable="python",
):
    """
    Run the single-file preprocessing script on all matching files in `input_dir`.

    Parameters
    ----------
    input_dir : str
        Directory containing raw hourly NetCDF files.
    output_dir : str
        Directory where processed outputs will be written.
    processor_script : str
        Path to the single-file preprocessing script (e.g. process_one_hour.py).
        The script must accept:
            python processor_script <infile> <output_dir>
    pattern : str
        File suffix to select inputs (default: '.nc').
    skip_existing : bool
        If True, skip files for which both expected outputs already exist.
    python_executable : str
        Python executable used to run the processor script (default: 'python').

    Returns
    -------
    processed : list of str
        List of input file paths that were processed (i.e. not skipped).
    skipped : list of str
        List of input file paths that were skipped because outputs existed.
    """
    processed = []
    skipped = []
    failed = []

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(pattern))

    for fname in files:
        infile = os.path.join(input_dir, fname)
        stem = os.path.splitext(fname)[0]

        out_hf = os.path.join(output_dir, f"{stem}_hf20.nc")
        out_lf = os.path.join(output_dir, f"{stem}_lf1.nc")

        # --- skip if outputs already exist ---
        if skip_existing and os.path.exists(out_hf) and os.path.exists(out_lf):
            skipped.append(infile)
            continue

        try:
            subprocess.run(
                [python_executable, processor_script, infile, output_dir],
                check=True,
                capture_output=True,
                text=True,
            )
            processed.append(infile)

        except subprocess.CalledProcessError as e:
            print("\n" + "="*80)
            print("FAILED:", infile)
            print("Return code:", e.returncode)
            if e.stderr:
                print("\n--- STDERR ---\n", e.stderr)
            print("="*80 + "\n")

            failed.append(infile)
            continue

    return processed, skipped, failed


##
if __name__ == "__main__":
    INPUT_DIR = "/Users/mauro_ghirardelli/Documents/SCP/raw"
    OUTPUT_DIR = "/Users/mauro_ghirardelli/Documents/SCP/hourly"
    PROCESSOR_SCRIPT = "/Users/mauro_ghirardelli/Documents/windy_pressure_scaling/SCP_builder.py"

    processed, skipped, failed = run_preprocessing_batch(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        processor_script=PROCESSOR_SCRIPT,
        pattern=".nc",
        skip_existing=True,
        python_executable="python",
    )
    
    print(f"Processed: {len(processed)} files")
    print(f"Skipped:   {len(skipped)} files")
    print(f"Failed:    {len(failed)} files")
