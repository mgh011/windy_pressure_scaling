# windy/main.py
#
# Step 0: read configuration, sanity-check folders, discover HF/LF pairs.
#
# Notes:
# - Documentation is in English.
# - No type annotations are used.
# - Pairing rule: files are paired by the common prefix before the last token
#   that starts with "hf" or "lf" (e.g. *_hf10.nc, *_lf1.nc).

import json
from pathlib import Path


def load_config(config_path):
    """
    Load a JSON configuration file.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the JSON config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    with config_path.open("r") as f:
        return json.load(f)


def split_hf_lf_key(filename):
    """
    Extract (key, kind) from a file name using the last underscore token.

    The expected naming convention is that the last underscore-separated token
    (before the extension) starts with either 'hf' or 'lf', e.g.:
      - scp_tc_20121129_04_hf10.nc -> key='scp_tc_20121129_04', kind='hf'
      - scp_tc_20121129_04_lf1.nc  -> key='scp_tc_20121129_04', kind='lf'
      - anything_else.nc           -> (None, None)

    Parameters
    ----------
    filename : str
        File name only (no directory needed).

    Returns
    -------
    (key, kind) : (str or None, str or None)
        kind is 'hf' or 'lf'. If not recognized, both are None.
    """
    p = Path(filename)
    stem = p.stem  # filename without extension
    parts = stem.split("_")
    if not parts:
        return None, None

    last = parts[-1].lower()
    if last.startswith("hf"):
        return "_".join(parts[:-1]), "hf"
    if last.startswith("lf"):
        return "_".join(parts[:-1]), "lf"
    return None, None


def discover_pairs_generic(hf_dir, lf_dir, ext=".nc"):
    """
    Discover and pair HF/LF files from two directories.

    Pairing rule:
      - HF files are those whose last token starts with 'hf'
      - LF files are those whose last token starts with 'lf'
      - Pair key is the common prefix before that last token

    Parameters
    ----------
    hf_dir : str or pathlib.Path
        Directory containing HF files.
    lf_dir : str or pathlib.Path
        Directory containing LF files.
    ext : str
        File extension to scan (default '.nc').

    Returns
    -------
    list
        List of (hf_path, lf_path, key) tuples, sorted by key.
    """
    hf_dir = Path(hf_dir)
    lf_dir = Path(lf_dir)

    hf_map = {}
    lf_map = {}

    # Scan HF directory
    for p in hf_dir.glob("*" + ext):
        key, kind = split_hf_lf_key(p.name)
        if kind != "hf":
            continue
        # If duplicates exist for the same key, keep the lexicographically smallest name
        if key not in hf_map:
            hf_map[key] = p
        else:
            hf_map[key] = min(hf_map[key], p)

    # Scan LF directory
    for p in lf_dir.glob("*" + ext):
        key, kind = split_hf_lf_key(p.name)
        if kind != "lf":
            continue
        if key not in lf_map:
            lf_map[key] = p
        else:
            lf_map[key] = min(lf_map[key], p)

    keys = sorted(set(hf_map.keys()).intersection(set(lf_map.keys())))
    pairs = []
    for k in keys:
        pairs.append((hf_map[k], lf_map[k], k))

    return pairs


def step0(config_path):
    """
    Step 0: read config, check folders, and build the HF/LF run list.

    This does not open any datasets or run any computations.
    It only prepares the list of file pairs to process.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the JSON configuration.

    Returns
    -------
    (cfg, pairs) : (dict, list)
        cfg is the config dict, pairs is list of (hf_path, lf_path, key).
    """
    cfg = load_config(config_path)

    # Required fields
    paths = cfg["paths"]
    hf_dir = Path(paths["hf_dir"])
    lf_dir = Path(paths["lf_dir"])
    out_dir = Path(paths["out_dir"])

    # Sanity checks
    if not hf_dir.exists():
        raise FileNotFoundError("hf_dir not found: " + str(hf_dir))
    if not lf_dir.exists():
        raise FileNotFoundError("lf_dir not found: " + str(lf_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover pairs
    pairs = discover_pairs_generic(hf_dir, lf_dir)

    # Report
    print("\n" + "=" * 90)
    print("[windy] STEP 0 report")
    print("=" * 90)
    print("dataset :", cfg.get("dataset"))
    print("window  :", cfg.get("window"))
    print("hf_dir  :", str(hf_dir))
    print("lf_dir  :", str(lf_dir))
    print("out_dir :", str(out_dir))
    print("pairs   :", len(pairs))

    if pairs:
        hf0, lf0, k0 = pairs[0]
        hfl, lfl, kl = pairs[-1]
        print("\nfirst pair")
        print("  key:", k0)
        print("  hf :", hf0.name)
        print("  lf :", lf0.name)

        print("\nlast pair")
        print("  key:", kl)
        print("  hf :", hfl.name)
        print("  lf :", lfl.name)

    return cfg, pairs


if __name__ == "__main__":
    # Example usage:
    # Update CFG to your real configuration file path.
    CFG = "/Users/mauro_ghirardelli/Documents/windy_pressure_scaling/windy/conf/SCP_configuration.txt"
    step0(CFG)
