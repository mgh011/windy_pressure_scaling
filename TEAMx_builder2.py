import os
import re
import datetime as dt
from collections import defaultdict
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import struct

# tue funzioni
# from file_finder_reader import find_files_by_name_patterns, extract_date, read_TOB1, df_into_ds_consistent
def find_files_by_name_patterns(base_path, pattern1, pattern2='', pattern3=''):
    """
    Returns a list of all files (absolute paths) that contain from one up to
    three specified str patterns in the name.
    
    Parameters
    ----------
    base_path : str
        Path to the main directory (e.g., "/Users/.../TEAMx/raw_data").
    pattern1 : str
        e.g. Name of the station (e.g., "station_1").
    pattern2 : str
        e.g. Name of the station (e.g., "station_1").
    pattern3 : str
        e.g. Name of the station (e.g., "station_1").
    
    Returns
    -------
    matching_files : list of str
        A list of absolute paths to the files that match the specified date.
    """


    matching_files = []
    
    # Check if the station directory actually exists
    if not os.path.isdir(base_path):
        print(f"WARNING: The directory '{base_path}' does not exist or is not valid.")
        #return matching_files
        
    
    # Recursively walk through the base_path
    for root, dirs, files in os.walk(base_path):
        for file in files:
            conditions = []  # Lista per le condizioni da verificare
            
            if pattern1:  # Se pattern1 non è vuoto, lo verifichiamo
                conditions.append(pattern1 in file)
            if pattern2:  # Se pattern2 non è vuoto, lo verifichiamo
                conditions.append(pattern2 in file)
            if pattern3:  # Se pattern3 non è vuoto, lo verifichiamo
                conditions.append(pattern3 in file)
    
            # if conditions are defined, all have to be true
            if all(conditions):  # check if all True
                full_path = os.path.join(root, file)
                matching_files.append(full_path)
    return matching_files

def extract_date(filename):
    """
    Extracts the date from the filename by searching for a pattern in the format: YYYY_MM_DD.
    Returns a datetime object or None if the pattern is not found.
    """
    # Search for the date pattern in the filename
    match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
    if match:
        date_str = match.group(1)
        # Convert the string to a datetime object
        return datetime.datetime.strptime(date_str, "%Y_%m_%d")
    return None


def read_TOB1(datafile):
    
    def decode_fp2_from_bits(fp2_bits):
        """
        Decodes a 16-bit FP2 binary string into a floating-point number.
        
        Assumes FP2 is stored in **little-endian** byte order.
        
        hhttps://help.campbellsci.com/tx325/shared/formats/fp2.htm?TocPath=Data%20formats%20and%20transmission%20durations%7CPseudobinary%20data%20formats%7C_____1
        """
        # Convert binary string to an integer
        fp2_val = int(fp2_bits, 2)
        
        # Ensure we interpret the value as little-endian
        fp2_val = struct.unpack('<H', struct.pack('>H', fp2_val))[0]  # Swap endianness
    
        # Extract sign bit (bit 16)
        sign = (fp2_val >> 15) & 0x01  # Extract bit 16 (MSB in big-endian, swapped now)
    
        # Extract exponent bits (bits 15-14)
        exponent_bits = (fp2_val >> 13) & 0x03  # Extract bits 15 and 14 (2 bits)
        
        # Extract mantissa (bits 13-0)
        mantissa = fp2_val & 0x1FFF  # Mask the lowest 13 bits (0x1FFF = 8191)
    
        # Handle special cases for exponent == 00
        if exponent_bits == 0:
            if mantissa == 8191:
                return float('inf') if sign == 0 else float('-inf')
            elif mantissa == 8190:
                return float('nan')
    
        # Map exponent bits to actual exponent values
        exponent_map = {
            0b11: -3,  # 11 → 10⁻³
            0b10: -2,  # 10 → 10⁻²
            0b01: -1,  # 01 → 10⁻¹
            0b00:  0   # 00 → 10⁰
        }
        
        exponent = exponent_map[exponent_bits]
    
        # Compute the floating-point FP2 value
        value = ((-1) ** sign) * (10 ** exponent) * mantissa
        return value

    with open(datafile, mode='rb') as rfile:
            content = rfile.read()
        
    # -------------------------------------------------- #
    # ------- READ HEADER & FIELDS & PRECISION --------- #
    # -------------------------------------------------- #
    
    # Find carriage return (CR) & line feed (LF) positions
    lb = [match.span() for match in re.finditer(b'\r\n', content)]
    
    #header1 = content[0:lb[0][0]]
    header2 = content[lb[0][1]:lb[1][0]]
    header3 = content[lb[1][1]:lb[2][0]]
    #header4 = content[lb[2][1]:lb[3][0]]
    header5 = content[lb[3][1]:lb[4][0]]
    
    # get the pos of field names (variables), units, and precision. Use ',' as marker
    # and store them in the lists
    fc2 = [fc.start() for fc in re.finditer(b',', header2)]
    fc3 = [fc.start() for fc in re.finditer(b',', header3)]
    fc5 = [fc.start() for fc in re.finditer(b',', header5)]
    
    fc2.insert(0, -1)
    fc3.insert(0, -1)
    fc5.insert(0, -1)
    
    hvars = []  #variable list
    hunits = [] #unit list
    hprec = [] #precision list
    
    for cnt in range(len(fc2)-1):
        var = header2[fc2[cnt]+2:fc2[cnt+1]-1].decode()
        unit = header3[fc3[cnt]+2:fc3[cnt+1]-1].decode()
        prec = header5[fc5[cnt]+2:fc5[cnt+1]-1].decode()
        hvars.append(var)
        hunits.append(unit)
        hprec.append(prec)
    
    # Handle the last field
    var = header2[fc2[-1]+2:-1].decode()
    unit = header3[fc3[-1]+2:-1].decode()
    prec = header5[fc5[-1]+2:-1].decode()
    hvars.append(var)
    hunits.append(unit)
    hprec.append(prec)
    
        
    # -------------------------------------------------- #
    # ------- MARK THE PRECISION------------------------ #
    # -------------------------------------------------- #
    
    # Build the struct format string and dtype list
    fmt = '<'  # Little-endian
    dtype_list = []
    columns_to_remove = []
    fp2_field_names = [] 
    
    for i, iprec in enumerate(hprec):
        if iprec == 'ULONG':
            fmt += 'L'
            dtype_list.append((hvars[i], 'u4'))
        elif iprec == 'LONG':
            fmt += 'l'
            dtype_list.append((hvars[i], 'i4'))
        elif iprec == 'IEEE4':
            fmt += 'f'
            dtype_list.append((hvars[i], 'f4'))
        elif iprec in ['IEEE8', 'DOUBLE']:
            fmt += 'd'
            dtype_list.append((hvars[i], 'f8'))
        elif 'ASCII' in iprec:
            size = int(re.findall(r'\d+', iprec)[0])
            fmt += f'{size}s'
            dtype_list.append((hvars[i], f'S{size}'))
        elif iprec == 'ASCII(84)':
            fmt += '84x'  # Skip 84 bytes
            columns_to_remove.append(hvars[i])
        elif iprec == 'FP2':
            # For FP2, read 16 bits as an unsigned short
            fmt += 'H'
            """
            IMPORTANT
            NumPy’s dtype system does not recognize a custom type label like 'fp2'. 
            We want to treat those fields as 16‑bit unsigned integers 
            (format 'H' in struct, corresponding to NumPy dtype 'u2') and later convert 
            them (or simply display their bits).
            """
            dtype_list.append((hvars[i], 'u2'))
            fp2_field_names.append(hvars[i]) #keeps track of the fp2, now called u2
        else:
            print(f"Unknown precision: {iprec}. Skipped field.")
            columns_to_remove.append(hvars[i])
            
    
    
    
    sz = struct.calcsize(fmt)
    i1 = lb[4][1]  # Start of the data section.
    nl = (len(content) - i1) // sz
    
    
    # Read the data into a NumPy structured array using the updated dtype_list.
    dtype = np.dtype(dtype_list)
    data_array = np.frombuffer(content[i1:i1+nl*sz], dtype=dtype, count=nl)
    
    # Optionally, convert the structured array to a DataFrame.
    data = pd.DataFrame(data_array)
    for field in fp2_field_names:
        data[field] = data[field].apply(lambda x: decode_fp2_from_bits(format(x, '016b')))
    
    
    data.index.name = 'Time'
    dt = pd.Timestamp('1990-01-01 00:00:00') - pd.Timestamp(0, unit='ns')
    date = pd.DatetimeIndex(data.SECONDS*1e9+data.NANOSECONDS)+dt
    data.index = date
    data.drop(columns=['RECORD', 'SECONDS', 'NANOSECONDS'], inplace=True)
    
    return data

def df_into_ds_consistent(df, invert_sonics=False):
    import numpy as np
    import pandas as pd
    import xarray as xr

    measurements = ['pressure', 'Ux', 'Uy', 'Uz', 'SonTemp', 'Diag']
    heights = [1, 2]

    # Mapping colonne piatte → (variabile, quota)
    rename_map = {}
    for col in df.columns:
        if col == "hf_pressure_lvl_1":
            rename_map[col] = ("pressure", 1)
        elif col == "hf_pressure_lvl_2":
            rename_map[col] = ("pressure", 2)
        elif col == "Ux":
            rename_map[col] = ("Ux", 1)
        elif col == "Uy":
            rename_map[col] = ("Uy", 1)
        elif col == "Uz":
            rename_map[col] = ("Uz", 1)
        elif col == "SonTemp":
            rename_map[col] = ("SonTemp", 1)
        elif col == "Diag":
            rename_map[col] = ("Diag", 1)
        elif col == "Ux2":
            rename_map[col] = ("Ux", 2)
        elif col == "Uy2":
            rename_map[col] = ("Uy", 2)
        elif col == "Uz2":
            rename_map[col] = ("Uz", 2)
        elif col == "SonTemp2":
            rename_map[col] = ("SonTemp", 2)
        elif col == "Diag2":
            rename_map[col] = ("Diag", 2)

    # Rinomina le colonne
    df = df.rename(columns=rename_map)
    df = df.loc[:, list(rename_map.values())]
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    coords = {'time': df.index, 'height': heights}
    data_vars = {}

    for meas in measurements:
        n_times = len(df)
        data = np.full((n_times, len(heights)), np.nan)

        for i, h in enumerate(heights):
            key = (meas, h)
            if key in df.columns:
                data[:, i] = df[key].values

        arr = xr.DataArray(data, dims=['time', 'height'], coords=coords)
        data_vars[meas] = arr

    ds = xr.Dataset(data_vars)

    # ↩️ Inverto i dati sonici se richiesto (stazione 1)
    if invert_sonics:
        sonic_vars = ['Ux', 'Uy', 'Uz', 'SonTemp', 'Diag']
        for var in sonic_vars:
            ds[var].loc[:] = ds[var].sel(height=heights[::-1]).values  # swap quota 1 e 2

    return ds


# -------------------------
# CONFIG
# -------------------------
REPO_ROOT = "/Volumes/weop_hochhaeuser"
OUT_DIR   = "/Users/mauro_ghirardelli/Documents/TEAMx/raw/"

START_DAY = "2025-01-20"
END_DAY   = "2025-02-28"

ALL_TOWERS = ["s1","s2","s3","s4","s5","s6"]
station_to_tower = {f"station_{i}": f"s{i}" for i in range(1, 7)}
invert = {f"station_{i}": False for i in range(1, 7)}
invert["station_1"] = True  # sonic swap only (pressure stays correct)

# Candidate window (filename start time)
CAND_HOURS = 2

# Matches ONLY dated hf_data files
PAT = re.compile(
    r"""
    ^weop_(station_\d+)_          # station_1..6
    (?:hf_)?                      # optional hf_ (station_5)
    (CR1000XSeries|CR6Series)_    # logger
    hf_data_(\d{4})_(\d{2})_(\d{2})_(\d{4})\.dat$
    """,
    re.VERBOSE
)

def parse_start_from_name(filename: str) -> pd.Timestamp:
    m = PAT.match(filename)
    if not m:
        raise ValueError(filename)
    yyyy, mm, dd, hhmm = m.group(3), m.group(4), m.group(5), m.group(6)
    hh, mi = hhmm[:2], hhmm[2:]
    return pd.Timestamp(f"{yyyy}-{mm}-{dd} {hh}:{mi}:00")

def apply_qc_and_standardize(ds: xr.Dataset) -> xr.Dataset:
    # saturation
    sat = (np.abs(ds["Ux"]) > 65) | (np.abs(ds["Uy"]) > 65) | (np.abs(ds["Uz"]) > 65)
    for v in ["Ux", "Uy", "Uz", "SonTemp"]:
        ds[v] = ds[v].where(~sat)

    # pressure floor
    ds["pressure"] = ds["pressure"].where(ds["pressure"] >= 600)

    # diag flags
    bad = (ds["Diag"] == 1) | (ds["Diag"] == 2) | (ds["Diag"] == 4)
    for v in ["Ux", "Uy", "Uz", "SonTemp"]:
        ds[v] = ds[v].where(~bad)

    # rename to final
    ds = ds.rename({"pressure": "P", "Ux": "u", "Uy": "v", "Uz": "w", "SonTemp": "tc"})
    if "Diag" in ds:
        ds = ds.drop_vars("Diag")

    # time cleaning
    ds["time"] = pd.to_datetime(ds["time"].values)
    ds = ds.sortby("time")
    m = ~pd.Index(pd.to_datetime(ds["time"].values)).duplicated(keep="first")
    ds = ds.isel(time=m)
    return ds

def index_files_for_station(station_dir: str) -> pd.DataFrame:
    """
    Build a table: [path, start] for dated hf_data files only.
    (We ignore the single undated hf_data.dat here.)
    """
    files = find_files_by_name_patterns(station_dir, "hf_data")
    rows = []
    for p in files:
        b = os.path.basename(p)
        if (not b.endswith(".dat")) or (PAT.match(b) is None):
            continue
        try:
            st = parse_start_from_name(b)
        except Exception:
            continue
        rows.append((p, st))
    df = pd.DataFrame(rows, columns=["path", "start"]).sort_values("start")
    return df

def build_one_hour_one_station(index_df: pd.DataFrame, tower: str, hour: pd.Timestamp, invert_sonics: bool):
    """
    Returns hourly ds for ONE tower (or None if no data).
    """
    t0 = hour - pd.Timedelta(hours=CAND_HOURS)
    t1 = hour + pd.Timedelta(hours=CAND_HOURS)

    cand = index_df[(index_df["start"] >= t0) & (index_df["start"] <= t1)]["path"].tolist()
    chunks = []

    for p in cand:
        df = read_TOB1(p)
        ds = df_into_ds_consistent(df, invert_sonics=invert_sonics)  # swaps sonic only if True
        ds = apply_qc_and_standardize(ds)
        ds = ds.sel(time=slice(hour, hour + pd.Timedelta(hours=1) - pd.Timedelta(nanoseconds=1)))
        if ds.dims.get("time", 0) > 0:
            chunks.append(ds)

    if not chunks:
        return None, {"tower": tower, "candidates": len(cand), "chunks": 0, "samples": 0}

    dsh = xr.concat(chunks, dim="time").sortby("time")
    m = ~pd.Index(pd.to_datetime(dsh["time"].values)).duplicated(keep="first")
    dsh = dsh.isel(time=m)

    dsh = dsh.expand_dims({"tower": [tower]})
    return dsh, {"tower": tower, "candidates": len(cand), "chunks": len(chunks), "samples": dsh.dims["time"]}

def write_hour(ds_hour: xr.Dataset, hour: pd.Timestamp, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"teamx_{hour:%Y%m%d_%H00}.nc")
    ds_hour.to_netcdf(out_path)
    return out_path

def run_batch():
    # hours (inclusive start, inclusive end at 23:00)
    hours = pd.date_range(
        pd.Timestamp(START_DAY + " 00:00:00"),
        pd.Timestamp(END_DAY + " 23:00:00"),
        freq="1H",
    )

    # build per-station file indexes once
    print("Indexing files per station (from filenames)...")
    station_index = {}
    for station in station_to_tower:
        station_dir = os.path.join(REPO_ROOT, station)
        station_index[station] = index_files_for_station(station_dir)
        print(station, "->", len(station_index[station]), "dated hf files")

    os.makedirs(OUT_DIR, exist_ok=True)

    total_written = 0
    summary = {"hours": 0, "hours_with_any_data": 0, "missing_tower_hours": 0}

    # NetCDF encoding: float32 + optional compression/chunking
    encoding = {}
    for v in ["P", "u", "v", "w", "tc"]:
        encoding[v] = {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "chunksizes": (7200, 2, 1),  # 6 min × 2 heights × 1 tower
        }

    for hour in tqdm(hours, desc="Building hourly TEAMx"):
        summary["hours"] += 1

        per_tower = []
        reports = []

        for station, tower in station_to_tower.items():
            idx = station_index[station]
            ds_t, rep = build_one_hour_one_station(
                idx,
                tower=tower,
                hour=hour,
                invert_sonics=invert[station],
            )
            reports.append(rep)
            if ds_t is not None:
                per_tower.append(ds_t)

        # ---- CASE: no data at all -> write full NaN hour (float32) ----
        if len(per_tower) == 0:
            full_time = pd.date_range(
                hour,
                hour + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=50),
                freq="50ms",
            )
            tmpl = xr.Dataset(
                {
                    "P":  (("time", "height", "tower"), np.full((len(full_time), 2, 6), np.nan, dtype="float32")),
                    "u":  (("time", "height", "tower"), np.full((len(full_time), 2, 6), np.nan, dtype="float32")),
                    "v":  (("time", "height", "tower"), np.full((len(full_time), 2, 6), np.nan, dtype="float32")),
                    "w":  (("time", "height", "tower"), np.full((len(full_time), 2, 6), np.nan, dtype="float32")),
                    "tc": (("time", "height", "tower"), np.full((len(full_time), 2, 6), np.nan, dtype="float32")),
                },
                coords={"time": full_time, "height": [1, 2], "tower": ALL_TOWERS},
            )

            out_path = os.path.join(OUT_DIR, f"teamx_{hour:%Y%m%d_%H00}.nc")
            tmpl.to_netcdf(out_path, encoding=encoding, format="NETCDF4")
            total_written += 1
            summary["missing_tower_hours"] += 6
            continue

        summary["hours_with_any_data"] += 1

        # merge towers present
        ds_hour = xr.concat(per_tower, dim="tower", join="outer")
        ds_hour = ds_hour.transpose("time", "height", "tower").sortby("time")

        # Force tower always = 6 (missing towers become NaN)
        ds_hour = ds_hour.reindex(tower=ALL_TOWERS)

        # Write
        out_path = os.path.join(OUT_DIR, f"teamx_{hour:%Y%m%d_%H00}.nc")
        ds_hour.to_netcdf(out_path, encoding=encoding, format="NETCDF4")
        total_written += 1

        # accounting: how many towers missing this hour?
        present = [r["tower"] for r in reports if r["samples"] > 0]
        summary["missing_tower_hours"] += (6 - len(present))

    print("\nDone.")
    print("Total hourly files written:", total_written)
    print("Hours processed:", summary["hours"])
    print("Hours with any data:", summary["hours_with_any_data"])
    print("Total missing tower-hours (sum over hours):", summary["missing_tower_hours"])
    print("Output dir:", OUT_DIR)

run_batch()



#%%
file = "/Users/mauro_ghirardelli/Documents/TEAMx/raw/teamx_20250120_0700.nc"
file2 = "/Users/mauro_ghirardelli/Documents/SCP/raw/scp_tc_20121117_20.nc"
import xarray as xr
ds = xr.open_dataset(file)
print(ds)
      
#%%
import xarray as xr
ds = xr.open_dataset("/Users/mauro_ghirardelli/Documents/TEAMx/raw/teamx_20250120_0100.nc")
print(ds)

print(dict(ds.dims))
print(ds["tower"].values)
print(ds["time"][0].values, ds["time"][-1].values)
print(float(ds["u"].sel(tower="s1", height=1).isnull().mean())*100, "% NaN in u(s1,h=1)")

