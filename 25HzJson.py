'''
This script builds a per-day dataset of "walking path" signature-ready arrays
from raw 25Hz triaxial accelerometer data and 1Hz activity labels, then saves
the result as a JSON file.

Pipeline (executed top to bottom when the script is run):
1. Load the raw 25Hz signal and 1Hz label CSVs for a subject/session.
2. Align the 1Hz labels to the 25Hz raw samples by flooring timestamps to
   the nearest second.
3. Print a spot-check of the alignment.
4. For every calendar date present in the raw data:
   a. Slice out that day's raw + label data and align them.
   b. Extract continuous "walking" bursts (label == 1) of at least
      min_burst_seconds duration.
   c. For each burst: bandpass filter, downsample from sample_rate to
      target_rate, per-burst standard-scale, and window into fixed-length
      "paths" of path_samples length, discarding flat/outlier paths.
   d. Concatenate all paths for the day; keep the day only if it has at
      least min_paths_per_day paths.
5. Collect all valid days into a dict of {date: paths}, convert to plain
   lists, and write it out as a JSON file under CONFIG['target_path'].

Functions:
- parse_time_column(df, col): Parses the accProcess timestamp string format
  (e.g. '2022-02-25 14:00:07.116+0000 [Europe/London]') into a
  timezone-naive pandas datetime column.
- load_raw_and_labels(raw_file, label_file): Loads the raw 25Hz
  accelerometer CSV and the label CSV, parses their time columns, sorts by
  time, and reports basic diagnostics (shape, sample rate, duration).
- align_labels_to_raw(raw_df, label_df, label_col): Merges 1Hz labels onto
  25Hz raw samples by flooring both to the nearest second, and reports
  merge/labelling statistics.
- extract_walking_bursts(merged_df, label_col, sample_rate,
  min_burst_seconds): Filters the merged data to walking-labelled samples,
  splits them into contiguous bursts based on timestamp gaps, and returns
  only bursts at least min_burst_seconds long.
- burst_to_paths_xyz(burst_df, path_length_seconds, sample_rate,
  target_rate, bandpass_low, bandpass_high, bandpass_order): Bandpass
  filters and downsamples a walking burst, standard-scales it, splits it
  into fixed-length "paths", and removes flat or outlier paths. Returns an
  array of shape (num_paths, path_samples, 3).
- verify_alignment(merged_df, n_samples): Prints a small table of raw
  samples with their labels and floored timestamps for manual sanity
  checking.
- process_day_xyz(raw_df, label_df, date_str, path_length, sample_rate,
  target_rate, bandpass_low, bandpass_high, min_paths): Runs the full
  per-day pipeline (filter to date, align, extract bursts, convert to
  paths) and returns the day's path array, or None if the day doesn't
  qualify.
'''

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json, sys, os

input_folder = sys.argv[1]
input_sub_folder = sys.argv[2]
print(f'Input Folder: {input_folder}')
print(f'Input Sub Folder: {input_sub_folder}')
# This script expects a 'CONFIG' dict to already be defined before this point
# (e.g. imported from a project-specific config module, or otherwise supplied
# by whatever caller runs this script). It must contain the following keys:
#   base_path            					str          Root directory containing the input data.
#   target_path           					str          Root directory to write the output JSON to.
#   sample_rate              				int/float    Hz, sample rate of the raw signal.
#   target_rate                				int/float    Hz, sample rate after downsampling.
#   bandpass_low, bandpass_high    			float        Hz, bandpass filter cutoffs.
#   path_length                     		int          Seconds per extracted path.
#   path_samples                         	int          Samples per path (path_length * target_rate).
#   min_paths_per_day                       int          Minimum paths required to keep a day.
#   label_col                               str          Column name holding the activity label.
#
# The keys below are derived at runtime from the command-line arguments and
# are attached to the caller-supplied CONFIG here.
CONFIG['folder']     = f'{input_folder}/{input_sub_folder}'
CONFIG['raw_file']   = f'{input_sub_folder}.csv'
CONFIG['label_file'] = f'{input_sub_folder}-timeSeries.csv'

def parse_time_column(df, col='time'):
    print(f"Raw sample: '{df[col].iloc[0]}'")

    df[col] = df[col].str.replace(
        r'\s*\[.*?\]',
        '',
        regex=True
    )

    print(f"After strip: '{df[col].iloc[0]}'")

    df[col] = pd.to_datetime(
        df[col],
        utc=True,
        format='%Y-%m-%d %H:%M:%S.%f%z'
    )

    df[col] = df[col].dt.tz_localize(None)

    print(f"Parsed dtype:  {df[col].dtype}")
    print(f"Sample parsed: {df[col].iloc[0]}")

    return df

def load_raw_and_labels(raw_file, label_file):
    print("Loading raw signal...")
    raw_df = pd.read_csv(
        raw_file,
        parse_dates=['time'],
        dtype={
            'x': np.float32,
            'y': np.float32,
            'z': np.float32
        }
    )
    raw_df = parse_time_column(raw_df, col='time')
    raw_df = raw_df.sort_values('time').reset_index(drop=True)
    raw_df['time'] = pd.to_datetime(
        raw_df['time']
    )
    time_diffs  = raw_df['time'].diff().dropna()
    median_diff = time_diffs.median().total_seconds()
    sample_rate = 1 / median_diff

    print(f"Shape:       {raw_df.shape}")
    print(f"Sample rate: {sample_rate:.1f}Hz")
    print(f"Duration:    {raw_df['time'].min()} → "
          f"{raw_df['time'].max()}")

    print("\nLoading labels...")
    label_df = pd.read_csv(
        label_file,
        parse_dates=['time']
    )
    label_df = parse_time_column(label_df, col='time')
    label_df = label_df.sort_values('time').reset_index(drop=True)

    print(f"Label shape:   {label_df.shape}")
    print(f"Label duration:{label_df['time'].min()} → "
          f"{label_df['time'].max()}")
    print(f"Label columns: {label_df.columns.tolist()}")
    print(f"Walking %:     "
          f"{label_df[CONFIG['label_col']].mean()*100:.1f}%")

    return raw_df, label_df

def align_labels_to_raw(raw_df, label_df,
                          label_col=CONFIG['label_col']):
    raw_df['time_floor'] = raw_df['time'].dt.floor('1s')

    label_df['time_floor'] = label_df['time'].dt.floor('1s')

    merged = raw_df.merge(
        label_df[['time_floor', label_col]],
        on='time_floor',
        how='left'
    )

    n_total    = len(merged)
    n_labelled = merged[label_col].notna().sum()
    n_walking  = (merged[label_col] == 1).sum()

    print(f"Total raw samples:    {n_total}")
    print(f"Labelled samples:     {n_labelled} "
          f"({n_labelled/n_total*100:.1f}%)")
    print(f"Walking samples:      {n_walking} "
          f"({n_walking/n_total*100:.1f}%)")
    print(f"Walking duration:     "
          f"{n_walking/25/60:.1f} minutes")

    return merged

def extract_walking_bursts(merged_df,
                            label_col=CONFIG['label_col'],
                            sample_rate=25,
                            min_burst_seconds=15):
    walking_df = merged_df[
        merged_df[label_col] == 1
    ].copy().reset_index(drop=True)

    if len(walking_df) == 0:
        print("No walking samples found")
        return []

    time_diff = walking_df['time'].diff()
    expected  = pd.Timedelta(milliseconds=1000/sample_rate)
    tolerance = pd.Timedelta(milliseconds=100)

    walking_df['new_burst'] = (
        time_diff > expected + tolerance
    ).fillna(True)
    walking_df['burst_id']  = walking_df['new_burst'].cumsum()

    burst_lengths = walking_df.groupby('burst_id').size()
    min_samples   = min_burst_seconds * sample_rate
    valid_bursts  = burst_lengths[
        burst_lengths >= min_samples
    ].index

    print(f"Total bursts found:   {walking_df['burst_id'].max()}")
    print(f"Valid bursts (≥{min_burst_seconds}s): "
          f"{len(valid_bursts)}")

    bursts = []
    for burst_id in valid_bursts:
        burst = walking_df[
            walking_df['burst_id'] == burst_id
        ][['time', 'x', 'y', 'z']].reset_index(drop=True)
        bursts.append(burst)

    return bursts

from scipy.signal import butter, sosfiltfilt, decimate

def burst_to_paths_xyz(burst_df,
                        path_length_seconds=15,
                        sample_rate=25,
                        target_rate=5,
                        bandpass_low=0.5,
                        bandpass_high=10.0,
                        bandpass_order=4):
    signal = burst_df[['x', 'y', 'z']].values

    if len(signal) < sample_rate * path_length_seconds:
        print(f"Burst too short: {len(signal)} samples")
        return None

    sos = butter(
        bandpass_order,
        [bandpass_low, bandpass_high],
        fs=sample_rate,
        btype='bandpass',
        output='sos'
    )
    filtered = np.zeros_like(signal)
    for ch in range(3):
        filtered[:, ch] = sosfiltfilt(sos, signal[:, ch])

    downsample_factor = sample_rate // target_rate

    n_samples_out = (len(filtered) // downsample_factor) \
                    * downsample_factor
    filtered      = filtered[:n_samples_out]

    downsampled = np.zeros(
        (len(filtered) // downsample_factor, 3)
    )
    for ch in range(3):
        downsampled[:, ch] = decimate(
            filtered[:, ch],
            q          = downsample_factor,
            ftype      = 'fir',
            zero_phase = True
        )

    from sklearn.preprocessing import StandardScaler
    scaler      = StandardScaler()
    scaled_2d   = scaler.fit_transform(downsampled)

    print(f"  Burst scaled mean: "
          f"{scaled_2d.mean(axis=0).round(4)}")
    print(f"  Burst scaled std:  "
          f"{scaled_2d.std(axis=0).round(4)}")

    path_samples = path_length_seconds * target_rate
    n_paths      = len(scaled_2d) // path_samples

    if n_paths == 0:
        print("Not enough samples for any path")
        return None

    paths = np.array([
        scaled_2d[i*path_samples:(i+1)*path_samples]
        for i in range(n_paths)
    ])

    norms = np.array([
        np.linalg.norm(paths[i])
        for i in range(len(paths))
    ])

    print(f"  Paths created:     {n_paths}")
    print(f"  Path shape:        {paths.shape}")
    print(f"  Norm mean/max:     "
          f"{norms.mean():.4f} / {norms.max():.4f}")

    stds  = np.array([
        paths[i].std(axis=0).min()
        for i in range(len(paths))
    ])
    norm_threshold = norms.mean() + 3 * norms.std()
    valid_mask     = (stds > 1e-6) & (norms < norm_threshold)

    if (~valid_mask).sum() > 0:
        print(f"  Removed {(~valid_mask).sum()} invalid paths")

    paths = paths[valid_mask]

    return paths.astype(np.float64)

def verify_alignment(merged_df, n_samples=5):
    print("Alignment verification:")
    print(f"{'Time (25Hz)':<30} {'x':>8} {'y':>8} "
          f"{'z':>8} {'label':>8} {'floor':>25}")
    print("-" * 90)

    walking = merged_df[
        merged_df[CONFIG['label_col']] == 1
    ].head(n_samples * 25)

    for _, row in walking.head(n_samples).iterrows():
        print(f"{str(row['time']):<30} "
              f"{row['x']:>8.4f} {row['y']:>8.4f} "
              f"{row['z']:>8.4f} "
              f"{int(row[CONFIG['label_col']]):>8} "
              f"{str(row['time_floor']):>25}")

def process_day_xyz(raw_df, label_df,
                     date_str,
                     path_length    = 15,
                     sample_rate    = 25,
                     target_rate    = 5,
                     bandpass_low   = 0.5,
                     bandpass_high  = 10.0,
                     min_paths      = 20):
    date_mask_raw   = raw_df['time'].dt.date.astype(str) \
                      == date_str
    date_mask_label = label_df['time'].dt.date.astype(str) \
                      == date_str

    day_raw   = raw_df[date_mask_raw].copy()
    day_label = label_df[date_mask_label].copy()

    if len(day_raw) == 0:
        print(f"{date_str}: no raw data")
        return None

    if len(day_label) == 0:
        print(f"{date_str}: no label data")
        return None

    merged = align_labels_to_raw(
        day_raw, day_label, label_col=CONFIG['label_col']
    )

    bursts = extract_walking_bursts(
        merged_df         = merged,
        label_col         = CONFIG['label_col'],
        sample_rate       = sample_rate,
        min_burst_seconds = path_length
    )

    if len(bursts) == 0:
        print(f"{date_str}: no valid bursts")
        return None

    all_paths = []
    for burst in bursts:
        paths = burst_to_paths_xyz(
            burst_df            = burst,
            path_length_seconds = path_length,
            sample_rate         = sample_rate,
            target_rate         = target_rate,
            bandpass_low        = bandpass_low,
            bandpass_high       = bandpass_high
        )
        if paths is not None and len(paths) > 0:
            all_paths.append(paths)

    if len(all_paths) == 0:
        print(f"{date_str}: no valid paths")
        return None

    paths_day = np.concatenate(all_paths, axis=0)

    if len(paths_day) < min_paths:
        print(f"Skipping {date_str} — "
              f"only {len(paths_day)} paths")
        return None

    return paths_day

RAW_FILE = f"{CONFIG['base_path']}/{CONFIG['folder']}/{CONFIG['raw_file']}"
LABEL_FILE = f"{CONFIG['base_path']}/{CONFIG['folder']}/{CONFIG['label_file']}"
raw_df, label_df = load_raw_and_labels(
    RAW_FILE, LABEL_FILE
)

merged = align_labels_to_raw(
    raw_df, label_df, label_col=CONFIG['label_col']
)

verify_alignment(merged, n_samples=5)

all_dates = sorted(raw_df['time'].dt.date.unique())

all_daily_paths = {}
for date in all_dates:
    date_str = str(date)
    paths    = process_day_xyz(
        raw_df        = raw_df,
        label_df      = label_df,
        date_str      = date_str,
        path_length   = CONFIG['path_length'],
        sample_rate   = CONFIG['sample_rate'],
        target_rate   = CONFIG['target_rate'],
        bandpass_low  = CONFIG['bandpass_low'],
        bandpass_high = CONFIG['bandpass_high'],
        min_paths     = CONFIG['min_paths_per_day']
    )

    if paths is not None:
        all_daily_paths[date_str] = paths
        print(f"{date_str}: {len(paths)} paths "
              f"{paths.shape}")
    else:
        print(f"{date_str}: skipped")

daily_paths_json = {}
for k,v in all_daily_paths.items():
    daily_paths_json[k] = v.tolist()
save_dir = f'{CONFIG["target_path"]}/{CONFIG["folder"]}'
save_path = f'{CONFIG["target_path"]}/{CONFIG["folder"]}/processed_paths_{CONFIG["label_col"]}_{CONFIG["path_samples"]}.json'
os.makedirs(save_dir, exist_ok=True)
with open(save_path, 'w') as f:
	json.dump(daily_paths_json, f)
