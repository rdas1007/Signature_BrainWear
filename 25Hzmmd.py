"""
This module expects a 'CONFIG' dict to already exist in the calling
namespace (e.g. imported from a separate config module) before 'main()`'
is invoked. Required CONFIG keys:

    base_path                 (str)                   Directory containing per-recording JSON path files, organised by patient folder.
    target_path               (str)                   Directory where MMD results and plots are written.
    process                   (str)                   'RAW' or 'ENMO' — selects whether raw 3-axis paths or ENMO+time paths are loaded.
    path_samples              (int)                   Number of samples per path; used to build the per-recording JSON filename and the output directory name.
    baseline_days             (int)                   Number of days requested for the baseline window before each clinic visit.
    windows                   (list[int])             MMD window sizes, in days, to compute relative to each clinic visit.
    dyadic_order              (int)                   Dyadic order parameter passed to the signature MMD computation.
    max_mmd                   (float)                 MMD values above this threshold are treated as invalid/overflow and replaced with NaN.
    max_paths                 (int)                   Maximum number of paths to subsample per baseline/window batch before computing MMD.
    patient_metadata_path     (str)                   Path to the spreadsheet containing each patient's stable/progression clinic visit dates.

The patient/session folder to process is read from sys.argv[1].

"""

import json
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import pysiglib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")


def transform_to_enmo_time(paths):
    norm = np.linalg.norm(paths, axis=-1)
    enmo = np.maximum(norm - 1.0, 0.0)
    enmo = enmo[..., np.newaxis]

    T = paths.shape[1]
    time = np.linspace(0.0, 1.0, T)
    time = np.broadcast_to(
        time[np.newaxis, :, np.newaxis],
        (paths.shape[0], T, 1)
    ).copy()

    return np.concatenate([enmo, time], axis=-1)


def load_and_merge_jsons(file_list):
    all_daily_paths = {}

    for json_file in file_list:
        try:
            print(f"Loading: {json_file}")

            with open(json_file, 'r') as f:
                file_paths = json.load(f)

            for date_str, paths_list in file_paths.items():
                arr = np.array(paths_list, dtype=np.float64)
                if date_str in all_daily_paths:
                    all_daily_paths[date_str] = np.concatenate(
                        [all_daily_paths[date_str], arr], axis=0
                    )
                else:
                    all_daily_paths[date_str] = arr

        except Exception as e:
            print(f"Failed to load {json_file}: {e}")

    if not all_daily_paths:
        print("\nNo data loaded.")
        return all_daily_paths

    print(f"\nTotal days:  {len(all_daily_paths)}")
    print(f"Date range:  "
          f"{min(all_daily_paths.keys())} → "
          f"{max(all_daily_paths.keys())}")

    return all_daily_paths


def load_and_merge_jsons_enmo(file_list):
    all_daily_paths = {}

    for json_file in file_list:
        try:
            print(f"Loading: {json_file}")

            with open(json_file, 'r') as f:
                file_paths = json.load(f)

            for date_str, paths_list in file_paths.items():
                raw = np.array(paths_list, dtype=np.float64)

                if raw.ndim != 3 or raw.shape[2] != 3:
                    print(f"  Unexpected shape {raw.shape} for "
                          f"{date_str} — skipping")
                    continue

                transformed = transform_to_enmo_time(raw)

                if date_str in all_daily_paths:
                    all_daily_paths[date_str] = np.concatenate(
                        [all_daily_paths[date_str], transformed], axis=0
                    )
                else:
                    all_daily_paths[date_str] = transformed

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    if not all_daily_paths:
        print("\nNo data loaded.")
        return all_daily_paths

    print(f"\nTotal days:  {len(all_daily_paths)}")
    print(f"Date range:  "
          f"{min(all_daily_paths.keys())} → "
          f"{max(all_daily_paths.keys())}")

    sample_date = next(iter(all_daily_paths))
    sample = all_daily_paths[sample_date]
    print(f"\nTransformed shape check ({sample_date}): "
          f"{sample.shape} — expected (n, T, 2)")
    print(f"  ENMO  range: "
          f"[{sample[:,:,0].min():.4f}, {sample[:,:,0].max():.4f}]")
    print(f"  Time  range: "
          f"[{sample[:,:,1].min():.4f}, {sample[:,:,1].max():.4f}]")

    return all_daily_paths


def build_final_dict(all_daily_paths,
                      clinic_dates,
                      baseline_days=30,
                      min_baseline_days=7):
    sorted_clinics = sorted(clinic_dates)

    available_dates = set(all_daily_paths.keys())

    dt_available = sorted([
        datetime.strptime(d, "%Y-%m-%d")
        for d in available_dates
    ])

    first_data_date = dt_available[0]
    last_data_date = dt_available[-1]

    print(f"First data date:      {first_data_date.date()}")
    print(f"Last data date:       {last_data_date.date()}")
    print(f"Total days with data: {len(available_dates)}")

    print(f"\nValidating clinic dates against data range:")
    valid_clinics = []

    for clinic_date in sorted_clinics:
        if clinic_date < first_data_date:
            print(f"{clinic_date.date()} is BEFORE first data date "
                  f"({first_data_date.date()}) — skipping")
            continue

        if clinic_date > last_data_date:
            print(f"{clinic_date.date()} is AFTER last data date "
                  f"({last_data_date.date()}) — skipping")
            continue

        print(f"{clinic_date.date()} within data range")
        valid_clinics.append(clinic_date)

    if len(valid_clinics) == 0:
        raise ValueError(
            f"No valid clinic dates found.\n"
            f"Data range: {first_data_date.date()} → "
            f"{last_data_date.date()}\n"
            f"Clinic dates: {[d.date() for d in sorted_clinics]}"
        )

    if len(valid_clinics) < len(sorted_clinics):
        print(f"\n{len(sorted_clinics) - len(valid_clinics)} "
              f"clinic dates outside data range — excluded")

    sorted_clinics = valid_clinics

    final = {}

    for idx, clinic_date in enumerate(sorted_clinics):

        print(f"\n{'='*50}")
        print(f"Clinic {idx+1} ({clinic_date.date()}):")

        if idx == 0:
            baseline_start = first_data_date
            print(f"  Baseline start: {baseline_start.date()} "
                  f"(first data day)")
        else:
            baseline_start = sorted_clinics[idx-1] + timedelta(days=1)
            print(f"  Baseline start: {baseline_start.date()} "
                  f"(day after clinic {idx})")

        phase1_end = baseline_start + timedelta(days=baseline_days - 1)
        phase1_end = min(phase1_end, clinic_date - timedelta(days=1))

        phase1_data = {
            date_str: all_daily_paths[date_str]
            for date_str in available_dates
            if baseline_start <= datetime.strptime(
                date_str, "%Y-%m-%d"
            ) <= phase1_end
        }

        days_found = len(phase1_data)
        calendar_days = (phase1_end - baseline_start).days + 1

        print(f"  Phase 1 window:   {baseline_start.date()} → "
              f"{phase1_end.date()} ({calendar_days} calendar days)")
        print(f"  Phase 1 data:     {days_found} days with data")

        if days_found >= min_baseline_days:
            baseline_end = phase1_end
            baseline_data = phase1_data
            expanded = False

        else:
            print(f"  Phase 1 insufficient ({days_found} < "
                  f"{min_baseline_days}) — expanding baseline...")

            current_day = phase1_end + timedelta(days=1)
            baseline_data = dict(phase1_data)
            baseline_end = phase1_end
            expanded = True

            while current_day < clinic_date:
                date_str = current_day.strftime("%Y-%m-%d")

                if date_str in available_dates:
                    baseline_data[date_str] = all_daily_paths[date_str]

                baseline_end = current_day

                if len(baseline_data) >= min_baseline_days:
                    break

                current_day += timedelta(days=1)

            if len(baseline_data) < min_baseline_days:
                print(f"   Only {len(baseline_data)} baseline days found "
                      f"even after expansion — skipping")
                continue

            expanded_calendar = (baseline_end - baseline_start).days + 1
            print(f"  Phase 2 expanded: {phase1_end.date()} → "
                  f"{baseline_end.date()} "
                  f"({expanded_calendar} total calendar days)")
            print(f"  Phase 2 data:     "
                  f"{len(baseline_data)} days with data")

        total_calendar = (baseline_end - baseline_start).days + 1
        print(f"  Baseline final:   {baseline_start.date()} → "
              f"{baseline_end.date()} "
              f"({len(baseline_data)} data days in "
              f"{total_calendar} calendar days"
              f"{' [expanded]' if expanded else ''})")

        window_start = baseline_end + timedelta(days=1)
        window_end = clinic_date
        window_size = (window_end - window_start).days

        print(f"  Window:           {window_start.date()} → "
              f"{window_end.date()} ({window_size} days)")

        if window_size <= 0:
            print(f"   Window size {window_size} — baseline reaches "
                  f"clinic date — skipping")
            continue

        if window_start >= clinic_date:
            print(f"   Window start ({window_start.date()}) >= "
                  f"clinic date ({clinic_date.date()}) — skipping")
            continue

        window_data = {
            date_str: all_daily_paths[date_str]
            for date_str in available_dates
            if window_start <= datetime.strptime(
                date_str, "%Y-%m-%d"
            ) <= window_end
        }

        print(f"  Window data:      {len(window_data)} days with data")

        if len(window_data) == 0:
            print(f"    No window data — skipping")
            continue

        window_coverage = len(window_data) / window_size
        coverage = len(baseline_data) / baseline_days
        min_coverage = 0.0

        if window_coverage < min_coverage:
            print(f"  Window coverage too low: "
                  f"{len(window_data)}/{window_size} days "
                  f"({window_coverage*100:.0f}%) "
                  f"< {min_coverage*100:.0f}% minimum — skipping")
            continue

        if coverage >= 0.8:
            quality = 'good'
        elif coverage >= 0.5:
            quality = 'partial'
        else:
            quality = 'poor'

        print(f"  Baseline quality: {quality} "
              f"({len(baseline_data)}/{baseline_days} requested, "
              f"{coverage*100:.0f}%)")

        final[clinic_date] = {
            'baseline_start': baseline_start,
            'baseline_end': baseline_end,
            'baseline_expanded': expanded,
            'baseline_calendar_days': total_calendar,
            'window_start': window_start,
            'window_end': window_end,
            'window_size': window_size,
            'baseline_data': baseline_data,
            'data': window_data,
            'baseline_days_available': len(baseline_data),
            'baseline_days_requested': baseline_days,
            'baseline_coverage': coverage,
            'baseline_quality': quality
        }

    print(f"\n{'='*50}")
    print(f"SUMMARY: {len(final)} clinic windows built")
    print(f"{'='*50}")

    for clinic_date, val in final.items():
        expanded_str = ' [expanded]' if val['baseline_expanded'] else ''
        print(f"\n  Clinic {clinic_date.date()}:")
        print(f"    Baseline: {val['baseline_start'].date()} → "
              f"{val['baseline_end'].date()} "
              f"({val['baseline_days_available']} data days in "
              f"{val['baseline_calendar_days']} calendar days"
              f"{expanded_str})")
        print(f"    Window:   {val['window_start'].date()} → "
              f"{val['window_end'].date()} "
              f"({val['window_size']} calendar days, "
              f"{len(val['data'])} data days)")
        print(f"    Quality:  {val['baseline_quality']} "
              f"({val['baseline_coverage']*100:.0f}%)")

    return final


def subsample_day(paths, n):
    if len(paths) <= n:
        return paths
    idx = np.random.choice(len(paths), n, replace=False)
    return paths[idx]


def compute_mmd_dynamic_windows(all_daily_paths,
                                 final,
                                 stable_dates,
                                 progress_dates,
                                 windows=[1, 3, 7],
                                 dyadic_order=0,
                                 max_mmd=1.0,
                                 max_paths=1000):
    result = {
        clinic_date: {n: {} for n in windows}
        for clinic_date in final.keys()
    }

    for clinic_date, val in final.items():

        visit_type = 'Stable' if clinic_date in stable_dates \
            else 'Progression'

        print(f"\n{'='*50}")
        print(f"Clinic: {clinic_date.date()} ({visit_type})")
        print(f"{'='*50}")
        print(f"Baseline: {val['baseline_start'].date()} → "
              f"{val['baseline_end'].date()}")
        print(f"Window:   {val['window_start'].date()} → "
              f"{val['window_end'].date()} ({val['window_size']} days)")

        baseline_paths = []

        for date_str in sorted(val['baseline_data'].keys()):
            paths = val['baseline_data'][date_str]
            if len(paths) == 0:
                continue
            baseline_paths.append(paths)

        if len(baseline_paths) == 0:
            print(f"No baseline paths — skipping")
            continue

        daily_paths_A = np.concatenate(baseline_paths, axis=0)

        norms_A = np.array([
            np.linalg.norm(daily_paths_A[i])
            for i in range(len(daily_paths_A))
        ])
        scale_factor = 1.0 / norms_A.mean()

        print(f"Baseline paths:  {len(daily_paths_A)}")
        print(f"Baseline norm:   {norms_A.mean():.4f}")
        print(f"Scale factor:    {scale_factor:.6f}")

        paths_A_scaled = daily_paths_A * scale_factor
        paths_A_scaled = subsample_day(paths_A_scaled, max_paths)

        sorted_dates = sorted(val['data'].keys())

        if len(sorted_dates) == 0:
            print(f"No window data — skipping")
            continue

        for n in windows:
            print(f"\n  Window size: {n} day(s)")
            n_computed = 0
            window_start_idx = 0

            while window_start_idx < len(sorted_dates):

                actual_end_idx = min(
                    window_start_idx + n - 1,
                    len(sorted_dates) - 1
                )

                window_dates = sorted_dates[
                    window_start_idx:actual_end_idx + 1
                ]

                window_end = datetime.strptime(
                    window_dates[-1], "%Y-%m-%d"
                )

                window_paths = []
                tens_len = max(1, 700 // n)

                for day_str in window_dates:
                    if day_str not in val['data']:
                        continue

                    paths = val['data'][day_str]
                    if len(paths) == 0:
                        continue

                    sampled = subsample_day(paths, tens_len)
                    window_paths.append(sampled)

                if len(window_paths) == 0:
                    window_start_idx += n
                    continue

                daily_paths_B = np.concatenate(window_paths, axis=0)

                if len(daily_paths_B) < 2:
                    window_start_idx += n
                    continue

                paths_B_scaled = daily_paths_B * scale_factor
                paths_B_scaled = subsample_day(paths_B_scaled, max_paths)

                norms_B = np.array([
                    np.linalg.norm(paths_B_scaled[i])
                    for i in range(len(paths_B_scaled))
                ])

                if norms_B.max() > 10.0:
                    print(f"      High norm on {window_end.date()}: "
                          f"{norms_B.max():.4f}")

                try:
                    print(f'Baseline shape: {paths_A_scaled.shape}')
                    print(f'Window shape:   {paths_B_scaled.shape} '
                          f'({window_dates[0]} → {window_dates[-1]})')

                    calc_mmd = abs(
                        pysiglib.sig_mmd(
                            paths_A_scaled,
                            paths_B_scaled,
                            dyadic_order=dyadic_order,
                            n_jobs=-1,
                            time_aug=True
                        ).item()
                    )
                    print(calc_mmd)

                    if (calc_mmd > max_mmd or
                            np.isnan(calc_mmd) or
                            np.isinf(calc_mmd)):
                        print(f"      Invalid MMD {window_end.date()}: "
                              f"{calc_mmd:.4e} → NaN")
                        calc_mmd = np.nan

                except Exception as e:
                    print(f"    MMD failed {window_end.date()}: {e}")
                    calc_mmd = np.nan

                date_key = window_end.strftime("%Y-%m-%d")
                result[clinic_date][n][date_key] = calc_mmd
                n_computed += 1

                window_start_idx += n

            print(f"  Computed {n_computed} MMD values")

    return result


def save_mmd_results(result, output_dir, patient_id):
    os.makedirs(output_dir, exist_ok=True)

    serialisable = {}
    for clinic_date, windows in result.items():
        clinic_str = clinic_date.strftime("%Y-%m-%d")
        serialisable[clinic_str] = {}
        for n, mmds in windows.items():
            serialisable[clinic_str][str(n)] = mmds

    out_path = f'{output_dir}/mmd_results.json'
    with open(out_path, 'w') as f:
        json.dump(serialisable, f, indent=2)

    print(f"Saved MMD results: {out_path}")
    return out_path


def load_mmd_results(json_path, clinic_dates):
    with open(json_path, 'r') as f:
        raw = json.load(f)

    result = {}
    for clinic_str, windows in raw.items():
        clinic_date = datetime.strptime(clinic_str, "%Y-%m-%d")
        result[clinic_date] = {}
        for n_str, mmds in windows.items():
            result[clinic_date][int(n_str)] = {
                date_str: float(v) if v is not None else np.nan
                for date_str, v in mmds.items()
            }

    return result


def find_clinic_dates(folder, metadata_path):
    # PLACEHOLDER: `metadata_path` should point to your patient/clinic
    # metadata spreadsheet, containing at least the columns below.
    patient_df = pd.read_excel(
        metadata_path,
        usecols=['BW_Number', 'PD_1', 'PD_2', 'PD_3',
                 'Stable_1', 'Stable_2', 'Stable_3']
    )
    date_cols = ['PD_1', 'PD_2', 'PD_3', 'Stable_1', 'Stable_2', 'Stable_3']
    patient_df[date_cols] = patient_df[date_cols].apply(
        lambda s: pd.to_datetime(s, format='%d-%m-%Y', errors='coerce')
    )
    bw_num = folder.split()[0].replace('-', '_')
    prow = patient_df.loc[
        patient_df['BW_Number'] == bw_num, date_cols[:3]
    ].iloc[0]
    progress_dates = prow[prow.notna()].tolist()
    srow = patient_df.loc[
        patient_df['BW_Number'] == bw_num, date_cols[3:]
    ].iloc[0]
    stable_dates = srow[srow.notna()].tolist()
    return progress_dates, stable_dates


def get_visit_color(clinic_date, stable_dates):
    return 'green' if clinic_date in stable_dates else 'red'


def get_visit_label(clinic_date, stable_dates):
    return 'Stable' if clinic_date in stable_dates else 'Progression'


def smooth_series(dates, values, window=3):
    series = pd.Series(values, index=pd.to_datetime(dates)).sort_index()
    smoothed = series.rolling(
        window=window, center=True, min_periods=1
    ).mean()
    return smoothed.index.tolist(), smoothed.values


def parse_result_dates(mmd_dict):
    items = sorted(mmd_dict.items())
    dates = [datetime.strptime(d, "%Y-%m-%d")
             for d, v in items if v is not None and not np.isnan(v)]
    values = [v for d, v in items if v is not None and not np.isnan(v)]
    return dates, values


def plot_mmd_absolute(result, final, stable_dates,
                       progress_dates, windows,
                       smooth_window=3,
                       save_dir='.'):
    fig, axes = plt.subplots(
        len(windows), 1,
        figsize=(16, 4 * len(windows)),
        sharex=True
    )

    if len(windows) == 1:
        axes = [axes]

    for ax, n in zip(axes, windows):

        for clinic_date, val in result.items():
            if n not in val or len(val[n]) == 0:
                continue

            dates, values = parse_result_dates(val[n])
            if len(dates) == 0:
                continue

            color = get_visit_color(clinic_date, stable_dates)

            ax.plot(dates, values,
                    color=color, linewidth=0.8,
                    alpha=0.3, linestyle='--',
                    marker='o', markersize=3)

            s_dates, s_values = smooth_series(
                dates, values, window=smooth_window
            )
            ax.plot(s_dates, s_values,
                    color=color, linewidth=2.0, alpha=0.9)

        for clinic_date in result.keys():
            color = get_visit_color(clinic_date, stable_dates)
            label = get_visit_label(clinic_date, stable_dates)
            ax.axvline(
                clinic_date,
                color=color, linestyle='--',
                linewidth=1.2, alpha=0.7
            )
            ax.text(
                clinic_date, ax.get_ylim()[1],
                label[0],
                color=color, fontsize=8,
                ha='center', va='bottom'
            )

        ax.set_ylabel(f'MMD ({n}-day window)', fontsize=11)
        ax.grid(True, alpha=0.3)

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    axes[-1].set_xlabel('Date', fontsize=12)

    legend_elements = [
        mpatches.Patch(color='green', label='Stable visit'),
        mpatches.Patch(color='red', label='Progression visit'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.suptitle(
        'MMD Over Time — All Clinic Windows\n'
        f'({smooth_window}-day smoothing)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    save_path = f'{save_dir}/mmd_absolute_time.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_mmd_relative(result, final, stable_dates,
                       progress_dates, windows,
                       smooth_window=3,
                       save_dir='.'):
    fig, axes = plt.subplots(
        len(windows), 1,
        figsize=(14, 4 * len(windows)),
        sharex=False
    )

    if len(windows) == 1:
        axes = [axes]

    for ax, n in zip(axes, windows):

        for clinic_date, val in result.items():
            if n not in val or len(val[n]) == 0:
                continue

            dates, values = parse_result_dates(val[n])
            if len(dates) == 0:
                continue

            rel_days = np.array([(d - clinic_date).days for d in dates])

            color = get_visit_color(clinic_date, stable_dates)

            sort_idx = np.argsort(rel_days)
            rel_days = rel_days[sort_idx]
            values = np.array(values)[sort_idx]

            ax.plot(rel_days, values,
                    color=color, linewidth=0.8,
                    alpha=0.3, linestyle='--',
                    marker='o', markersize=3)

            series = pd.Series(values, index=rel_days)
            smoothed = series.rolling(
                window=smooth_window, center=True, min_periods=1
            ).mean()
            ax.plot(smoothed.index, smoothed.values,
                    color=color, linewidth=2.0, alpha=0.9)

        ax.axvline(0, color='black', linestyle='--',
                   linewidth=1.5, alpha=0.8, label='Clinic visit')

        ax.set_ylabel(f'MMD ({n}-day window)', fontsize=11)
        ax.set_xlabel('Days relative to clinic visit', fontsize=11)
        ax.grid(True, alpha=0.3)

    legend_elements = [
        mpatches.Patch(color='green', label='Stable'),
        mpatches.Patch(color='red', label='Progression'),
        plt.Line2D([0], [0], color='black',
                   linestyle='--', label='Clinic date')
    ]
    axes[0].legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.suptitle(
        'MMD Relative to Clinic Date\n'
        f'({smooth_window}-day smoothing)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    save_path = f'{save_dir}/mmd_relative_time.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_mmd_per_visit(result, final, stable_dates,
                        progress_dates, windows,
                        smooth_window=3,
                        save_dir='.'):
    for clinic_date, val in result.items():

        visit_type = get_visit_label(clinic_date, stable_dates)
        color = get_visit_color(clinic_date, stable_dates)

        fig, axes = plt.subplots(
            len(windows), 1,
            figsize=(14, 4 * len(windows)),
            sharex=False
        )

        if len(windows) == 1:
            axes = [axes]

        for ax, n in zip(axes, windows):

            if n not in val or len(val[n]) == 0:
                ax.set_title(f'{n}-day window: no data')
                continue

            dates, values = parse_result_dates(val[n])
            if len(dates) == 0:
                continue

            rel_days = np.array([(d - clinic_date).days for d in dates])

            sort_idx = np.argsort(rel_days)
            rel_days = rel_days[sort_idx]
            values = np.array(values)[sort_idx]

            ax.plot(rel_days, values,
                    color=color, linewidth=0.8,
                    alpha=0.3, linestyle='--',
                    marker='o', markersize=4)

            series = pd.Series(values, index=rel_days)
            smoothed = series.rolling(
                window=smooth_window, center=True, min_periods=1
            ).mean()
            ax.plot(smoothed.index, smoothed.values,
                    color=color, linewidth=2.5, alpha=0.9,
                    label=f'{n}-day MMD (smoothed)')

            ax.axvline(0, color='black', linestyle='--',
                       linewidth=1.5, label='Clinic visit')

            baseline_start = final[clinic_date]['baseline_start']
            baseline_end = final[clinic_date]['baseline_end']
            bl_start_rel = (baseline_start - clinic_date).days
            bl_end_rel = (baseline_end - clinic_date).days

            ax.axvspan(bl_start_rel, bl_end_rel,
                       alpha=0.1, color='blue', label='Baseline period')

            mean_mmd = np.mean(values)
            max_mmd = np.max(values)
            max_day = rel_days[np.argmax(values)]

            ax.axhline(mean_mmd, color=color,
                       linestyle=':', linewidth=1.0,
                       alpha=0.5, label=f'Mean={mean_mmd:.4e}')

            ax.set_ylabel(f'MMD', fontsize=10)
            ax.set_xlabel('Days relative to clinic', fontsize=10)
            ax.set_title(
                f'{n}-day window | '
                f'Mean={mean_mmd:.4e} | '
                f'Max={max_mmd:.4e} at day {max_day}',
                fontsize=10
            )
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f'Clinic: {clinic_date.date()} — {visit_type}\n'
            f'Baseline: '
            f'{final[clinic_date]["baseline_start"].date()} → '
            f'{final[clinic_date]["baseline_end"].date()} | '
            f'Window: '
            f'{final[clinic_date]["window_start"].date()} → '
            f'{final[clinic_date]["window_end"].date()}',
            fontsize=12, fontweight='bold',
            color=color
        )
        plt.tight_layout()

        save_path = (f'{save_dir}/mmd_visit_'
                     f'{clinic_date.strftime("%Y%m%d")}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def plot_mmd_distribution(result, stable_dates,
                           progress_dates, windows,
                           save_dir='.'):
    fig, axes = plt.subplots(
        1, len(windows),
        figsize=(6 * len(windows), 6)
    )

    if len(windows) == 1:
        axes = [axes]

    for ax, n in zip(axes, windows):

        stable_mmds = []
        progression_mmds = []

        for clinic_date, val in result.items():
            if n not in val or len(val[n]) == 0:
                continue

            _, values = parse_result_dates(val[n])
            if len(values) == 0:
                continue

            if clinic_date in stable_dates:
                stable_mmds.extend(values)
            else:
                progression_mmds.extend(values)

        data = [stable_mmds, progression_mmds]
        labels = ['Stable', 'Progression']
        colors = ['green', 'red']

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for i, (vals, color) in enumerate(zip(data, colors), start=1):
            jitter = np.random.normal(0, 0.05, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals, color=color,
                alpha=0.4, s=20, zorder=5
            )

        if len(stable_mmds) > 0 and len(progression_mmds) > 0:
            from scipy.stats import mannwhitneyu
            stat, p = mannwhitneyu(
                progression_mmds, stable_mmds, alternative='greater'
            )
            ax.set_title(f'{n}-day window\np={p:.4f} ', fontsize=11)
        else:
            ax.set_title(f'{n}-day window', fontsize=11)

        ax.set_ylabel('MMD', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        'MMD Distribution — Stable vs Progression',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    save_path = f'{save_dir}/mmd_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def main():
    folder = sys.argv[1]
    print(f'Processing Folder: {folder}\n')

    base_dir = f"{CONFIG['base_path']}/{folder}"
    with open(f"{base_dir}/dirs.txt", 'r') as f:
        dir_lines = f.readlines()

    json_file = f"processed_paths_{CONFIG['path_samples']}.json"
    file_list = [
        f"{base_dir}/{line.strip().split('/')[-1]}/{json_file}"
        for line in dir_lines
    ]

    if CONFIG['process'] == "RAW":
        all_daily_paths = load_and_merge_jsons(file_list=file_list)
    else:
        all_daily_paths = load_and_merge_jsons_enmo(file_list=file_list)

    progress_dates, stable_dates = find_clinic_dates(
        folder, CONFIG['patient_metadata_path']
    )
    clinic_dates = stable_dates + progress_dates

    final = build_final_dict(
        all_daily_paths=all_daily_paths,
        clinic_dates=clinic_dates,
        baseline_days=CONFIG['baseline_days']
    )

    result = compute_mmd_dynamic_windows(
        all_daily_paths=all_daily_paths,
        final=final,
        stable_dates=stable_dates,
        progress_dates=progress_dates,
        windows=CONFIG['windows'],
        dyadic_order=CONFIG['dyadic_order'],
        max_mmd=CONFIG['max_mmd'],
        max_paths=CONFIG['max_paths']
    )

    print(f"\n{'='*50}")
    print(f"MMD SUMMARY")
    print(f"{'='*50}")

    for clinic_date, windows in result.items():
        visit_type = 'Stable' if clinic_date in stable_dates \
            else 'Progression'
        print(f"\n{clinic_date.date()} ({visit_type}):")

        for n, mmds in windows.items():
            valid = [v for v in mmds.values() if not np.isnan(v)]
            if len(valid) == 0:
                print(f"  {n}-day window: no valid MMDs")
                continue
            print(f"  {n}-day window: "
                  f"{len(valid)} values | "
                  f"mean={np.mean(valid):.4e} | "
                  f"min={np.min(valid):.4e} | "
                  f"max={np.max(valid):.4e}")

    save_dir = (f"{CONFIG['target_path']}/{folder}/"
                f"{CONFIG['path_samples']}/{CONFIG['process']}/"
                f"TimeAug/{CONFIG['baseline_days']}baseline_0.0")
    os.makedirs(save_dir, exist_ok=True)

    save_mmd_results(
        result=result,
        output_dir=save_dir,
        patient_id=folder
    )

    plot_mmd_absolute(
        result=result,
        final=final,
        stable_dates=stable_dates,
        progress_dates=progress_dates,
        windows=CONFIG['windows'],
        smooth_window=3,
        save_dir=save_dir
    )

    plot_mmd_relative(
        result=result,
        final=final,
        stable_dates=stable_dates,
        progress_dates=progress_dates,
        windows=CONFIG['windows'],
        smooth_window=3,
        save_dir=save_dir
    )

    plot_mmd_per_visit(
        result=result,
        final=final,
        stable_dates=stable_dates,
        progress_dates=progress_dates,
        windows=CONFIG['windows'],
        smooth_window=3,
        save_dir=save_dir
    )

    plot_mmd_distribution(
        result=result,
        stable_dates=stable_dates,
        progress_dates=progress_dates,
        windows=CONFIG['windows'],
        save_dir=save_dir
    )


if __name__ == "__main__":
    main()
