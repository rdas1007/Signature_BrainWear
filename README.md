# Movement Signature / MMD Analysis Pipeline

This repository contains three scripts that form an end-to-end pipeline for
turning raw 25Hz triaxial accelerometer data into a clinical stable-vs-progression
classifier based on signature Maximum Mean Discrepancy (MMD).

```
25HzJson.py          25Hzmmd.py                  mmd_analysis.py
raw CSVs   ─────▶   per-day path JSON  ─────▶   MMD results JSON  ─────▶   LOPO
(25Hz + labels)      (walking "paths")           (baseline vs.            classifier
                                                    rolling windows)        + report
```

Each script is self-contained and documented with a top-of-file docstring
describing its purpose, required inputs, and every function. Sensitive/
site-specific values (file paths, participant IDs, credentials) have been
removed and replaced with `TODO` placeholders or externalized into a
`CONFIG` dict that the caller must supply.

---

## 1. `25HzJson.py`

**Purpose:** Builds a per-day dataset of "walking path" arrays from raw 25Hz
triaxial accelerometer data and 1Hz activity labels, and saves the result as
a JSON file.

**Pipeline:**
1. Load raw 25Hz signal and 1Hz label CSVs for a subject/session.
2. Align 1Hz labels to 25Hz samples (floor timestamps to the nearest second).
3. For each calendar date: extract continuous walking bursts, bandpass
   filter, downsample, standard-scale, and window into fixed-length "paths."
4. Keep only days with enough valid paths; write all days to a JSON file.

**Inputs:**
- CLI args: `input_folder`, `input_sub_folder` (folder/session identifiers).
- A `CONFIG` dict must be defined by the caller before this script runs (see
  the module docstring for the full list of required keys — paths, sample
  rates, filter cutoffs, path/burst length thresholds, etc.).
- Raw CSV (`<session>.csv`) and label CSV (`<session>-timeSeries.csv`) under
  `CONFIG['base_path']`.

**Output:** `processed_paths_<label_col>_<path_samples>.json` under
`CONFIG['target_path']`.

---

## 2. `25Hzmmd.py`

**Purpose:** Computes and visualizes signature-based MMD trends between a
per-patient baseline period and rolling windows of daily movement paths,
centered on clinic visit dates (stable vs. progression).

**Pipeline:**
1. Load and merge per-recording path JSON files (output of `25HzJson.py`)
   for a patient, optionally converting to ENMO + time representation.
2. For each clinic visit date, build a baseline window and one or more
   observation windows of daily paths.
3. Compute signature MMD between baseline and each window.
4. Save MMD results to JSON and generate plots (absolute time, relative
   time, per-visit, and stable-vs-progression distribution).

**Inputs:**
- CLI arg: `sys.argv[1]` — the patient/session folder to process.
- A `CONFIG` dict must be defined by the caller (paths, `process` mode,
  `path_samples`, `baseline_days`, `windows`, `dyadic_order`, `max_mmd`,
  `max_paths`, and `patient_metadata_path` — see module docstring for the
  full list).
- Per-recording path JSON files produced by `25HzJson.py`.
- A patient metadata spreadsheet (stable/progression clinic visit dates).

**Output:** MMD results JSON and diagnostic plots under `CONFIG['target_path']`.

---

## 3. `mmd_analysis.py`

**Purpose:** Aggregates per-patient MMD results (window = 1, ENMO pipeline)
into a per-clinic-visit dataset, converts each value into a ratio relative to
that patient's own stable baseline, and evaluates a Youden's-J threshold
classifier using Leave-One-Participant-Out (LOPO) cross-validation with
bootstrap confidence intervals.

**Pipeline:**
1. Load patient metadata (stable/progression clinic visit dates).
2. Load each patient's `mmd_results.json` (output of `25Hzmmd.py`) and
   aggregate MMD values per clinic visit.
3. Compute each patient's stable-visit baseline and transform every visit's
   MMD value into a ratio relative to that baseline.
4. Run LOPO cross-validation: per-fold Youden's-J threshold selection,
   pooled sensitivity/specificity/PPV/NPV/F1/AUC with bootstrap CIs, and
   summary/diagnostic plots.

**Inputs (all marked `TODO` in the script — fill in before running):**
- `PATIENT_METADATA_PATH` — Excel file with columns `BW_Number`, `PD_1`–`PD_3`
  (progression visit dates), `Stable_1`–`Stable_3` (stable visit dates).
- `BASE_RESULTS_DIR` — base directory containing each patient's
  `mmd_results.json` (output of `25Hzmmd.py`), under `ENMO_RESULT_SUBPATH`.
- `PATIENT_IDS` — list of patient identifiers matching folder names under
  `BASE_RESULTS_DIR`.

**Note:** `lopo_window_threshold_report()` calls `build_window_ratio_dataframe()`,
an external helper **not defined in this file**. It must be defined or
imported elsewhere before running this script.

**Output:** Console summary tables (LOPO metrics with 95% CIs), plus
matplotlib/seaborn plots (ratio distributions, threshold stability,
confusion matrix).

---

## Requirements

- `pandas`, `numpy`, `scipy`
- `pysiglib` (signature MMD computation, used by `25Hzmmd.py`)
- `matplotlib`, `seaborn`
- `scikit-learn` (`StandardScaler`, `roc_curve`, `roc_auc_score`,
  `ConfusionMatrixDisplay`)

## Setup

1. Fill in the `TODO` placeholders and `CONFIG` keys in each script with
   your site-specific paths and parameters (see each script's docstring for
   the full list).
2. Run `25HzJson.py` per subject/session to produce path JSON files.
3. Run `25Hzmmd.py` per patient to produce `mmd_results.json`.
4. Run `mmd_analysis.py` (after supplying `build_window_ratio_dataframe`) to
   produce the LOPO classification report.
