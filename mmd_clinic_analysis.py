"""
MMD Clinic Visit Analysis (window = 1)

This script builds a per-patient, per-clinic-visit dataset of Maximum Mean
Discrepancy (MMD) values (window size 1 only) computed from accelerometer
"ENMO" signal processing, labels each visit as "stable" or "progression"
using dates from a patient metadata spreadsheet, converts each patient's
MMD values into a ratio relative to that patient's own stable baseline,
and evaluates a Youden's-J threshold classifier using Leave-One-Participant-
Out (LOPO) cross-validation with participant-level bootstrap confidence
intervals.

Required inputs (see TODO placeholders below):
  - PATIENT_METADATA_PATH: Excel file with columns BW_Number, PD_1-3
    (progression visit dates) and Stable_1-3 (stable visit dates).
  - BASE_RESULTS_DIR: base directory containing each patient's
    mmd_results.json file (ENMO pipeline), following ENMO_RESULT_SUBPATH.
  - PATIENT_IDS: list of patient identifiers matching folder names under
    BASE_RESULTS_DIR (e.g. "BW-003 - LP").

Note: lopo_window_threshold_report() calls build_window_ratio_dataframe(),
which is an external helper (not defined in this file) expected to turn
(X, y, meta) into a long-format dataframe containing a "Personalised
SigMMD Ratio" column and the participant identifier column. It must be
defined or imported elsewhere before this script is run.

Functions:
  - load_patient_metadata(path): Loads the patient metadata Excel file and
    parses the progression/stable visit date columns.
  - find_clinic_dates(patient_df, folder): Looks up a single patient's
    progression and stable clinic visit dates from the metadata dataframe.
  - load_result_json(json_path): Loads a patient's mmd_results.json file
    and returns, per clinic date, the window=1 MMD values keyed by
    reference date.
  - build_all_patient_per_clinic(result_jsons, stable_dates_all,
    progress_dates_all, window): Builds a nested dict of
    {patient_id: {clinic_date: {'visit_type', window: {mean, median,
    n_vals}}}} by aggregating each patient's MMD values per clinic visit.
  - flatten_mmd_dict(mmd_dict): Converts the nested per-clinic MMD
    dictionary into a long-format dataframe with one row per
    (patient, clinic date, window).
  - compute_patient_window_baseline(patient_df, feature_col,
    baseline_method): Computes a single patient's stable-visit MMD
    baseline for window=1.
  - transform_relative_to_baseline(value, baseline, mode, eps): Converts a
    raw MMD value into a ratio, difference, or log-ratio relative to a
    baseline value.
  - build_personalized_mmd_dataset(mmd_dict, feature_col, baseline_method,
    transform_mode, min_stable_per_window, include_n_vals): Builds the
    final (X, y, meta, df) modelling dataset by transforming each visit's
    MMD value relative to that patient's own stable baseline.
  - lopo_window_threshold_report(X, y, meta, participant_col, plot,
    n_bootstrap, ci_level, random_state): Runs Leave-One-Participant-Out
    Youden's-J threshold selection and reports pooled sensitivity,
    specificity, PPV, NPV, F1 and AUC with participant-level bootstrap
    confidence intervals, optionally plotting threshold stability and a
    pooled confusion matrix.
"""

import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

PATIENT_METADATA_PATH = "TODO: add path to the patient metadata Excel file (columns: BW_Number, PD_1, PD_2, PD_3, Stable_1, Stable_2, Stable_3)"
BASE_RESULTS_DIR = "TODO: add base directory containing each patient's ENMO mmd_results.json file"
ENMO_RESULT_SUBPATH = "/75/ENMO/14baseline_0.5/mmd_results.json"
PATIENT_IDS = []  # TODO: add list of patient identifiers, e.g. ["BW-003 - LP", "BW-005 - GP", ...]
WINDOW = 1


def load_patient_metadata(path):
    date_cols = ["PD_1", "PD_2", "PD_3", "Stable_1", "Stable_2", "Stable_3"]
    patient_df = pd.read_excel(path, usecols=["BW_Number"] + date_cols)
    patient_df[date_cols] = patient_df[date_cols].apply(
        lambda s: pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    )
    return patient_df


def find_clinic_dates(patient_df, folder):
    date_cols = ["PD_1", "PD_2", "PD_3", "Stable_1", "Stable_2", "Stable_3"]
    bw_num = folder.split()[0].replace("-", "_")
    row = patient_df.loc[patient_df["BW_Number"] == bw_num, date_cols]
    if row.empty:
        return [], []
    row = row.iloc[0]
    progress_dates = row[date_cols[:3]][row[date_cols[:3]].notna()].tolist()
    stable_dates = row[date_cols[3:]][row[date_cols[3:]].notna()].tolist()
    return progress_dates, stable_dates


def load_result_json(json_path):
    with open(json_path, "r") as f:
        raw = json.load(f)

    result = {}
    for clinic_str, windows in raw.items():
        mmds = windows.get(str(WINDOW))
        if not isinstance(mmds, dict):
            continue
        clinic_date = datetime.strptime(clinic_str, "%Y-%m-%d")
        result[clinic_date] = {
            WINDOW: {
                date_str: float(v)
                if v is not None and str(v) not in ("nan", "inf", "-inf")
                else np.nan
                for date_str, v in mmds.items()
            }
        }

    return result


def build_all_patient_per_clinic(result_jsons, stable_dates_all, progress_dates_all, window=WINDOW):
    all_patient_per_clinic = {}

    for patient_id, json_path in result_jsons.items():
        if not os.path.exists(json_path):
            print(f"Not found: {json_path}")
            continue

        stable_dates = stable_dates_all.get(patient_id, [])
        progress_dates = progress_dates_all.get(patient_id, [])

        result = load_result_json(json_path)
        if len(result) == 0:
            print(f"Empty result: {patient_id}")
            continue

        patient_dict = {}

        for clinic_date, clinic_mmds in sorted(result.items()):
            clinic_str = clinic_date.strftime("%Y-%m-%d")

            if clinic_date in stable_dates:
                visit_type = "stable"
            elif clinic_date in progress_dates:
                visit_type = "progression"
            else:
                continue

            if window not in clinic_mmds:
                continue

            mmd_vals = [
                v for v in clinic_mmds[window].values()
                if v is not None and not np.isnan(v) and not np.isinf(v)
            ]
            if len(mmd_vals) == 0:
                continue

            patient_dict[clinic_str] = {
                "visit_type": visit_type,
                window: {
                    "mean": float(np.mean(mmd_vals)),
                    "median": float(np.median(mmd_vals)),
                    "n_vals": len(mmd_vals),
                },
            }

        if len(patient_dict) > 0:
            all_patient_per_clinic[patient_id] = patient_dict

    n_visits = sum(len(v) for v in all_patient_per_clinic.values())
    print(f"all_patient_per_clinic: {len(all_patient_per_clinic)} patients, {n_visits} visits")

    return all_patient_per_clinic


def flatten_mmd_dict(mmd_dict):
    rows = []

    for patient_id, visits in mmd_dict.items():
        for clinic_date, visit_data in visits.items():
            visit_type = visit_data.get("visit_type", None)
            if visit_type not in ["stable", "progression"]:
                continue

            label = 0 if visit_type == "stable" else 1

            for key, val in visit_data.items():
                if key == "visit_type":
                    continue
                if not isinstance(key, (int, np.integer)):
                    continue

                mean_mmd = val.get("mean", np.nan)
                median_mmd = val.get("median", np.nan)
                n_vals = val.get("n_vals", np.nan)

                if not np.isfinite(mean_mmd) or not np.isfinite(median_mmd):
                    continue

                rows.append({
                    "patient_id": patient_id,
                    "clinic_date": clinic_date,
                    "visit_type": visit_type,
                    "label": label,
                    "window": key,
                    "mean_mmd": float(mean_mmd),
                    "median_mmd": float(median_mmd),
                    "n_vals": int(n_vals) if np.isfinite(n_vals) else np.nan,
                })

    return pd.DataFrame(rows)


def compute_patient_window_baseline(patient_df, feature_col="mean_mmd", baseline_method="mean"):
    stable_df = patient_df[patient_df["label"] == 0]
    baseline_dict = {}

    for window, wdf in stable_df.groupby("window"):
        vals = wdf[feature_col].values
        if len(vals) == 0:
            continue

        if baseline_method == "mean":
            baseline = np.mean(vals)
        elif baseline_method == "median":
            baseline = np.median(vals)
        else:
            raise ValueError("baseline_method must be 'mean' or 'median'")

        if np.isfinite(baseline):
            baseline_dict[window] = float(baseline)

    return baseline_dict


def transform_relative_to_baseline(value, baseline, mode="ratio", eps=1e-8):
    if baseline is None or not np.isfinite(baseline):
        return np.nan

    if mode == "ratio":
        return value / (baseline + eps)
    elif mode == "difference":
        return value - baseline
    elif mode == "log_ratio":
        return np.log((value + eps) / (baseline + eps))
    else:
        raise ValueError("mode must be 'ratio', 'difference', or 'log_ratio'")


def build_personalized_mmd_dataset(
    mmd_dict,
    feature_col="mean_mmd",
    baseline_method="mean",
    transform_mode="ratio",
    min_stable_per_window=1,
    include_n_vals=False,
):
    df = flatten_mmd_dict(mmd_dict)

    X = []
    y = []
    meta = []

    for patient_id, pdf in df.groupby("patient_id"):
        baseline_dict = compute_patient_window_baseline(
            pdf, feature_col=feature_col, baseline_method=baseline_method
        )

        stable_counts = pdf[pdf["label"] == 0].groupby("window").size().to_dict()

        for _, row in pdf.iterrows():
            window = row["window"]

            if stable_counts.get(window, 0) < min_stable_per_window:
                continue

            baseline = baseline_dict.get(window, None)
            if baseline is None:
                continue

            transformed = transform_relative_to_baseline(
                row[feature_col], baseline, mode=transform_mode
            )
            if not np.isfinite(transformed):
                continue

            if include_n_vals:
                feat = np.array([transformed, row["n_vals"]], dtype=np.float64)
            else:
                feat = np.array([transformed], dtype=np.float64)

            X.append(feat)
            y.append(row["label"])
            meta.append({
                "patient_id": row["patient_id"],
                "clinic_date": row["clinic_date"],
                "visit_type": row["visit_type"],
                "window": row["window"],
                "raw_value": row[feature_col],
                "baseline_value": baseline,
                "transformed_value": transformed,
                "n_vals": row["n_vals"],
            })

    return np.asarray(X), np.asarray(y), meta, df


def lopo_window_threshold_report(
    X,
    y,
    meta,
    participant_col="patient_id",
    plot=True,
    n_bootstrap=2000,
    ci_level=0.95,
    random_state=None,
):
    rng = np.random.default_rng(random_state)
    alpha = 1.0 - ci_level
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)

    def _confusion_metrics(true_arr, pred_arr):
        tp = int(np.sum((pred_arr == 1) & (true_arr == 1)))
        fn = int(np.sum((pred_arr == 0) & (true_arr == 1)))
        fp = int(np.sum((pred_arr == 1) & (true_arr == 0)))
        tn = int(np.sum((pred_arr == 0) & (true_arr == 0)))
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv_ = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv_ = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        f1_ = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
        return tp, fp, fn, tn, sens, spec, ppv_, npv_, f1_

    def _safe_auc(true_arr, score_arr):
        if len(np.unique(true_arr)) < 2:
            return np.nan
        try:
            return roc_auc_score(true_arr, score_arr)
        except Exception:
            return np.nan

    def _percentile_ci(values):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return np.nan, np.nan
        return float(np.percentile(arr, lo_q)), float(np.percentile(arr, hi_q))

    def _bootstrap_ci(per_participant_index, n_boot):
        stat_names = ["pooled_auc", "sensitivity", "specificity", "ppv", "npv", "f1"]
        boots = {name: [] for name in stat_names}

        uniq_participants = np.array(list(per_participant_index.keys()))
        if n_boot <= 0 or len(uniq_participants) < 2:
            return {name: (np.nan, np.nan) for name in stat_names}

        for _ in range(n_boot):
            sampled = rng.choice(uniq_participants, size=len(uniq_participants), replace=True)
            idxs = np.concatenate([per_participant_index[p] for p in sampled])

            bt_true = oof_true[idxs]
            bt_pred = oof_pred[idxs]
            bt_score = oof_score[idxs]

            _, _, _, _, sens, spec, ppv_, npv_, f1_ = _confusion_metrics(bt_true, bt_pred)
            auc_ = _safe_auc(bt_true, bt_score)

            boots["pooled_auc"].append(auc_)
            boots["sensitivity"].append(sens)
            boots["specificity"].append(spec)
            boots["ppv"].append(ppv_)
            boots["npv"].append(npv_)
            boots["f1"].append(f1_)

        return {name: _percentile_ci(vals) for name, vals in boots.items()}

    def _bootstrap_threshold_ci(fold_thr_by_participant, n_boot):
        uniq_participants = np.array(list(fold_thr_by_participant.keys()))
        if n_boot <= 0 or len(uniq_participants) < 2:
            return np.nan, np.nan
        means = []
        for _ in range(n_boot):
            sampled = rng.choice(uniq_participants, size=len(uniq_participants), replace=True)
            vals = np.array([fold_thr_by_participant[p] for p in sampled], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                means.append(vals.mean())
        return _percentile_ci(means)

    df_plot = build_window_ratio_dataframe(X, y, meta)

    if participant_col not in df_plot.columns:
        raise KeyError(
            f"'{participant_col}' not found in df_plot columns ({list(df_plot.columns)}). "
            f"Pass the correct participant identifier via participant_col=."
        )

    summary_rows = []
    fold_rows = []

    windows = sorted(df_plot["window"].unique())

    for window in windows:
        wdf = df_plot[df_plot["window"] == window].copy()
        participants = wdf[participant_col].unique()

        if len(participants) < 2:
            print(f"[window {window}] fewer than 2 participants, skipping LOPO")
            continue

        oof_true_list, oof_pred_list, oof_score_list, oof_participant_list = [], [], [], []
        fold_thresholds = []
        fold_threshold_by_participant = {}

        for held_out in participants:
            train = wdf[wdf[participant_col] != held_out]
            test = wdf[wdf[participant_col] == held_out]

            y_train = train["label"].values
            score_train = train["Personalised SigMMD Ratio"].values

            try:
                fpr, tpr, roc_thresholds = roc_curve(y_train, score_train)
            except Exception as e:
                print(f"[window {window}] roc_curve failed holding out {held_out}: {e}")
                continue

            j_scores = tpr - fpr
            idx = int(np.argmax(j_scores))
            if not np.isfinite(roc_thresholds[idx]) and len(j_scores) > 1:
                idx = int(np.argmax(j_scores[1:])) + 1
            thr = float(roc_thresholds[idx])
            fold_thresholds.append(thr)
            fold_threshold_by_participant[held_out] = thr

            y_test = test["label"].values
            score_test = test["Personalised SigMMD Ratio"].values
            pred_test = (score_test >= thr).astype(int)

            oof_true_list.extend(y_test.tolist())
            oof_pred_list.extend(pred_test.tolist())
            oof_score_list.extend(score_test.tolist())
            oof_participant_list.extend([held_out] * len(y_test))

            for yt, yp, ys in zip(y_test, pred_test, score_test):
                fold_rows.append({
                    "window": window,
                    "held_out_participant": held_out,
                    "fold_threshold": thr,
                    "label": yt,
                    "score": ys,
                    "pred_at_fold_threshold": yp,
                })

        if len(oof_true_list) == 0:
            print(f"[window {window}] no valid folds, skipping")
            continue

        oof_true = np.array(oof_true_list)
        oof_pred = np.array(oof_pred_list)
        oof_score = np.array(oof_score_list)
        oof_participant = np.array(oof_participant_list)

        per_participant_index = {
            p: np.where(oof_participant == p)[0] for p in np.unique(oof_participant)
        }

        pooled_auc = _safe_auc(oof_true, oof_score)
        tp, fp, fn, tn, sensitivity, specificity, ppv, npv, f1 = _confusion_metrics(oof_true, oof_pred)

        thr_arr = np.array(fold_thresholds)
        thr_arr = thr_arr[np.isfinite(thr_arr)]

        ci = _bootstrap_ci(per_participant_index, n_bootstrap)
        thr_ci_low, thr_ci_high = _bootstrap_threshold_ci(fold_threshold_by_participant, n_bootstrap)

        print("\n" + "=" * 80)
        print(f"WINDOW = {window}  (LOPO, n_participants={len(participants)}, n_valid_folds={len(fold_thresholds)})")
        print("=" * 80)
        print(f"Pooled AUC (reference): {pooled_auc:.4f}  95% CI [{ci['pooled_auc'][0]:.4f}, {ci['pooled_auc'][1]:.4f}]")
        if len(thr_arr):
            print(f"Fold threshold: mean={thr_arr.mean():.4f} 95% CI [{thr_ci_low:.4f}, {thr_ci_high:.4f}] std={thr_arr.std():.4f}")
        print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"Sensitivity: {sensitivity:.4f} [{ci['sensitivity'][0]:.4f}, {ci['sensitivity'][1]:.4f}]")
        print(f"Specificity: {specificity:.4f} [{ci['specificity'][0]:.4f}, {ci['specificity'][1]:.4f}]")
        print(f"PPV: {ppv:.4f} [{ci['ppv'][0]:.4f}, {ci['ppv'][1]:.4f}]")
        print(f"NPV: {npv:.4f} [{ci['npv'][0]:.4f}, {ci['npv'][1]:.4f}]")
        print(f"F1: {f1:.4f} [{ci['f1'][0]:.4f}, {ci['f1'][1]:.4f}]")

        summary_rows.append({
            "window": window,
            "n_participants": len(participants),
            "n_valid_folds": len(fold_thresholds),
            "pooled_auc": pooled_auc,
            "pooled_auc_ci_low": ci["pooled_auc"][0],
            "pooled_auc_ci_high": ci["pooled_auc"][1],
            "threshold_mean": thr_arr.mean() if len(thr_arr) else np.nan,
            "threshold_mean_ci_low": thr_ci_low,
            "threshold_mean_ci_high": thr_ci_high,
            "threshold_std": thr_arr.std() if len(thr_arr) else np.nan,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": sensitivity,
            "sensitivity_ci_low": ci["sensitivity"][0],
            "sensitivity_ci_high": ci["sensitivity"][1],
            "specificity": specificity,
            "specificity_ci_low": ci["specificity"][0],
            "specificity_ci_high": ci["specificity"][1],
            "ppv": ppv,
            "ppv_ci_low": ci["ppv"][0],
            "ppv_ci_high": ci["ppv"][1],
            "npv": npv,
            "npv_ci_low": ci["npv"][0],
            "npv_ci_high": ci["npv"][1],
            "f1": f1,
            "f1_ci_low": ci["f1"][0],
            "f1_ci_high": ci["f1"][1],
        })

        if plot and len(thr_arr):
            fig, ax = plt.subplots(figsize=(3.3, 2.2), layout="constrained")
            ax.hist(thr_arr, bins=min(15, max(3, len(thr_arr))), color="#0072B2", edgecolor="black", linewidth=0.5)
            ax.set_xlabel("Fold-selected threshold (Youden's J)", fontsize=8, labelpad=3)
            ax.set_ylabel("Count", fontsize=8, labelpad=3)
            ax.tick_params(axis="both", which="major", labelsize=7, length=3, width=0.8)
            ax.set_title(
                f"Window {window}: LOPO threshold stability (mean 95% CI [{thr_ci_low:.3f}, {thr_ci_high:.3f}])",
                fontsize=7.5,
            )
            plt.show()

            cm = ConfusionMatrixDisplay.from_predictions(
                oof_true, oof_pred, display_labels=["Stable", "Progressive"], colorbar=False, cmap="Blues"
            )
            cm.ax_.set_title(f"Window {window}: LOPO confusion matrix", fontsize=8.5)
            plt.show()

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)

    print("\n" + "=" * 80)
    print("LOPO WINDOW-SPECIFIC SUMMARY TABLE (with 95% CIs)")
    print("=" * 80)
    if len(summary_df):
        print(summary_df.to_string(index=False))
    else:
        print("(no windows produced valid LOPO results)")

    return summary_df, fold_df, df_plot


if __name__ == "__main__":
    patient_df = load_patient_metadata(PATIENT_METADATA_PATH)

    stable_dates_all = {}
    progress_dates_all = {}
    for pid in PATIENT_IDS:
        progress, stable = find_clinic_dates(patient_df, pid)
        progress_dates_all[pid] = progress
        stable_dates_all[pid] = stable

    result_jsons_enmo = {pid: BASE_RESULTS_DIR + pid + ENMO_RESULT_SUBPATH for pid in PATIENT_IDS}

    mmd_dict_enmo = build_all_patient_per_clinic(
        result_jsons_enmo, stable_dates_all, progress_dates_all, window=WINDOW
    )

    X, y, meta, df_long = build_personalized_mmd_dataset(
        mmd_dict=mmd_dict_enmo,
        feature_col="mean_mmd",
        baseline_method="mean",
        transform_mode="ratio",
        min_stable_per_window=1,
        include_n_vals=False,
    )

    summary_df, threshold_df, df_plot = lopo_window_threshold_report(
        X=X, y=y, meta=meta, plot=True
    )
