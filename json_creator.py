"""
json_creator.py

Purpose:
    Builds per-day, fixed-length walking-activity segments (x, y, z accelerometer
    samples plus a circular time-of-day feature) from patient accelerometer data
    and signal-processing time-series labels, then exports the segments to a
    single JSON file.

Command-line arguments:
    sys.argv[1] (file): base file name (without extension) shared by the two
        input CSVs, e.g. "<patient_id>_<date>".
    sys.argv[2] (dir):  directory containing the input CSVs and where the
        output JSON will be written.
        # TODO: Set this to the input/output directory for your patient data
        # (e.g. an environment variable or config value), do not hardcode
        # internal server paths or patient identifiers in source control.

Inputs:
    "{dir}/{file}.csv": raw 3D accelerometer data (expects 'time', 'x', 'y', 'z' columns).
    "{dir}/{file}-timeSeries.csv": derived time-series data (expects 'time',
        'walking', 'acc' columns).

Processing steps performed by this script:
    1. Loads both CSVs with Dask, parses the 'time' column to UTC timestamps,
       and merges them on 'time'.
    2. Sorts and repartitions the merged dataframe, then splits it into one
       Dask partition per calendar date.
    3. For each date, forward-fills and drops remaining NaNs, then isolates
       rows flagged as 'walking'.
    4. Groups consecutive walking rows into contiguous "batches" (runs of
       adjacent walking samples), then splits each batch into fixed-length
       (default 50-sample) segments.
    5. For each valid segment, builds a feature array of [x, y, z, time_circ],
       where time_circ is a sine-encoded circular representation of the
       segment's start time of day. Segments shorter than the fixed length
       are skipped.
    6. Collects all segments per date under the 'walking' label, converts them
       to plain Python lists, and writes the resulting nested dictionary to
       "{dir}/{file}.json".

Output:
    A JSON file structured as {"walking": {"<date>": [[[x, y, z, time_circ], ...], ...]}}.
"""

import pysiglib
import iisignature
import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json
import sys

file_names = []
level = 2
print('code has started running')
file = sys.argv[1]
dir = sys.argv[2]

print('In Directory: ', dir)
print('Processing file: ', file)
df_3d = dd.read_csv("{dir}/{file}.csv".format(dir=dir, file=file))
df_l = dd.read_csv("{dir}/{file}-timeSeries.csv".format(dir=dir, file=file), usecols=['time', 'walking', 'acc'])

df_3d['time'] = dd.to_datetime(df_3d['time'].str.split(r' \[').str[0], utc=True)
df_l['time'] = dd.to_datetime(df_l['time'].str.split(r' \[').str[0], utc=True)
df_merge = dd.merge(df_3d, df_l, on='time', how='left')
df_merge = df_merge.sort_values(by='time', ascending=True).reset_index(drop=True)
df_merge = df_merge.repartition(partition_size="168MB")
df_merge['date'] = df_merge['time'].dt.date
unique_dates = df_merge['date'].unique().compute()
dask_dict = {str(d): df_merge[df_merge['date'] == d] for d in unique_dates}
final_dict = {'walking': {}}

json_dict = {}

for key in dask_dict.keys():
	print('Processing {key}'.format(key=key))
	df = dask_dict[key].compute()
	df = df.ffill().dropna()
	for label in ["walking"]:
		walking_rows = df[df[label] == 1].copy()

		if len(walking_rows)==0:
			print('0 walking rows')
			continue
		
		walking_rows["Batch"] = (walking_rows.index.to_series().diff().fillna(2) != 1).cumsum()
		walking_rows = walking_rows.reset_index(drop=True)
		length = 50

		no_of_batches = len(walking_rows)//length
		dim = 1
		
		x = []
		index = 0
		print('No of batches: ', walking_rows["Batch"].max())
		for i in range(1, int(walking_rows["Batch"].max())+1):
			current_batch = walking_rows[walking_rows["Batch"] == i].copy()
			current_batch['group'] = np.arange(len(current_batch)) // length
			series_values_x = current_batch.groupby('group')['x'].apply(list)
			series_values_y = current_batch.groupby('group')['y'].apply(list)
			series_values_z = current_batch.groupby('group')['z'].apply(list)

			batch_time_seconds = (
			current_batch['time'].iloc[0].hour * 3600 +
			current_batch['time'].iloc[0].minute * 60 +
			current_batch['time'].iloc[0].second
			)
			time_circ = np.sin(
				2 * np.pi * batch_time_seconds / 86400
			)
			time = np.full(length, time_circ)
			for ind in range(len(series_values_x)):
				series_length = len(series_values_x.iloc[ind])
				if series_length < length:
					continue
				values_copy_x = series_values_x.iloc[ind]
				values_copy_y = series_values_y.iloc[ind]
				values_copy_z = series_values_z.iloc[ind]
				combined = np.column_stack((values_copy_x, values_copy_y, values_copy_z, time))
				values_scaled = combined
				x.append(values_scaled)
			print('batch {i} processed'.format(i=i))
		final_dict[label].update({key: x})
for i in final_dict:
	json_dict[i] = {}
	for j in final_dict[i]:
		json_dict[i][j] = []
		for k in final_dict[i][j]:
			item = k.tolist()
			json_dict[i][j].append(item)
with open(f'{dir}/{file}.json', 'w') as f:
	json.dump(json_dict, f)
