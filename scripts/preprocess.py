import pandas as pd
import numpy as np
import ast
from datetime import datetime

# Read the CSV file
df = pd.read_csv('D:/CDIOFN/data/cd/filtered_lines.csv', sep=';')

# Convert start_time to datetime
df['start_time'] = pd.to_datetime(df['start_time'], unit='s')

# Round down to nearest 15-minute interval
df['time_bin'] = df['start_time'].dt.floor('15min')

# Parse path column to list of integers
df['path'] = df['path'].apply(ast.literal_eval)

# Determine time bins
min_time = df['time_bin'].min()
max_time = df['time_bin'].max()
time_bins = pd.date_range(start=min_time, end=max_time, freq='15min')
time_to_idx = {t: i for i, t in enumerate(time_bins)}
num_time_steps = len(time_bins)
num_roads = 6567  # 0 to 6566 inclusive

# Initialize traffic volume matrix
X = np.zeros((num_time_steps, num_roads), dtype=np.int32)

# Explode the dataframe to have one row per road per trajectory
df_exploded = df.explode('path')
df_exploded['time_idx'] = df_exploded['time_bin'].map(time_to_idx)
df_exploded['road'] = df_exploded['path']

# Group by time_idx and road, count occurrences
counts = df_exploded.groupby(['time_idx', 'road']).size()

# Populate the matrix
for (t_idx, road), count in counts.items():
    X[t_idx, road] = count

# Save the matrix
np.save('traffic_flow.npy', X)
