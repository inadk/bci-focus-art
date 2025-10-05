import pandas as pd
import numpy as np
import time
from scipy.signal import butter, filtfilt
import os
import glob

# Folder where Unicorn Recorder saves files
folder_path = r"C:\Users\Administrator\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Recorder"

# Function to find latest non-"raw" CSV file in folder
def get_latest_csv_file(folder):
    list_of_files = glob.glob(os.path.join(folder, '*.csv'))
    # Exclude files containing 'raw'
    list_of_files = [f for f in list_of_files if 'raw' not in f.lower()]
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# Sampling rate and filter settings
fs = 250  # adjust if different
lowcut = 0.1
highcut = 50
order = 4
blink_threshold = 50  # µV
amplitude_limit = 200  # µV

# Butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Find latest valid file
file_path = get_latest_csv_file(folder_path)
if not file_path:
    print("No non-raw CSV files found in folder. Please start recording and try again.")
    exit()

print(f"Monitoring file: {file_path}")

rows_read = 0

while True:
    try:
        # Check if a newer file has appeared
        latest_file = get_latest_csv_file(folder_path)
        if latest_file != file_path:
            print(f"New file detected: {latest_file}")
            file_path = latest_file
            rows_read = 0  # reset counter for new file

        # Read new rows
        df = pd.read_csv(file_path, skiprows=range(1, rows_read + 1))
        df = df.apply(pd.to_numeric, errors = 'coerce').dropna()
        
        if df.empty:
            time.sleep(0.5)
            continue

        rows_read += len(df)

        # Skip initial calibration samples
        if rows_read <= 4000:
            print(f"Skipping calibration samples ({rows_read} rows so far)...")
            continue
        elif rows_read - len(df) < 4000:
            df = df.iloc[(4000 - (rows_read - len(df))):]

        # Apply bandpass filter to EEG channels
        for col in df.columns[:-1]:  # exclude TRIG
            df[col] = butter_bandpass_filter(df[col], lowcut, highcut, fs, order)

        # Drop rows where any EEG channel exceeds ±200 µV
        valid_rows = df[(df[df.columns[:-1]].abs() <= amplitude_limit).all(axis=1)]

        if valid_rows.empty:
            continue

        # Blink detection on EEG 1
        signal = valid_rows['EEG 1'].values
        above_threshold = signal > blink_threshold
        blink_starts = np.where(np.diff(above_threshold.astype(int)) == 1)[0]

        for idx in blink_starts:
            global_idx = rows_read - len(df) + valid_rows.index[idx]
            print(f"Blink detected at sample index {global_idx}")

        time.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopped monitoring.")
        break

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)
