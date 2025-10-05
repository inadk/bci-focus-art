import pandas as pd
import numpy as np
import time
from scipy.signal import butter, filtfilt
import os
import glob
import mido
from mido import Message

# --- MIDI SETUP ---
print("Available MIDI output ports:")
output_ports = mido.get_output_names()
print(output_ports)

# Open the correct MIDI output port
midi_port_name = 'PythonPort 1'  # Change this if different
outport = mido.open_output(midi_port_name)
print(f"Connected to MIDI port: {midi_port_name}")

# --- Unicorn CSV File Folder ---
folder_path = r"C:\Users\hackaton\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Recorder"

# --- Helper: Get the latest non-"raw" CSV file ---
def get_latest_csv_file(folder):
    list_of_files = glob.glob(os.path.join(folder, '*.csv'))
    list_of_files = [f for f in list_of_files if 'raw' not in f.lower()]
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

# --- Filtering ---
fs = 250  # Sampling rate
lowcut = 0.1
highcut = 50
order = 4
blink_threshold = 50     # µV
amplitude_limit = 200    # µV

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        raise ValueError(f"Not enough samples for filtering: {len(data)} <= padlen ({padlen})")
    return filtfilt(b, a, data)

# --- Start Monitoring ---
file_path = get_latest_csv_file(folder_path)
if not file_path:
    print("No non-raw CSV files found. Start recording and try again.")
    outport.close()
    exit()

print(f"Monitoring file: {file_path}")
rows_read = 0

# Define expected column names (change if different)
num_channels = 8  # Number of EEG channels
column_names = [f'EEG {i+1}' for i in range(num_channels)] + ['Timestamp']

try:
    while True:
        try:
            # Detect file change
            latest_file = get_latest_csv_file(folder_path)
            if latest_file != file_path:
                print(f"New file detected: {latest_file}")
                file_path = latest_file
                rows_read = 0

            # Read new rows (no header in file)
            try:
                df = pd.read_csv(file_path, skiprows=range(1, rows_read + 1), header=None)
            except pd.errors.EmptyDataError:
                print("CSV file empty or in use. Waiting...")
                time.sleep(0.5)
                continue

            if df.empty or len(df) <= 30:
                print(f"Too few samples ({len(df)}), waiting...")
                time.sleep(0.5)
                continue

            if df.shape[1] != len(column_names):
                print(f"Unexpected number of columns ({df.shape[1]}), expected {len(column_names)}. Skipping...")
                time.sleep(0.5)
                continue

            df.columns = column_names
            print(f"Read {len(df)} new rows. Columns: {df.columns.tolist()}")
            rows_read += len(df)

            # Skip calibration
            if rows_read <= 4000:
                print(f"Skipping calibration ({rows_read} rows so far)...")
                continue
            elif rows_read - len(df) < 4000:
                df = df.iloc[(4000 - (rows_read - len(df))):]

            # Filter each EEG channel
            for col in df.columns[:-1]:
                try:
                    df[col] = butter_bandpass_filter(df[col], lowcut, highcut, fs, order)
                except ValueError as ve:
                    print(f"Skipping filter for {col}: {ve}")
                    continue

            # Drop rows with large noise
            valid_rows = df[(df[df.columns[:-1]].abs() <= amplitude_limit).all(axis=1)]
            if valid_rows.empty:
                continue

            # Detect blinks on EEG 1
            signal = valid_rows['EEG 1'].values
            above_threshold = signal > blink_threshold
            blink_starts = np.where(np.diff(above_threshold.astype(int)) == 1)[0]

            for idx in blink_starts:
                global_idx = rows_read - len(df) + valid_rows.index[idx]
                print(f"Blink detected at sample index {global_idx}")

                # Send MIDI note
                outport.send(Message('note_on', note=60, velocity=100))
                time.sleep(0.1)
                outport.send(Message('note_off', note=60, velocity=0))

            time.sleep(0.5)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

except KeyboardInterrupt:
    print("Stopped monitoring.")

finally:
    outport.close()
    print("MIDI port closed.")
