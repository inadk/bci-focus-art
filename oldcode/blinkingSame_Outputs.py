import pandas as pd
import numpy as np
import time
from scipy.signal import butter, filtfilt
import os
import glob
import mido
from mido import Message

# --- MIDI SETUP ---
print("Available MIDI output ports:", mido.get_output_names())
midi_port = 'PythonPort 1'  # Adjust as needed
outport = mido.open_output(midi_port)
print(f"Using MIDI port: {midi_port}")

# --- SETTINGS ---
folder_path = r"C:\Users\hackaton\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Recorder"
fs = 250  # Sampling rate
lowcut = 0.1
highcut = 50
order = 4
blink_threshold = 65  # µV
amplitude_limit = 200  # µV
calibration_skip = 4000  # Samples to skip at beginning

# --- FILTER FUNCTION ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# --- FIND NEWEST FILE ---
def get_latest_csv_file(folder):
    list_of_files = glob.glob(os.path.join(folder, '*.csv'))
    list_of_files = [f for f in list_of_files if 'raw' not in f.lower()]
    return max(list_of_files, key=os.path.getctime) if list_of_files else None

# --- INITIALIZATION ---
file_path = get_latest_csv_file(folder_path)
if not file_path:
    print("No non-raw CSV files found in folder. Please start recording and try again.")
    outport.close()
    exit()

print(f"Monitoring file: {file_path}")

rows_read = 0
last_blink_sample = -1
last_note = None

try:
    while True:
        latest_file = get_latest_csv_file(folder_path)
        if latest_file != file_path:
            print(f"New file detected: {latest_file}")
            file_path = latest_file
            rows_read = 0
            last_blink_sample = -1

        # Read new rows
        df = pd.read_csv(file_path, skiprows=range(1, rows_read + 1))
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        if df.empty:
            time.sleep(0.5)
            continue

        rows_read += len(df)

        # Skip calibration
        if rows_read <= calibration_skip:
            print(f"Skipping calibration samples ({rows_read} rows so far)...")
            continue
        elif rows_read - len(df) < calibration_skip:
            df = df.iloc[(calibration_skip - (rows_read - len(df))):]

        # Apply bandpass filter
        for col in df.columns[:-1]:  # Exclude TRIG
            df[col] = butter_bandpass_filter(df[col], lowcut, highcut, fs, order)

        # Remove out-of-range samples
        valid_rows = df[(df[df.columns[:-1]].abs() <= amplitude_limit).all(axis=1)]
        if valid_rows.empty:
            continue

        # Blink detection on EEG 1
        signal = valid_rows['EEG 1'].values
        above_threshold = signal > blink_threshold
        blink_starts = np.where(np.diff(above_threshold.astype(int)) == 1)[0]

        for idx in blink_starts:
            global_idx = rows_read - len(df) + valid_rows.index[idx]
            if global_idx > last_blink_sample:
                print(f"Blink detected at sample index {global_idx}")

                # Send MIDI note for blink
                note = 60  # Middle C
                outport.send(Message('note_on', note=note, velocity=100))
                time.sleep(0.1)
                outport.send(Message('note_off', note=note, velocity=0))

                last_blink_sample = global_idx
                last_note = note

        # --- EXAMPLE: Trigger another note every 10 seconds (just a placeholder) ---
        if int(time.time()) % 10 == 0:
            note = 65  # F note
            if note != last_note:
                print("Sending timed note (F)")
                outport.send(Message('note_on', note=note, velocity=100))
                time.sleep(0.1)
                outport.send(Message('note_off', note=note, velocity=0))
                last_note = note

        time.sleep(0.5)

except KeyboardInterrupt:
    print("Stopped monitoring.")

finally:
    outport.close()
    print("MIDI connection closed.")
