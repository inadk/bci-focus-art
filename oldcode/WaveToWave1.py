import pandas as pd
import numpy as np
import time
from scipy.signal import butter, filtfilt
import os
import glob
import mido
from mido import Message

# remove everything with midi



# --- MIDI SETUP ---
print("Available MIDI output ports:", mido.get_output_names())
midi_port = 'PythonPort 1'  # Change to your actual MIDI port
outport = mido.open_output(midi_port)
print(f"Using MIDI port: {midi_port}")

# --- SETTINGS ---
folder_path = r"C:\Users\hackaton\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Recorder"
fs = 250  # Sampling rate
lowcut = 0.1
highcut = 50
order = 4
blink_threshold = 60       # µV
amplitude_limit = 120      # µV
calibration_skip = 4000    # Samples to skip
neutral_pct = 15            # Hemispheric comparison threshold

# Expected column headers (adjust based on your device setup)
eeg_columns = ['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8', 'TRIG']

# --- FILTER FUNCTION ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# --- GET LATEST FILE ---
def get_latest_csv_file(folder):
    files = glob.glob(os.path.join(folder, '*.csv'))
    files = [f for f in files if 'raw' not in f.lower()]
    return max(files, key=os.path.getctime) if files else None

# --- INITIAL SETUP ---
file_path = get_latest_csv_file(folder_path)
if not file_path:
    print("No valid CSV file found. Please start a Unicorn recording.")
    outport.close()
    exit()

print(f"Monitoring file: {file_path}")

rows_read = 0
last_blink_sample = -1
last_note = None

# --- Hemispheric comparison additions ---
buffer = pd.DataFrame()
last_hem_time = time.time()
last_hem_note = None

try:
    while True:
        # Check for new file
        latest_file = get_latest_csv_file(folder_path)
        if latest_file != file_path:
            print(f"New file detected: {latest_file}")
            file_path = latest_file
            rows_read = 0
            last_blink_sample = -1
            last_note = None
            last_hem_note = None
            buffer = pd.DataFrame()

        # Read new rows as raw data, assign column names
        df = pd.read_csv(file_path, header=None, skiprows=rows_read + 1)
        df.columns = eeg_columns  # Set correct headers
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        if df.empty:
            time.sleep(0.5)
            continue

        rows_read += len(df)

        # Skip calibration period
        if rows_read <= calibration_skip:
            print(f"Skipping calibration samples ({rows_read} rows so far)...")
            continue
        elif rows_read - len(df) < calibration_skip:
            df = df.iloc[(calibration_skip - (rows_read - len(df))):]

        # Apply bandpass filter to all EEG columns (except TRIG)
        for col in eeg_columns[:-1]:
            df[col] = butter_bandpass_filter(df[col], lowcut, highcut, fs, order)

        # Remove out-of-range samples
        valid_rows = df[(df[eeg_columns[:-1]].abs() <= amplitude_limit).all(axis=1)]
        if valid_rows.empty:
            continue

        # --- Append to rolling buffer for hemispheric comparison ---
        buffer = pd.concat([buffer, valid_rows], ignore_index=True)
        if len(buffer) > 3 * fs:
            buffer = buffer.iloc[-3 * fs:]

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

        # --- Hemispheric comparison every 3s ---
        if time.time() - last_hem_time >= 3 and len(buffer) >= 3 * fs:
            l_rms = np.sqrt((buffer['EEG 4']**2).mean())
            r_rms = np.sqrt((buffer['EEG 2']**2).mean())
            pct_diff = abs(l_rms - r_rms) / max(l_rms, r_rms) * 100 if max(l_rms, r_rms) != 0 else 0

            if pct_diff < neutral_pct:
                note, side = 65, "Neutral"
            elif l_rms > r_rms:
                note, side = 64, "Left"
            else:
                note, side = 62, "Right"

            if note != last_hem_note:
                print(f"[{time.strftime('%H:%M:%S')}] {side} hemisphere more active (L={l_rms:.1f}, R={r_rms:.1f}, Δ={pct_diff:.1f}%)")
                outport.send(Message('note_on', note=note, velocity=100))
                time.sleep(0.1)
                outport.send(Message('note_off', note=note, velocity=0))
                last_hem_note = note

            last_hem_time = time.time()

        time.sleep(0.5)

except KeyboardInterrupt:
    print("Monitoring stopped by user.")

finally:
    outport.close()
    print("MIDI port closed.")
