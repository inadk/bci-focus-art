import pandas as pd
import numpy as np
import time
from scipy.signal import butter, filtfilt
import os
import glob
import mido
from mido import Message
from scipy.signal import welch
from collections import deque

# --- CHANNEL GROUPS & NAME MAP ---
# Expected channel groups (case-insensitive names listed; we'll map to actual CSV columns via name_map)
groups = {
    "frontal":   ["Fz"],
    "central":   ["Cz", "C3", "C4"],
    "parietal":  ["PO7", "PO8", "Pz"],
    "occipital": ["Oz"],
}

# Mapping of canonical electrode names -> actual CSV column names
name_map = {
    "Fz":  "EEG 1",
    "C3":  "EEG 2",
    "Cz":  "EEG 3",
    "C4":  "EEG 4",
    "Pz":  "EEG 5",
    "PO7": "EEG 6",
    "Oz":  "EEG 7",
    "PO8": "EEG 8",
}


# --- PSD / Bandpower Helpers ---
def _bandpower_from_psd(f, Pxx, fmin, fmax):
    """Integrate the PSD between fmin and fmax (inclusive)."""
    idx = np.logical_and(f >= fmin, f <= fmax)
    if not np.any(idx):
        return 0.0
    # Trapezoidal integration
    return np.trapz(Pxx[idx], f[idx])

# --- BANDPOWER COMPUTATION ---
def compute_band_powers_1ch(x, fs):
    """
    Compute Welch PSD for a single channel, return power in alpha, beta, total(4-40 Hz).
    """
    if len(x) < int(0.5 * fs):
        # too short for meaningful PSD—avoid noisy estimates
        return {"alpha": 0.0, "beta": 0.0, "total": 1e-12}

    # Welch PSD; nperseg chosen to fit short windows robustly
    nperseg = min(256, len(x))
    noverlap = nperseg // 2
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")

    # Bands (adjust if you prefer slightly different edges)
    alpha = _bandpower_from_psd(f, Pxx, 8.0, 12.0)   # or 8–13 Hz
    beta  = _bandpower_from_psd(f, Pxx, 13.0, 30.0)
    total = _bandpower_from_psd(f, Pxx, 4.0, 40.0)   # exclude DC/very low

    # Avoid division by zero
    if total <= 0:
        total = 1e-12

    return {"alpha": alpha, "beta": beta, "total": total}

# --- FOCUS CLASSIFICATION ---
def classify_focus(window_df, fs):
    """
    Classify a 2s window as focused/unfocused based on bandpower fractions.

    Heuristic:
      - Frontal + Central + Parietal: focus_score += (beta_frac - alpha_frac)
      - Occipital (Oz):               focus_score -= 0.5 * (alpha_frac)

    Where x_frac = band / total(4–40 Hz), averaged over available channels.
    Threshold 0 → 'focused' else 'unfocused'.
    """
    # Resolve each group to actual dataframe column names via name_map, keep only those that exist
    def mapped_present(names):
        cols = []
        for n in names:
            col = name_map.get(n)
            if col in window_df.columns:
                cols.append(col)
        return cols

    present = {g: mapped_present(chs) for g, chs in groups.items()}

    # If nothing useful, bail early
    if not (present["frontal"] or present["central"] or present["parietal"] or present["occipital"]):
        return "unknown", 0.0, {"reason": "No expected EEG channels present via name_map"}

    # Compute per-channel fractions then average
    def _avg_fraction(ch_list, band):
        vals = []
        for ch in ch_list:
            bp = compute_band_powers_1ch(window_df[ch].values, fs)
            frac = bp[band] / bp["total"]
            vals.append(frac)
        return float(np.mean(vals)) if vals else None

    # Frontal + Central + Parietal combined
    fcp_channels = present["frontal"] + present["central"] + present["parietal"]
    alpha_fcp = _avg_fraction(fcp_channels, "alpha") if fcp_channels else None
    beta_fcp  = _avg_fraction(fcp_channels, "beta")  if fcp_channels else None

    # Occipital alpha (Oz only per mapping)
    alpha_occ = _avg_fraction(present["occipital"], "alpha") if present["occipital"] else None

    # Build score
    score = 0.0
    terms = {}

    if beta_fcp is not None and alpha_fcp is not None:
        score += (beta_fcp - alpha_fcp)
        terms.update({"beta_fcp": beta_fcp, "alpha_fcp": alpha_fcp})

    if alpha_occ is not None:
        score -= 0.5 * alpha_occ
        terms.update({"alpha_occ": alpha_occ})

    # Decide label
    if beta_fcp is None or alpha_fcp is None:
        label = "unknown"  # Not enough F/C/P channels to decide
    else:
        label = "focused" if score > 0.0 else "unfocused"

    return label, float(score), terms

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

# --- DECOMPOSE WAWES ---
# --- EEG BAND DECOMPOSITION ---
# Standard EEG bands (Hz). You can tweak these if needed.
EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),   # keep "low gamma" to stay well below Nyquist & mains hum
}

def decompose_eeg_bands(df, fs, order=4, bands=None, channel_cols=None, suffix_fmt="{col}_{band}"):
    """
    Band-decompose each EEG channel in `df` into the specified frequency bands.

    Parameters
    ----------
    df : pd.DataFrame
        Input samples with EEG columns and (optionally) other columns like TRIG.
    fs : float
        Sampling rate (Hz).
    order : int
        Butterworth filter order.
    bands : dict[str, tuple[float, float]] | None
        Mapping band name -> (low_hz, high_hz). Defaults to EEG_BANDS.
    channel_cols : list[str] | None
        Which columns to treat as EEG channels. Defaults to all columns except the last if it's 'TRIG',
        or any column whose name starts with 'EEG'.
    suffix_fmt : str
        Format for naming output columns. Must include {col} and {band}.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same index as `df` containing bandpassed signals.
        Each column is named like "<channel>_<band>".
    """
    if bands is None:
        bands = EEG_BANDS

    # Heuristic to pick EEG channel columns:
    if channel_cols is None:
        cols = list(df.columns)
        # If your last column is TRIG, skip it
        if len(cols) and cols[-1].strip().upper() == "TRIG":
            cols = cols[:-1]
        # Prefer columns that look like EEG channels
        eeg_like = [c for c in cols if c.lower().startswith("eeg")]
        channel_cols = eeg_like if eeg_like else cols

    nyq = 0.5 * fs
    out = pd.DataFrame(index=df.index)

    for col in channel_cols:
        x = df[col].values
        for band_name, (lo, hi) in bands.items():
            # Keep edges within valid range for the samplerate
            lo_adj = max(lo, 0.01)
            hi_adj = min(hi, nyq - 0.01)
            if hi_adj <= lo_adj:
                # If a band is invalid for the current fs, skip it gracefully
                continue
            y = butter_bandpass_filter(x, lo_adj, hi_adj, fs, order)
            out[suffix_fmt.format(col=col, band=band_name)] = y

    return out


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
