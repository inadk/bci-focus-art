import pandas as pd
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt, welch
import os
import glob
import matplotlib.pyplot as plt

# ========================
# CONFIG
# ========================
mode = "offline"   # "offline" or "online"
folder_path = r"C:\Users\hackaton\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Recorder"
offline_file_path = r"olddata\UnicornRawDataRecorder_03_05_2025_20_13_360.csv"  # set to a specific CSV path or leave None to use newest in folder

fs = 250           # Sampling rate (Hz)
lowcut = 0.1
highcut = 50
order = 4
amplitude_limit = 200    # µV
calibration_skip = 4000  # Samples to skip at beginning

# Focus windowing
window_seconds = 2
window_len = int(fs * window_seconds)

# OFFLINE: non-overlapping segments ("one after the other")
offline_hop_seconds = 2
offline_hop_len = int(fs * offline_hop_seconds)

# ONLINE: rolling updates (you can change to 1s hop later if you want overlap)
online_hop_seconds = 2
online_hop_len = int(fs * online_hop_seconds)

# Optional label smoothing (majority vote over last N windows) – used in online mode
smooth_N = 5

# ========================
# CHANNEL GROUPS & NAME MAP
# ========================
groups = {
    "frontal":   ["Fz"],
    "central":   ["Cz", "C3", "C4"],
    "parietal":  ["PO7", "PO8", "Pz"],
    "occipital": ["Oz"],
}
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

# ========================
# SIGNAL HELPERS
# ========================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_latest_csv_file(folder):
    files = glob.glob(os.path.join(folder, '*.csv'))
    files = [f for f in files if 'raw' not in f.lower()]
    return max(files, key=os.path.getctime) if files else None

def _bandpower_from_psd(f, Pxx, fmin, fmax):
    idx = np.logical_and(f >= fmin, f <= fmax)
    if not np.any(idx):
        return 0.0
    return np.trapz(Pxx[idx], f[idx])

def compute_band_powers_1ch(x, fs):
    if len(x) < int(0.5 * fs):
        return {"alpha": 0.0, "beta": 0.0, "total": 1e-12}
    nperseg = min(256, len(x))
    noverlap = nperseg // 2
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    alpha = _bandpower_from_psd(f, Pxx, 8.0, 12.0)
    beta  = _bandpower_from_psd(f, Pxx, 13.0, 30.0)
    total = _bandpower_from_psd(f, Pxx, 4.0, 40.0)
    if total <= 0:
        total = 1e-12
    return {"alpha": alpha, "beta": beta, "total": total}

def classify_focus(window_df, fs):
    """
    Heuristic:
      - Frontal + Central + Parietal: score += (beta_frac - alpha_frac)
      - Occipital (Oz):               score -= 0.5 * alpha_frac
      label = 'focused' if score > 0 else 'unfocused'
    """
    def mapped_present(names):
        cols = []
        for n in names:
            col = name_map.get(n)
            if col in window_df.columns:
                cols.append(col)
        return cols

    present = {g: mapped_present(chs) for g, chs in groups.items()}
    if not (present["frontal"] or present["central"] or present["parietal"] or present["occipital"]):
        return "unknown", 0.0, {"reason": "No expected EEG channels present via name_map"}

    def _avg_fraction(ch_list, band):
        vals = []
        for ch in ch_list:
            bp = compute_band_powers_1ch(window_df[ch].values, fs)
            frac = bp[band] / bp["total"]
            vals.append(frac)
        return float(np.mean(vals)) if vals else None

    fcp_channels = present["frontal"] + present["central"] + present["parietal"]
    alpha_fcp = _avg_fraction(fcp_channels, "alpha") if fcp_channels else None
    beta_fcp  = _avg_fraction(fcp_channels, "beta")  if fcp_channels else None
    alpha_occ = _avg_fraction(present["occipital"], "alpha") if present["occipital"] else None

    score = 0.0
    terms = {}
    if beta_fcp is not None and alpha_fcp is not None:
        score += (beta_fcp - alpha_fcp)
        terms.update({"beta_fcp": beta_fcp, "alpha_fcp": alpha_fcp})
    if alpha_occ is not None:
        score -= 0.5 * alpha_occ
        terms.update({"alpha_occ": alpha_occ})

    if beta_fcp is None or alpha_fcp is None:
        label = "unknown"
    else:
        label = "focused" if score > 0.0 else "unfocused"

    return label, float(score), terms

def labels_to_binary(labels):
    """Map labels -> binary: focused=1, unfocused/unknown=0."""
    return [1 if (lbl == "focused") else 0 for lbl in labels]

def _runs_of_value(binary, value):
    """
    Return list of (start_idx, length) runs where binary==value.
    Non-overlapping segment indexing.
    """
    runs = []
    start = None
    for i, b in enumerate(binary):
        if b == value:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - start))
                start = None
    if start is not None:
        runs.append((start, len(binary) - start))
    return runs

def compute_focus_stats(binary, window_seconds):
    """Compute summary stats from a binary focus vector."""
    n = len(binary)
    total_time = n * window_seconds
    focused_time = sum(binary) * window_seconds
    unfocused_time = total_time - focused_time

    focused_runs = _runs_of_value(binary, 1)
    unfocused_runs = _runs_of_value(binary, 0)

    longest_focused_len = max([l for _, l in focused_runs], default=0)
    longest_unfocused_len = max([l for _, l in unfocused_runs], default=0)

    stats = {
        "segments": n,
        "window_seconds": window_seconds,
        "total_time_seconds": total_time,
        "focused_time_seconds": focused_time,
        "unfocused_time_seconds": unfocused_time,
        "longest_continuous_focused_seconds": longest_focused_len * window_seconds,
        "longest_continuous_defocused_seconds": longest_unfocused_len * window_seconds,
        "number_of_focused_periods": len(focused_runs),
    }
    return stats, focused_runs, unfocused_runs

def save_focus_plot(time_centers, binary, out_path):
    """
    Save a time series plot (step plot) of focus status over time.
    y=1 focused, y=0 unfocused.
    """
    plt.figure(figsize=(12, 3))
    # Step plot across segment centers; make it look like blocks by repeating points
    # Build step-like arrays:
    if len(binary) > 0:
        # Convert centers to edges for a cleaner step look
        step_x = []
        step_y = []
        half = (time_centers[1] - time_centers[0]) / 2 if len(time_centers) > 1 else 1.0
        for i, (t, b) in enumerate(zip(time_centers, binary)):
            left = t - half
            right = t + half
            step_x.extend([left, right])
            step_y.extend([b, b])
        plt.plot(step_x, step_y, drawstyle="default")
    plt.yticks([0,1], ["Unfocused","Focused"])
    plt.xlabel("Time (s)")
    plt.title("Focus status over time (2 s segments)")
    plt.ylim(-0.2, 1.2)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ========================
# OFFLINE MODE
# ========================
def run_offline():
    file_path = offline_file_path or get_latest_csv_file(folder_path)
    if not file_path:
        print("No non-raw CSV files found. Provide offline_file_path or place a CSV in the folder.")
        return [], {}

    print(f"[OFFLINE] Processing file: {file_path}")

    # Load whole file
    df = pd.read_csv(file_path)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    # Only mapped EEG columns that exist
    mapped_eeg_cols = [col for col in name_map.values() if col in df.columns]
    if not mapped_eeg_cols:
        print("No expected EEG columns found per name_map.")
        return [], {}

    # Remove calibration region
    if len(df) <= calibration_skip:
        print("File shorter than calibration skip; nothing to process.")
        return [], {}

    df = df.iloc[calibration_skip:].reset_index(drop=True)

    # Bandpass filter
    for col in mapped_eeg_cols:
        df[col] = butter_bandpass_filter(df[col].values, lowcut, highcut, fs, order)

    # Artifact guard
    df = df[(df[mapped_eeg_cols].abs() <= amplitude_limit).all(axis=1)].reset_index(drop=True)
    if df.empty:
        print("All samples removed by artifact guard.")
        return [], {}

    # Segment non-overlapping 2 s windows
    N = len(df)
    labels = []
    scores = []
    details_list = []
    time_centers = []

    for start in range(0, N - window_len + 1, offline_hop_len):
        end = start + window_len
        window = df.loc[start:end-1, mapped_eeg_cols]

        label, score, details = classify_focus(window, fs)
        labels.append(label)
        scores.append(score)
        details_list.append(details)

        # center time of segment
        t_center = (start + end) / 2 / fs
        time_centers.append(t_center)

    # Convert to binary (focused=1; unfocused/unknown=0)
    binary = labels_to_binary(labels)

    # Stats
    stats, focused_runs, unfocused_runs = compute_focus_stats(binary, window_seconds)

    # Save plot next to CSV
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_png = os.path.join(os.path.dirname(file_path), f"{base}_focus_timeseries.png")
    save_focus_plot(time_centers, binary, out_png)

    # Console summary
    print("\n=== Focus Summary (OFFLINE) ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"plot_path: {out_png}")

    # If you want, also print the binary array (may be long)
    # print('binary:', binary)

    return binary, stats, out_png


# ========================
# ONLINE MODE
# ========================
def run_online():
    file_path = get_latest_csv_file(folder_path)
    if not file_path:
        print("No non-raw CSV files found in folder. Start recording and try again.")
        return
    print(f"[ONLINE] Monitoring file: {file_path}")

    rows_read = 0
    seg_buffer = pd.DataFrame()
    label_history = deque(maxlen=smooth_N)

    try:
        while True:
            latest_file = get_latest_csv_file(folder_path)
            if latest_file != file_path:
                print(f"New file detected: {latest_file}")
                file_path = latest_file
                rows_read = 0
                seg_buffer = pd.DataFrame()
                label_history.clear()

            # Read newly appended rows (skip header = row 0)
            df = pd.read_csv(file_path, skiprows=range(1, rows_read + 1))
            df = df.apply(pd.to_numeric, errors='coerce').dropna()

            if df.empty:
                time.sleep(0.5)
                continue

            rows_read += len(df)

            # Skip initial calibration
            if rows_read <= calibration_skip:
                print(f"Skipping calibration samples ({rows_read} rows so far)...")
                continue
            elif rows_read - len(df) < calibration_skip:
                df = df.iloc[(calibration_skip - (rows_read - len(df))):]

            mapped_eeg_cols = [col for col in name_map.values() if col in df.columns]
            if not mapped_eeg_cols:
                time.sleep(0.5)
                continue

            # Filter
            for col in mapped_eeg_cols:
                df[col] = butter_bandpass_filter(df[col].values, lowcut, highcut, fs, order)

            # Artifact guard
            valid_rows = df[(df[mapped_eeg_cols].abs() <= amplitude_limit).all(axis=1)]
            if valid_rows.empty:
                time.sleep(0.2)
                continue

            # Append to rolling buffer
            seg_buffer = pd.concat([seg_buffer, valid_rows[mapped_eeg_cols]], axis=0)

            # Process as many full windows as available with chosen hop
            while len(seg_buffer) >= window_len:
                window = seg_buffer.iloc[:window_len]
                label, score, details = classify_focus(window, fs)

                label_history.append(label)
                usable = [l for l in label_history if l != "unknown"]
                smoothed = max(usable, key=usable.count) if usable else label

                t_sec = rows_read / fs
                print(f"[ONLINE ] t~{t_sec:8.2f}s | label={smoothed.upper()} | "
                      f"score={score:.3f} | details={details}")

                # Advance by hop (non-overlapping = online_hop_len == window_len; change if desired)
                seg_buffer = seg_buffer.iloc[online_hop_len:]

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopped monitoring.")

# ========================
# ENTRY
# ========================
if __name__ == "__main__":
    if mode.lower() == "offline":
        binary, stats, png_path = run_offline()
        # Explicitly show the array and stats as requested:
        print("\nBinary focus array (1=focused, 0=unfocused):")
        print(binary)
        print("\nStatistics:")
        print(stats)
        print(f"\nSaved focus time-series plot to: {png_path}")
    else:
        run_online()

