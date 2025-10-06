import pandas as pd
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt, welch
import os
import glob
import matplotlib.pyplot as plt
from collections import deque

# ========================
# CONFIG
# ========================
# ========================
# STREAM OUTPUT CONFIG
# ========================
emit_binary_stdout = True          # write plain 0/1 to stdout (one per line)
emit_binary_file = None            # e.g. "focus_stream.txt" to also append, or None to disable
quiet_mode = True                  # if True, suppress verbose prints so stdout is only 0/1
# --- NEW: score memory / auto-correlation ---
ar_history_len = 5        # how many past scores to include
ar_decay = 0.6            # geometric decay (0<ar_decay<1). Newer past scores weigh more.

# --- NEW: dynamic thresholding over a rolling buffer of scores ---
thr_history_len = 120    # ~ last 4 minutes if 2s hop (adjust as you like)
thr_low_pct = 45          # enforce at least ~45% below threshold
thr_high_pct = 55         # and ~45% above; threshold ~median-ish but guaranteed split
thr_eps = 1e-4            # small separation in degenerate cases


mode = "online"   # "offline" or "online"
folder_path = r"C:\Users\hackaton\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Recorder"
offline_file_path = r"olddata\UnicornRecorder_20251005_205735.csv"  # set to a specific CSV path or leave None to use newest in folder
# print isfile
print(os.path.isfile(offline_file_path))

image_output_path = "outputs"  # If None, will save next to CSV in offline mode

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
    return np.trapezoid(Pxx[idx], f[idx])

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

from collections import deque

class DynamicThreshold:
    """
    Keeps a rolling buffer of scores and returns a threshold so both classes appear.
    We compute a percentile bracket (e.g., 45th..55th) and take the midpoint.
    This guarantees some mass on each side as long as we have enough history.
    """
    def __init__(self, maxlen=600, low_pct=45, high_pct=55, eps=1e-4):
        self.buf = deque(maxlen=maxlen)
        self.low_pct = float(low_pct)
        self.high_pct = float(high_pct)
        self.eps = float(eps)

    def update(self, score: float):
        self.buf.append(float(score))

    def ready(self, min_points=20):
        return len(self.buf) >= min_points

    def threshold(self, default=0.0):
        if not self.buf:
            return default
        arr = np.asarray(self.buf)
        if np.allclose(arr.min(), arr.max()):
            # flat history: put threshold at that flat value +/- eps so both sides exist
            return arr.mean()
        lo = np.percentile(arr, self.low_pct)
        hi = np.percentile(arr, self.high_pct)
        if hi - lo < self.eps:
            # extremely tight: nudge to ensure separation
            mid = (lo + hi) / 2.0
            lo = mid - self.eps
            hi = mid + self.eps
        return (lo + hi) / 2.0


class ARScore:
    """
    Auto-correlated score = current + weighted sum of previous few scores with geometric decay.
    Equivalent to a short AR filter; not normalized so the scale remains comparable.
    """
    def __init__(self, history_len=5, decay=0.6):
        self.decay = float(decay)
        self.hist = deque(maxlen=history_len)

    def push_and_compute(self, current: float) -> float:
        # weights: [decay^1, decay^2, ...] for hist[0] (most recent) onward
        ar = current
        for i, prev in enumerate(reversed(self.hist), start=1):
            ar += (self.decay ** i) * prev
        # update history AFTER computing (so "previous" means strictly before current)
        self.hist.append(current)
        return ar

def label_with_threshold(score: float, thr: float) -> str:
    return "focused" if score > thr else "unfocused"


# ========================
# OFFLINE MODE
# ========================
def run_offline():
    file_path = offline_file_path# or get_latest_csv_file(folder_path)
    print(offline_file_path)
    print(file_path)
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
    base = base.replace("UnicornRawDataRecorder_", "")
    # save to outputs folder
    image_output_path = "outputs"
    os.makedirs(image_output_path, exist_ok=True)
    out_png = f"outputs/{base}_focus_timeseries.png"
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

def _majority_vote(labels):
    """Return 'focused' if focused votes > half, else 'unfocused'.
    Unknowns count as unfocused in the vote."""
    if not labels:
        return "unknown"
    votes = labels_to_binary(labels)  # focused=1, else 0
    return "focused" if sum(votes) > (len(votes) / 2.0) else "unfocused"

def run_online():
    """
    Every `online_hop_seconds`, read the newest non-raw CSV in `folder_path`,
    take the last 500 rows (2 s), bandpass + artifact guard, classify, and print.
    Uses a smoothing deque of length `smooth_N` for a majority-vote label.
    """
    print(f"[ONLINE] Watching folder: {folder_path}")
    print(f"[ONLINE] Polling every {online_hop_seconds}s; segment size: {window_len} samples (~{window_seconds}s)\n")

    smooth_q = deque(maxlen=smooth_N)
    # NEW: managers that persist during the session
    ar = ARScore(history_len=ar_history_len, decay=ar_decay)
    dyn_thr = DynamicThreshold(maxlen=thr_history_len, low_pct=thr_low_pct, high_pct=thr_high_pct, eps=thr_eps)

    last_file = None

    try:
        while True:
            file_path = offline_file_path
            if not file_path:
                print("[ONLINE] No non-raw CSV files found yet…")
                time.sleep(online_hop_seconds)
                continue

            # If a new recording starts, reset smoothing history.
            if file_path != last_file:
                last_file = file_path
                smooth_q.clear()
                print(f"[ONLINE] Now reading: {file_path}")

            # Read CSV (robust to partial writes)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"[ONLINE] Read error ({type(e).__name__}): {e}. Retrying…")
                time.sleep(online_hop_seconds)
                continue

            # Numeric only
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            n_total = len(df)

            # Wait until calibration phase has enough data
            if n_total < calibration_skip + window_len:
                remaining = max(calibration_skip + window_len - n_total, 0)
                print(f"[ONLINE] Waiting for calibration data: have {n_total}, need {calibration_skip + window_len} "
                      f"(~{remaining/fs:.1f}s more).")
                time.sleep(online_hop_seconds)
                continue

            # Tail 2s window (500 samples)
            tail = df.tail(window_len).reset_index(drop=True)

            # Only mapped EEG columns that exist
            mapped_eeg_cols = [col for col in name_map.values() if col in tail.columns]
            if not mapped_eeg_cols:
                print("[ONLINE] No expected EEG channels present per name_map; will retry…")
                time.sleep(online_hop_seconds)
                continue

            # Bandpass filter per channel
            try:
                for col in mapped_eeg_cols:
                    tail[col] = butter_bandpass_filter(tail[col].values, lowcut, highcut, fs, order)
            except Exception as e:
                print(f"[ONLINE] Filter error ({type(e).__name__}): {e}. Skipping this hop…")
                time.sleep(online_hop_seconds)
                continue
            """
            # Artifact guard
            clean = tail[(tail[mapped_eeg_cols].abs() <= amplitude_limit).all(axis=1)].reset_index(drop=True)
            # If too many samples are rejected, skip this segment to avoid biased PSD
            if len(clean) < int(0.8 * window_len):
                print("[ONLINE] Segment too noisy (artifact guard). Skipping…")
                time.sleep(online_hop_seconds)
                continue
            """
            clean = tail


            # Classify this 2s segment -> raw score
            raw_label, raw_score, terms = classify_focus(clean[mapped_eeg_cols], fs)

            # Auto-correlated score (uses recent history with geometric decay)
            s_ar = ar.push_and_compute(raw_score)

            # Update threshold buffer and compute dynamic threshold
            dyn_thr.update(s_ar)
            thr = dyn_thr.threshold(default=0.0)

            # Thresholded label (guaranteed split given enough history)
            dyn_label = label_with_threshold(s_ar, thr)

            # OPTIONAL: keep your majority vote smoothing, but apply it to dyn_label
            smooth_q.append(dyn_label)
            smoothed_label = _majority_vote(list(smooth_q))

            # Rough timestamp: total seconds since recording start
            t_seconds = n_total / fs

            # Decide binary bit from the smoothed dynamic label
            bit = 1 if smoothed_label == "focused" else 0

            if not quiet_mode:
                alpha_fcp = terms.get("alpha_fcp")
                beta_fcp  = terms.get("beta_fcp")
                alpha_occ = terms.get("alpha_occ")
                def _fmt(x): 
                    return "—" if x is None else f"{x:.3f}"
                print(
                    f"[t={t_seconds:7.2f}s] raw={raw_label:9s}  dyn={dyn_label:9s}  "
                    f"smoothed={smoothed_label:9s}  raw_score={raw_score:+.3f}  "
                    f"ar_score={s_ar:+.3f}  thr={thr:+.3f}  "
                    f"(β_fcp={_fmt(beta_fcp)}, α_fcp={_fmt(alpha_fcp)}, α_occ={_fmt(alpha_occ)})"
                )

            # --- emit the 0/1 stream ---
            if emit_binary_stdout:
                print(bit, flush=True)



            time.sleep(online_hop_seconds)

    except KeyboardInterrupt:
        print("\n[ONLINE] Stopped by user.")

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

