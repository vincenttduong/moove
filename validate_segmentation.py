"""
MOOVE Segmentation Validation Script
======================================
Generates a single figure for quickly eyeballing segmentation quality: every
extracted syllable segment's power-vs-time trace ("spectral sum": the
spectrogram collapsed across frequency, giving one energy value per time
bin), stacked vertically in order of increasing duration, all aligned so
that t=0 is that segment's true onset. Offsets are marked with a distinct,
easy-to-track marker (a red triangle sitting just above each trace, at that
row's true offset position) so you can visually trace the "staircase" of
offsets across rows and spot outliers.

Traces are NOT padded/truncated to a common length: each trace's width
reflects its own true duration (onset to offset, plus a small fixed pad on
each side for context), so the actual spread of syllable durations remains
visible in the figure.

MOOVE folder structure (as used by moove.utils.AppState._get_batch_files):

    <bird_root>/
        <experiment>/
            <day_1>/
                <recording_1>.wav
                <recording_1>.wav.not.mat
                ...
                batch.txt
            <day_2>/
                ...

Requires the `moove` package to be installed so it can reuse
evfuncs.load_notmat, moove.utils.get_display_data, and moove.utils.decibel.

Usage: fill in the USER-MODIFIABLE VARIABLES section below, then run:
    python validate_segmentation.py
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from moove.utils import get_display_data, decibel

# ======================================================================
# USER-MODIFIABLE VARIABLES
# ======================================================================

BIRD_ROOT = r"PATH_TO_BIRD_ROOT"
EXPERIMENT = "EXPERIMENT_FOLDER_NAME"
DAYS = ["DAY_FOLDER_1", "DAY_FOLDER_2"]
SAMPLE_FRACTION = 0.1

NPERSEG = 1024
NOVERLAP = 896
NFFT = 1024
FREQ_CUTOFFS = (500, 10000)  # (low_hz, high_hz)
PAD_SECONDS = 0.02
RANDOM_SEED = 42
OUTPUT_FIGURE_PATH = "segmentation_validation.png"

# ======================================================================
# IMPLEMENTATION
# ======================================================================


def find_wav_files(day_path):
    """Return sorted list of .wav file paths directly inside a day folder."""
    return sorted(
        os.path.join(day_path, f)
        for f in os.listdir(day_path)
        if f.lower().endswith(".wav")
    )


def load_notmat_segments(wav_path):
    """Return (onsets, offsets) IN SECONDS for a .wav file's sibling
    .not.mat, or (None, None) if no .not.mat exists / it has no segments.

    evfuncs.load_notmat's onsets/offsets are stored in MILLISECONDS in the
    underlying .not.mat format (confirmed directly in MATLAB against this
    dataset's .not.mat files -- the EvTAF/evsonganaly convention) -- they
    are converted to seconds here, once, at the source, so every
    downstream consumer of this function can assume seconds.
    """
    import evfuncs

    notmat_path = wav_path + ".not.mat"
    if not os.path.exists(notmat_path):
        return None, None
    notmat_dict = evfuncs.load_notmat(notmat_path)
    onsets = notmat_dict.get("onsets", [])
    offsets = notmat_dict.get("offsets", [])
    if len(onsets) == 0 or len(offsets) == 0:
        return None, None
    onsets_sec = np.asarray(onsets, dtype=float) / 1000.0
    offsets_sec = np.asarray(offsets, dtype=float) / 1000.0
    return onsets_sec, offsets_sec


def gather_all_segments(bird_root, experiment, days):
    """Walk every day folder, every .wav file within it, and every
    onset/offset pair within that file's .not.mat (already converted to
    seconds by load_notmat_segments). Returns a flat list of dicts:
    {wav_path, onset, offset, day}, with no label filtering."""
    all_segments = []
    for day in days:
        day_path = os.path.join(bird_root, experiment, day)
        if not os.path.isdir(day_path):
            print(f"WARNING: day folder not found, skipping: {day_path}")
            continue
        wav_files = find_wav_files(day_path)
        if not wav_files:
            print(f"WARNING: no .wav files found in {day_path}")
            continue
        for wav_path in wav_files:
            onsets, offsets = load_notmat_segments(wav_path)
            if onsets is None:
                continue
            n = min(len(onsets), len(offsets))
            for i in range(n):
                all_segments.append({
                    "wav_path": wav_path,
                    "onset": float(onsets[i]),
                    "offset": float(offsets[i]),
                    "day": day,
                })
    return all_segments


def run_diagnostic_check(all_segments, config):
    """Print raw numbers for a few segments up front, before the full run,
    so any units/path mismatch is immediately visible rather than causing
    silent failures with no clue why. Checks segment duration sanity and
    whether onset/offset actually fall within the recording's audio length
    once loaded."""
    print("\n--- Diagnostic check on first 3 segments ---")
    for seg in all_segments[:3]:
        duration = seg["offset"] - seg["onset"]
        print(f"  {os.path.basename(seg['wav_path'])}: "
              f"onset={seg['onset']:.4f}s offset={seg['offset']:.4f}s "
              f"duration={duration * 1000:.1f}ms")
        try:
            file_path = {"file_name": os.path.basename(seg["wav_path"]),
                        "file_path": seg["wav_path"]}
            display_dict = get_display_data(file_path, config)
            sr = int(display_dict["sampling_rate"])
            recording_duration = len(display_dict["song_data"]) / sr
            print(f"    -> recording: sampling_rate={sr} Hz, "
                  f"duration={recording_duration:.3f}s, "
                  f"onset_within_recording={seg['onset'] <= recording_duration}, "
                  f"offset_within_recording={seg['offset'] <= recording_duration}")
        except Exception as e:
            print(f"    -> could not load recording for diagnostic: {e}")
    print("--- End diagnostic check ---\n")


def extract_segment_spectral_sum(wav_path, onset, offset, config,
                                  nperseg, noverlap, nfft, freq_cutoffs,
                                  pad_seconds, verbose_errors=True):
    """Load raw audio spanning [onset - pad_seconds, offset + pad_seconds],
    compute its spectrogram, restrict to freq_cutoffs, and collapse across
    frequency (sum) to get a 1D power-vs-time trace in dB. Time is returned
    ALIGNED TO ONSET: t=0 always corresponds to the true onset.

    Returns (t_aligned_to_onset, power_trace_db, offset_rel_to_onset,
    sampling_rate) on success, or None on failure. When verbose_errors is
    True, every failure path prints exactly why it failed.
    """
    file_path = {"file_name": os.path.basename(wav_path), "file_path": wav_path}
    try:
        display_dict = get_display_data(file_path, config)
    except Exception as e:
        if verbose_errors:
            print(f"  FAIL [{os.path.basename(wav_path)} @ {onset:.3f}-{offset:.3f}s]: "
                  f"get_display_data() raised: {e}")
        return None

    sampling_rate = int(display_dict["sampling_rate"])
    rawsong = display_dict["song_data"]

    win_start = max(0.0, onset - pad_seconds)
    win_end = offset + pad_seconds
    start_idx = int(win_start * sampling_rate)
    end_idx = int(win_end * sampling_rate)
    end_idx = min(end_idx, len(rawsong))

    if start_idx >= end_idx:
        if verbose_errors:
            print(f"  FAIL [{os.path.basename(wav_path)} @ {onset:.3f}-{offset:.3f}s]: "
                  f"start_idx ({start_idx}) >= end_idx ({end_idx}). "
                  f"len(rawsong)={len(rawsong)}, sampling_rate={sampling_rate}.")
        return None

    windowed_audio = rawsong[start_idx:end_idx]
    if len(windowed_audio) < nperseg:
        if verbose_errors:
            print(f"  FAIL [{os.path.basename(wav_path)} @ {onset:.3f}-{offset:.3f}s]: "
                  f"windowed_audio length ({len(windowed_audio)}) < nperseg ({nperseg}).")
        return None

    f, t, Sxx = spectrogram(windowed_audio, fs=sampling_rate,
                            nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    freq_mask = (f >= freq_cutoffs[0]) & (f <= freq_cutoffs[1])
    Sxx = Sxx[freq_mask, :]
    if Sxx.shape[0] == 0:
        if verbose_errors:
            print(f"  FAIL [{os.path.basename(wav_path)} @ {onset:.3f}-{offset:.3f}s]: "
                  f"no frequency bins in range {freq_cutoffs} Hz "
                  f"(f ranges {f.min():.1f}-{f.max():.1f} Hz at sampling_rate={sampling_rate}).")
        return None

    power_trace_db = decibel(Sxx.sum(axis=0))
    t_aligned = t - (onset - win_start)
    offset_rel_to_onset = offset - onset

    return t_aligned, power_trace_db, offset_rel_to_onset, sampling_rate


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    print(f"Gathering segments from {len(DAYS)} day folder(s)...")
    all_segments = gather_all_segments(BIRD_ROOT, EXPERIMENT, DAYS)
    total_found = len(all_segments)
    print(f"Found {total_found} total segments across all specified days.")

    if total_found == 0:
        print("ERROR: no segments found. Check BIRD_ROOT/EXPERIMENT/DAYS and "
              "that .not.mat files exist alongside the .wav files.")
        return

    config = {"global_dir": BIRD_ROOT}
    run_diagnostic_check(all_segments, config)

    n_sample = max(1, int(round(total_found * SAMPLE_FRACTION)))
    n_sample = min(n_sample, total_found)
    sampled_segments = random.sample(all_segments, n_sample)
    print(f"Randomly sampled {n_sample} segments ({100 * n_sample / total_found:.1f}% "
          f"of total) for validation figure.")

    print("Extracting onset-aligned spectral-sum traces for sampled segments...")
    traces = []
    n_printed_failures = 0
    MAX_PRINTED_FAILURES = 10  # cap verbose output once the pattern is clear
    for i, seg in enumerate(sampled_segments):
        verbose = n_printed_failures < MAX_PRINTED_FAILURES
        result = extract_segment_spectral_sum(
            seg["wav_path"], seg["onset"], seg["offset"], config,
            NPERSEG, NOVERLAP, NFFT, FREQ_CUTOFFS, PAD_SECONDS,
            verbose_errors=verbose,
        )
        if result is None:
            n_printed_failures += 1
            continue
        t_aligned, power_trace_db, offset_rel_to_onset, sampling_rate = result
        duration = seg["offset"] - seg["onset"]
        traces.append({
            "t": t_aligned, "power": power_trace_db,
            "offset_rel_to_onset": offset_rel_to_onset,
            "duration": duration, "day": seg["day"],
        })
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_sample} segments...")

    n_failed = n_sample - len(traces)
    if n_failed > 0:
        print(f"WARNING: {n_failed}/{n_sample} sampled segments failed to load "
              f"and were excluded from the figure."
              + (f" (only first {MAX_PRINTED_FAILURES} failures printed above)"
                 if n_failed > MAX_PRINTED_FAILURES else ""))

    if not traces:
        print("ERROR: no segments could be successfully processed.")
        return

    traces.sort(key=lambda tr: tr["duration"])

    print(f"Rendering figure with {len(traces)} segments, sorted by duration, "
          f"aligned to onset...")
    n = len(traces)
    min_t = min(tr["t"][0] if len(tr["t"]) else 0 for tr in traces)
    max_t = max(tr["t"][-1] if len(tr["t"]) else 0 for tr in traces)

    fig_height = max(6, min(0.12 * n, 60))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    row_gap = 1.0
    trace_top = 0.9
    for row_idx, tr in enumerate(traces):
        power = tr["power"]
        power_norm = power - np.min(power)
        max_range = np.max(power_norm) if np.max(power_norm) > 0 else 1.0
        power_norm = power_norm / max_range * trace_top

        y_offset = row_idx * row_gap
        ax.plot(tr["t"], power_norm + y_offset, color="black", linewidth=0.6)
        ax.plot([0, 0], [y_offset, y_offset + trace_top],
               color="tab:green", linewidth=0.8, zorder=3)
        ax.plot(tr["offset_rel_to_onset"], y_offset + trace_top + 0.06,
               marker="v", color="tab:red", markersize=4,
               linestyle="none", zorder=3)

    ax.axvline(0, color="tab:green", linewidth=0.5, alpha=0.25, zorder=1)
    ax.set_xlim(min_t, max_t + 0.05)
    ax.set_ylim(-0.5, n * row_gap)
    ax.set_xlabel("Time relative to segment onset (s)")
    ax.set_ylabel("Segment (sorted by increasing duration, bottom to top)")
    ax.set_title(
        f"Segmentation Validation: {n} sampled segments, aligned to onset\n"
        f"(green line = onset (t=0); red \u25bd = offset; "
        f"\u00b1{PAD_SECONDS * 1000:.0f}ms context padding each side)"
    )
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_PATH, dpi=150)
    print(f"Saved figure to {OUTPUT_FIGURE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
