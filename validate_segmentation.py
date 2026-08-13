"""
MOOVE Segmentation Validation Script
======================================
Generates THREE figures for quickly eyeballing segmentation quality, all
using the same pooled/sampled/onset-aligned/duration-sorted segment set:

  1. Energy heatmap: each segment's power-vs-time trace (spectrogram
     collapsed across frequency), stacked as rows of a single 2D heatmap
     image (imshow), sorted by increasing duration, onset-aligned so every
     row's true onset is at the same x-pixel. Vertically condensed vs. a
     line-per-row plot.
  2. Energy line plot (original style): same data as (1), but as
     individual line traces with an offset marker, kept for cases where
     the line-plot level of detail is preferred over a heatmap.
  3. Spectrogram heatmap: each segment's FULL spectrogram (not collapsed
     across frequency), frequency axis compressed (binned/averaged) so
     many stacked segments still fit in a reasonable figure height, laid
     out as one tall image with segments stacked vertically, onset-aligned
     and duration-sorted identically to (1).

To stack into single rectangular images, every segment's trace/spectrogram
is right-padded (not truncated) out to a common width of
    RIGHT_PAD_TARGET = 1.1 * max_sampled_duration + PAD_SECONDS
measured from onset (t=0), so the widest realistic segment plus its
context pad defines the matrix width and everything shorter is padded with
a distinct "no data" fill (NaN, rendered as a fixed background color) --
never zero, since zero could be misread as real low-energy signal. This
padding is for DISPLAY ONLY: it does not affect clustering or any other
downstream analysis, only how this validation figure is rendered.

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
from matplotlib.colors import Normalize
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

# Number of frequency bins to compress the full spectrogram down to for
# figure 3 (averaged/binned from the native NFFT-derived frequency axis).
# Keeps stacked spectrogram figure height reasonable regardless of how
# many segments are sampled.
SPECTROGRAM_FREQ_BINS = 24

RANDOM_SEED = 42
OUTPUT_HEATMAP_PATH = "segmentation_validation_heatmap.png"
OUTPUT_LINEPLOT_PATH = "segmentation_validation_lineplot.png"
OUTPUT_SPECTROGRAM_PATH = "segmentation_validation_spectrograms.png"

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
    underlying .not.mat format (confirmed in MATLAB against this dataset's
    .not.mat files) -- converted to seconds here, once, at the source.
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
    onset/offset pair within that file's .not.mat. Returns a flat list of
    dicts: {wav_path, onset, offset, day}, with no label filtering."""
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
    so any units/path mismatch is immediately visible."""
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


def extract_segment_data(wav_path, onset, offset, config,
                         nperseg, noverlap, nfft, freq_cutoffs,
                         pad_seconds, verbose_errors=True):
    """Load raw audio spanning [onset - pad_seconds, offset + pad_seconds],
    compute its spectrogram restricted to freq_cutoffs, and return BOTH the
    frequency-collapsed 1D power trace AND the full 2D spectrogram, so
    figures 1/2 and figure 3 reuse a single audio load + spectrogram call
    per segment rather than computing it twice.

    Time (t) and the spectrogram's time axis are ALIGNED TO ONSET: t=0 is
    always the true onset.

    Returns a dict with keys: t, power_db, offset_rel_to_onset, f,
    spectrogram_db (2D, freq x time), sampling_rate -- or None on failure.
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
                  f"start_idx ({start_idx}) >= end_idx ({end_idx}).")
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
    f = f[freq_mask]
    Sxx = Sxx[freq_mask, :]
    if Sxx.shape[0] == 0:
        if verbose_errors:
            print(f"  FAIL [{os.path.basename(wav_path)} @ {onset:.3f}-{offset:.3f}s]: "
                  f"no frequency bins in range {freq_cutoffs} Hz.")
        return None

    spectrogram_db = decibel(Sxx)
    power_db = decibel(Sxx.sum(axis=0))
    t_aligned = t - (onset - win_start)
    offset_rel_to_onset = offset - onset

    return {
        "t": t_aligned, "power_db": power_db,
        "offset_rel_to_onset": offset_rel_to_onset,
        "f": f, "spectrogram_db": spectrogram_db,
        "sampling_rate": sampling_rate,
    }


def bin_frequency_axis(spectrogram_db, n_bins):
    """Average-pool a (n_freq, n_time) spectrogram down to (n_bins, n_time)
    along the frequency axis, so stacking many segments' spectrograms
    vertically still fits a reasonable total figure height."""
    n_freq = spectrogram_db.shape[0]
    if n_bins >= n_freq:
        return spectrogram_db
    edges = np.linspace(0, n_freq, n_bins + 1).astype(int)
    binned = np.zeros((n_bins, spectrogram_db.shape[1]), dtype=spectrogram_db.dtype)
    for i in range(n_bins):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        binned[i, :] = spectrogram_db[lo:hi, :].mean(axis=0)
    return binned


def build_padded_matrix(traces_2d_or_1d, n_time_bins, dt, is_2d):
    """Right-pad a list of (variable-width) traces onto a common time axis
    of length n_time_bins (sampled at spacing dt, starting at the shared
    onset-aligned t=0 reference, i.e. col 0 corresponds to whichever t is
    closest to each row's own window start). Padding uses NaN so it can be
    rendered as a distinct background color, never mistaken for real
    (zero-energy) signal.

    is_2d=True expects each item to itself be (n_freq_bins, n_time_native)
    (for the spectrogram figure); is_2d=False expects 1D power traces.
    Returns a 2D matrix (n_rows, n_time_bins) if is_2d=False, or a 3D
    array (n_rows, n_freq_bins, n_time_bins) if is_2d=True.
    """
    n_rows = len(traces_2d_or_1d)
    if is_2d:
        n_freq_bins = traces_2d_or_1d[0][0].shape[0]
        matrix = np.full((n_rows, n_freq_bins, n_time_bins), np.nan, dtype=float)
    else:
        matrix = np.full((n_rows, n_time_bins), np.nan, dtype=float)

    for row_idx, item in enumerate(traces_2d_or_1d):
        data, t = item
        n_cols = data.shape[1] if is_2d else len(data)

        col_start = int(round((t[0] + PAD_SECONDS_GLOBAL) / dt)) if n_cols else 0
        col_start = max(0, col_start)
        col_end = min(n_time_bins, col_start + n_cols)
        n_fit = col_end - col_start
        if n_fit <= 0:
            continue

        if is_2d:
            matrix[row_idx, :, col_start:col_end] = data[:, :n_fit]
        else:
            matrix[row_idx, col_start:col_end] = data[:n_fit]

    return matrix


def main():
    global PAD_SECONDS_GLOBAL
    PAD_SECONDS_GLOBAL = PAD_SECONDS

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
          f"of total).")

    print("Extracting onset-aligned traces and spectrograms for sampled segments...")
    entries = []
    n_printed_failures = 0
    MAX_PRINTED_FAILURES = 10
    for i, seg in enumerate(sampled_segments):
        verbose = n_printed_failures < MAX_PRINTED_FAILURES
        result = extract_segment_data(
            seg["wav_path"], seg["onset"], seg["offset"], config,
            NPERSEG, NOVERLAP, NFFT, FREQ_CUTOFFS, PAD_SECONDS,
            verbose_errors=verbose,
        )
        if result is None:
            n_printed_failures += 1
            continue
        result["duration"] = seg["offset"] - seg["onset"]
        result["day"] = seg["day"]
        entries.append(result)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_sample} segments...")

    n_failed = n_sample - len(entries)
    if n_failed > 0:
        print(f"WARNING: {n_failed}/{n_sample} sampled segments failed to load "
              f"and were excluded."
              + (f" (only first {MAX_PRINTED_FAILURES} failures printed above)"
                 if n_failed > MAX_PRINTED_FAILURES else ""))

    if not entries:
        print("ERROR: no segments could be successfully processed.")
        return

    entries.sort(key=lambda e: e["duration"])
    n = len(entries)
    max_duration = max(e["duration"] for e in entries)
    right_pad_target = 1.1 * max_duration + PAD_SECONDS
    dt = entries[0]["t"][1] - entries[0]["t"][0] if len(entries[0]["t"]) > 1 else 0.001
    n_time_bins = int(np.ceil((PAD_SECONDS + right_pad_target) / dt))

    print(f"Building padded matrices: {n} rows x {n_time_bins} time bins "
          f"(right-padded to 1.1*max_duration + pad = {right_pad_target * 1000:.1f}ms "
          f"from onset)...")

    # ------------------------------------------------------------------
    # Figure 1: energy heatmap (frequency-collapsed power trace per row,
    # stacked into one 2D image rather than individually-drawn lines).
    # ------------------------------------------------------------------
    energy_matrix = build_padded_matrix(
        [(e["power_db"], e["t"]) for e in entries], n_time_bins, dt, is_2d=False)

    fig_h1 = max(4, min(0.04 * n, 20))
    fig1, ax1 = plt.subplots(figsize=(10, fig_h1))
    cmap1 = plt.get_cmap("viridis").copy()
    cmap1.set_bad(color="lightgray")
    extent = [-PAD_SECONDS, n_time_bins * dt - PAD_SECONDS, n, 0]
    im1 = ax1.imshow(np.ma.masked_invalid(energy_matrix), aspect="auto",
                     cmap=cmap1, extent=extent, interpolation="nearest")
    ax1.axvline(0, color="white", linewidth=0.8, alpha=0.7)
    for row_idx, e in enumerate(entries):
        ax1.plot(e["offset_rel_to_onset"], row_idx + 0.5, marker="v",
                 color="red", markersize=3, linestyle="none")
    ax1.set_xlabel("Time relative to segment onset (s)")
    ax1.set_ylabel("Segment (sorted by increasing duration, top to bottom)")
    ax1.set_title(f"Segmentation Validation: Energy Heatmap ({n} segments)\n"
                  f"(white line = onset; red \u25bd = offset; gray = padding)")
    fig1.colorbar(im1, ax=ax1, label="Power (dB)", fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_HEATMAP_PATH, dpi=150)
    print(f"Saved energy heatmap to {OUTPUT_HEATMAP_PATH}")

    # ------------------------------------------------------------------
    # Figure 2: energy line plot (original style, kept alongside heatmap).
    # ------------------------------------------------------------------
    fig_h2 = max(6, min(0.12 * n, 60))
    fig2, ax2 = plt.subplots(figsize=(10, fig_h2))
    row_gap, trace_top = 1.0, 0.9
    for row_idx, e in enumerate(entries):
        power = e["power_db"]
        power_norm = power - np.min(power)
        max_range = np.max(power_norm) if np.max(power_norm) > 0 else 1.0
        power_norm = power_norm / max_range * trace_top
        y_off = row_idx * row_gap
        ax2.plot(e["t"], power_norm + y_off, color="black", linewidth=0.6)
        ax2.plot([0, 0], [y_off, y_off + trace_top], color="tab:green", linewidth=0.8, zorder=3)
        ax2.plot(e["offset_rel_to_onset"], y_off + trace_top + 0.06, marker="v",
                 color="tab:red", markersize=4, linestyle="none", zorder=3)
    ax2.axvline(0, color="tab:green", linewidth=0.5, alpha=0.25, zorder=1)
    min_t = min(e["t"][0] if len(e["t"]) else 0 for e in entries)
    max_t = max(e["t"][-1] if len(e["t"]) else 0 for e in entries)
    ax2.set_xlim(min_t, max_t + 0.05)
    ax2.set_ylim(-0.5, n * row_gap)
    ax2.set_xlabel("Time relative to segment onset (s)")
    ax2.set_ylabel("Segment (sorted by increasing duration, bottom to top)")
    ax2.set_title(f"Segmentation Validation: Energy Line Plot ({n} segments, aligned to onset)\n"
                  f"(green line = onset (t=0); red \u25bd = offset)")
    ax2.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUTPUT_LINEPLOT_PATH, dpi=150)
    print(f"Saved energy line plot to {OUTPUT_LINEPLOT_PATH}")

    # ------------------------------------------------------------------
    # Figure 3: full spectrograms, frequency-compressed, stacked.
    # ------------------------------------------------------------------
    print(f"Binning frequency axis to {SPECTROGRAM_FREQ_BINS} bins per segment...")
    binned_specs = [bin_frequency_axis(e["spectrogram_db"], SPECTROGRAM_FREQ_BINS) for e in entries]
    spec_matrix_3d = build_padded_matrix(
        [(binned_specs[i], entries[i]["t"]) for i in range(n)], n_time_bins, dt, is_2d=True)

    # Stack rows vertically: each segment occupies SPECTROGRAM_FREQ_BINS
    # pixel-rows of the final image, separated by a thin NaN gap row so
    # segment boundaries remain visible.
    gap_rows = 1
    total_rows = n * (SPECTROGRAM_FREQ_BINS + gap_rows)
    stacked_spec = np.full((total_rows, n_time_bins), np.nan, dtype=float)
    for row_idx in range(n):
        start_row = row_idx * (SPECTROGRAM_FREQ_BINS + gap_rows)
        stacked_spec[start_row:start_row + SPECTROGRAM_FREQ_BINS, :] = spec_matrix_3d[row_idx]

    fig_h3 = max(6, min(0.02 * total_rows, 60))
    fig3, ax3 = plt.subplots(figsize=(10, fig_h3))
    cmap3 = plt.get_cmap("magma").copy()
    cmap3.set_bad(color="lightgray")
    extent3 = [-PAD_SECONDS, n_time_bins * dt - PAD_SECONDS, total_rows, 0]
    im3 = ax3.imshow(np.ma.masked_invalid(stacked_spec), aspect="auto",
                     cmap=cmap3, extent=extent3, interpolation="nearest")
    ax3.axvline(0, color="white", linewidth=0.8, alpha=0.7)
    for row_idx in range(n):
        row_center = row_idx * (SPECTROGRAM_FREQ_BINS + gap_rows) + SPECTROGRAM_FREQ_BINS / 2
        ax3.plot(entries[row_idx]["offset_rel_to_onset"], row_center, marker="v",
                 color="cyan", markersize=3, linestyle="none")
    ax3.set_xlabel("Time relative to segment onset (s)")
    ax3.set_ylabel(f"Segment (sorted by increasing duration, top to bottom)\n"
                   f"[{SPECTROGRAM_FREQ_BINS} freq bins/segment, {FREQ_CUTOFFS[0]}-{FREQ_CUTOFFS[1]} Hz]")
    ax3.set_yticks([])
    ax3.set_title(f"Segmentation Validation: Full Spectrograms ({n} segments)\n"
                  f"(white line = onset; cyan \u25bd = offset; gray = padding)")
    fig3.colorbar(im3, ax=ax3, label="Power (dB)", fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_SPECTROGRAM_PATH, dpi=150)
    print(f"Saved stacked spectrogram figure to {OUTPUT_SPECTROGRAM_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
