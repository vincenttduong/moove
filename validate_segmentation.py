"""
MOOVE Segmentation Validation Script
======================================
Generates THREE figures for quickly eyeballing segmentation quality, all
showing each sampled syllable IN ITS ORIGINAL CONTEXT -- i.e. with real
signal shown well before onset and well after offset, not just a small
fixed pad -- using a common, duration-independent window shared across
every row so the figures stack into clean rectangular matrices:

  1. Energy heatmap: each segment's power-vs-time trace (spectrogram
     collapsed across frequency), stacked as rows of a single 2D heatmap
     image (imshow), sorted by increasing duration, onset-aligned so every
     row's true onset is at the same x-pixel.
  2. Energy line plot: same data as (1), as individual line traces with an
     offset marker, for cases where the line-plot level of detail is
     preferred over the heatmap.
  3. Spectrogram heatmap: each segment's FULL spectrogram (not collapsed
     across frequency), frequency axis compressed (binned/averaged) so
     many stacked segments still fit in a reasonable figure height.

WINDOWING (this is the part that shows original context around each
syllable, rather than just the syllable + a small pad):

  - LEFT_EXTENSION_SECONDS: how far before onset (t=0) to show, for every
    row. This is NOT capped by that row's own duration -- it's real signal
    preceding the segment, extended by a user-chosen amount.
  - RIGHT_EXTENSION_SECONDS: extends past the OFFSET OF THE LONGEST SAMPLED
    SYLLABLE, not past each row's own offset. Every row shares this exact
    same right boundary (t = max_duration + RIGHT_EXTENSION_SECONDS from
    onset), so a short syllable's window extends well past its own offset
    to match the longest syllable's window length -- this is what keeps
    the stacked matrix rectangular while showing every row a consistent
    amount of context.

    Worked example: syllable 1 duration=0.05s, syllable 2 (longest)
    duration=0.10s, RIGHT_EXTENSION_SECONDS=0.05. Both syllables' windows
    extend to t=0.10+0.05=0.15 from their own onset (t=0), even though
    syllable 1's own offset is only at t=0.05.

  - If a row's window (using its own onset/offset in the original
    recording) would extend before the start or past the end of that
    recording, it is truncated at the recording boundary and the missing
    portion is padded with the same "no data" gray fill used elsewhere
    (NaN under the colormap's set_bad color) -- never zero, since zero
    could be misread as real low-energy signal.

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

# How far before onset (t=0) to show, for every sampled syllable, to
# reveal original signal context preceding the segment.
LEFT_EXTENSION_SECONDS = 0.1

# How far past the OFFSET OF THE LONGEST SAMPLED SYLLABLE (not each row's
# own offset) every row's shared right window boundary extends. See the
# worked example in the module docstring above.
RIGHT_EXTENSION_SECONDS = 0.1

# Number of frequency bins to compress the full spectrogram down to for
# figure 3 (averaged/binned from the native NFFT-derived frequency axis).
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


def load_audio_and_compute_windowed_spectrogram(wav_path, onset, win_start_t, win_end_t,
                                                 config, nperseg, noverlap, nfft, freq_cutoffs,
                                                 verbose_errors=True):
    """Load raw audio for the ABSOLUTE window [onset + win_start_t, onset + win_end_t]
    (i.e. win_start_t/win_end_t are already relative to onset, negative
    win_start_t means "before onset"), clipped to the recording's actual
    [0, recording_duration] bounds, compute its spectrogram restricted to
    freq_cutoffs, and return time aligned so t=0 is still the true onset
    (even though the window may have been clipped at a recording
    boundary -- clipping only shortens how much of the requested window is
    actually shown for that row, it never shifts the onset reference).

    Returns a dict with keys: t, power_db, f, spectrogram_db (2D, freq x
    time), sampling_rate, clipped_left (bool), clipped_right (bool) -- or
    None on failure.
    """
    file_path = {"file_name": os.path.basename(wav_path), "file_path": wav_path}
    try:
        display_dict = get_display_data(file_path, config)
    except Exception as e:
        if verbose_errors:
            print(f"  FAIL [{os.path.basename(wav_path)} @ onset={onset:.3f}s]: "
                  f"get_display_data() raised: {e}")
        return None

    sampling_rate = int(display_dict["sampling_rate"])
    rawsong = display_dict["song_data"]
    recording_duration = len(rawsong) / sampling_rate

    requested_abs_start = onset + win_start_t
    requested_abs_end = onset + win_end_t

    abs_start = max(0.0, requested_abs_start)
    abs_end = min(recording_duration, requested_abs_end)
    clipped_left = abs_start > requested_abs_start
    clipped_right = abs_end < requested_abs_end

    start_idx = int(abs_start * sampling_rate)
    end_idx = min(int(abs_end * sampling_rate), len(rawsong))

    if start_idx >= end_idx:
        if verbose_errors:
            print(f"  FAIL [{os.path.basename(wav_path)} @ onset={onset:.3f}s]: "
                  f"requested window entirely outside recording bounds "
                  f"(recording_duration={recording_duration:.3f}s).")
        return None

    windowed_audio = rawsong[start_idx:end_idx]
    if len(windowed_audio) < nperseg:
        if verbose_errors:
            print(f"  FAIL [{os.path.basename(wav_path)} @ onset={onset:.3f}s]: "
                  f"windowed_audio length ({len(windowed_audio)}) < nperseg ({nperseg}) "
                  f"after clipping to recording bounds.")
        return None

    f, t, Sxx = spectrogram(windowed_audio, fs=sampling_rate,
                            nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    freq_mask = (f >= freq_cutoffs[0]) & (f <= freq_cutoffs[1])
    f = f[freq_mask]
    Sxx = Sxx[freq_mask, :]
    if Sxx.shape[0] == 0:
        if verbose_errors:
            print(f"  FAIL [{os.path.basename(wav_path)} @ onset={onset:.3f}s]: "
                  f"no frequency bins in range {freq_cutoffs} Hz.")
        return None

    spectrogram_db = decibel(Sxx)
    power_db = decibel(Sxx.sum(axis=0))
    # t is relative to abs_start; shift so t=0 is the true onset.
    t_aligned = t - (onset - abs_start)

    return {
        "t": t_aligned, "power_db": power_db, "f": f,
        "spectrogram_db": spectrogram_db, "sampling_rate": sampling_rate,
        "clipped_left": clipped_left, "clipped_right": clipped_right,
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


def build_padded_matrix(items, n_time_bins, dt, left_extension_seconds, is_2d):
    """Right/left-pad a list of (variable-width, possibly recording-boundary
    -clipped) traces onto a common time axis of length n_time_bins, where
    column 0 corresponds to t = -left_extension_seconds (the shared left
    edge of every row's window) and t=0 (true onset) therefore always
    lands at the same fixed column for every row. Padding uses NaN so it
    renders as a distinct background color, never mistaken for real
    (zero-energy) signal -- this covers both recording-boundary clipping
    and the fact that shorter syllables' windows are shorter than the
    shared right boundary defined by the longest sampled syllable.

    is_2d=True expects each item's data to be (n_freq_bins, n_time_native)
    (for the spectrogram figure); is_2d=False expects 1D power traces.
    """
    n_rows = len(items)
    if is_2d:
        n_freq_bins = items[0][0].shape[0]
        matrix = np.full((n_rows, n_freq_bins, n_time_bins), np.nan, dtype=float)
    else:
        matrix = np.full((n_rows, n_time_bins), np.nan, dtype=float)

    for row_idx, (data, t) in enumerate(items):
        n_cols = data.shape[1] if is_2d else len(data)
        if n_cols == 0:
            continue

        col_start = int(round((t[0] + left_extension_seconds) / dt))
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

    max_duration = max(seg["offset"] - seg["onset"] for seg in sampled_segments)
    # Shared window (relative to each row's own onset) applied to EVERY row:
    #   left edge  = -LEFT_EXTENSION_SECONDS
    #   right edge = max_duration + RIGHT_EXTENSION_SECONDS
    # This is what keeps the final matrix rectangular while showing every
    # syllable a consistent, duration-independent amount of context.
    win_start_t = -LEFT_EXTENSION_SECONDS
    win_end_t = max_duration + RIGHT_EXTENSION_SECONDS
    print(f"Longest sampled syllable duration: {max_duration * 1000:.1f}ms. "
          f"Shared window per row: [onset {win_start_t * 1000:+.1f}ms, "
          f"onset {win_end_t * 1000:+.1f}ms] "
          f"(width={ (win_end_t - win_start_t) * 1000:.1f}ms).")

    print("Extracting onset-aligned traces and spectrograms for sampled segments...")
    entries = []
    n_printed_failures = 0
    n_clipped = 0
    MAX_PRINTED_FAILURES = 10
    for i, seg in enumerate(sampled_segments):
        verbose = n_printed_failures < MAX_PRINTED_FAILURES
        result = load_audio_and_compute_windowed_spectrogram(
            seg["wav_path"], seg["onset"], win_start_t, win_end_t, config,
            NPERSEG, NOVERLAP, NFFT, FREQ_CUTOFFS, verbose_errors=verbose,
        )
        if result is None:
            n_printed_failures += 1
            continue
        if result["clipped_left"] or result["clipped_right"]:
            n_clipped += 1
        result["duration"] = seg["offset"] - seg["onset"]
        result["offset_rel_to_onset"] = result["duration"]
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
    if n_clipped > 0:
        print(f"NOTE: {n_clipped}/{len(entries)} segments had their requested window "
              f"clipped at a recording boundary (start or end of that .wav file); "
              f"the missing portion is padded gray in the figures.")

    if not entries:
        print("ERROR: no segments could be successfully processed.")
        return

    entries.sort(key=lambda e: e["duration"])
    n = len(entries)
    dt = entries[0]["t"][1] - entries[0]["t"][0] if len(entries[0]["t"]) > 1 else 0.001
    n_time_bins = int(np.ceil((win_end_t - win_start_t) / dt))

    print(f"Building padded matrices: {n} rows x {n_time_bins} time bins...")

    # ------------------------------------------------------------------
    # Figure 1: energy heatmap.
    # ------------------------------------------------------------------
    energy_matrix = build_padded_matrix(
        [(e["power_db"], e["t"]) for e in entries], n_time_bins, dt,
        LEFT_EXTENSION_SECONDS, is_2d=False)

    fig_h1 = max(4, min(0.04 * n, 20))
    fig1, ax1 = plt.subplots(figsize=(10, fig_h1))
    cmap1 = plt.get_cmap("viridis").copy()
    cmap1.set_bad(color="lightgray")
    extent = [win_start_t, win_start_t + n_time_bins * dt, n, 0]
    im1 = ax1.imshow(np.ma.masked_invalid(energy_matrix), aspect="auto",
                     cmap=cmap1, extent=extent, interpolation="nearest")
    ax1.axvline(0, color="white", linewidth=0.8, alpha=0.7)
    for row_idx, e in enumerate(entries):
        ax1.plot(e["offset_rel_to_onset"], row_idx + 0.5, marker="v",
                 color="red", markersize=3, linestyle="none")
    ax1.set_xlabel("Time relative to segment onset (s)")
    ax1.set_ylabel("Segment (sorted by increasing duration, top to bottom)")
    ax1.set_title(f"Segmentation Validation: Energy Heatmap ({n} segments)\n"
                  f"(white line = onset; red \u25bd = offset; gray = padding/recording boundary; "
                  f"window: onset {win_start_t*1000:+.0f}ms to {win_end_t*1000:+.0f}ms)")
    fig1.colorbar(im1, ax=ax1, label="Power (dB)", fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_HEATMAP_PATH, dpi=150)
    print(f"Saved energy heatmap to {OUTPUT_HEATMAP_PATH}")

    # ------------------------------------------------------------------
    # Figure 2: energy line plot.
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
    ax2.set_xlim(win_start_t, win_end_t)
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
        [(binned_specs[i], entries[i]["t"]) for i in range(n)], n_time_bins, dt,
        LEFT_EXTENSION_SECONDS, is_2d=True)

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
    extent3 = [win_start_t, win_start_t + n_time_bins * dt, total_rows, 0]
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
                  f"(white line = onset; cyan \u25bd = offset; gray = padding/recording boundary)")
    fig3.colorbar(im3, ax=ax3, label="Power (dB)", fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_SPECTROGRAM_PATH, dpi=150)
    print(f"Saved stacked spectrogram figure to {OUTPUT_SPECTROGRAM_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
