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
offsets across rows and spot outliers -- e.g. a segment whose offset marker
falls far from where its neighbors' do for a similar-looking trace, or a
trace whose energy clearly continues past its offset marker (a truncated
segment) or ends well before it (a segment capturing trailing silence).

Traces are NOT padded/truncated to a common length: each trace's width
reflects its own true duration (onset to offset, plus a small fixed pad on
each side for context), so the actual spread of syllable durations remains
visible in the figure -- which is the point of this validation step, since
duration is diagnostic of segmentation quality (missed/extra onsets,
merged/split syllables, etc. show up as duration outliers).

MOOVE folder structure (as used by moove.utils.AppState._get_batch_files):

    <bird_root>/
        <experiment>/
            <day_1>/
                <recording_1>.wav
                <recording_1>.wav.not.mat
                <recording_2>.wav
                <recording_2>.wav.not.mat
                ...
                batch.txt
            <day_2>/
                ...

Each day folder holds many short .wav recordings. Each recording has a
sibling `<recording>.wav.not.mat` file (evfuncs/EvTAF format) storing that
recording's segmentation as parallel `onsets`/`offsets` arrays, in seconds.
A cluster/training dataset is built by walking every .wav in a day folder,
reading its .not.mat, and treating every (onset, offset) pair as one
syllable segment -- this script does the same, without any label filtering,
since segmentation should be validated before syllable labels are assigned.

Requires the `moove` package to be installed (editable or otherwise) so it
can reuse evfuncs.load_notmat, moove.utils.get_display_data, and
moove.utils.decibel for audio/segment loading, exactly matching how MOOVE's
own cluster-dataset-creation step reads segments.

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

# 1) Root folder for the bird, e.g. r"F:\Data\bird1"
BIRD_ROOT = r"PATH_TO_BIRD_ROOT"

# 2) Experiment folder name (single subfolder of BIRD_ROOT), e.g. "pre_lesion"
EXPERIMENT = "EXPERIMENT_FOLDER_NAME"

# 3) List of experiment day folder names (subfolders of the experiment
#    folder) to pool segments from, e.g. ["day1", "day2", "day3"]
DAYS = ["DAY_FOLDER_1", "DAY_FOLDER_2"]

# 4) Fraction (0 < f <= 1) of the total pooled segments (across all DAYS
#    combined) to randomly sample for the figure. Sampling is done once,
#    consistently, across the pooled set -- not independently per day --
#    so the relative contribution of each day to the final figure reflects
#    that day's share of total segments.
SAMPLE_FRACTION = 0.1

# Spectrogram parameters (should match the values you use elsewhere in
# MOOVE's Cluster/Training dialogs for consistency, but can be changed).
NPERSEG = 1024
NOVERLAP = 896
NFFT = 1024
FREQ_CUTOFFS = (500, 10000)  # (low_hz, high_hz)

# Fixed context padding shown before onset and after offset in each trace's
# extraction window (does NOT equalize trace lengths -- every trace still
# has width = duration + 2*PAD_SECONDS, so true duration remains visible).
PAD_SECONDS = 0.02

# Random seed for reproducible sampling (set to None for a fresh sample
# every run).
RANDOM_SEED = 42

# Output figure path.
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
    """Return (onsets, offsets) in seconds for a .wav file's sibling
    .not.mat, or (None, None) if no .not.mat exists / it has no segments."""
    import evfuncs

    notmat_path = wav_path + ".not.mat"
    if not os.path.exists(notmat_path):
        return None, None
    notmat_dict = evfuncs.load_notmat(notmat_path)
    onsets = notmat_dict.get("onsets", [])
    offsets = notmat_dict.get("offsets", [])
    if len(onsets) == 0 or len(offsets) == 0:
        return None, None
    return onsets, offsets


def gather_all_segments(bird_root, experiment, days):
    """Walk every day folder, every .wav file within it, and every
    onset/offset pair within that file's .not.mat. Returns a flat list of
    dicts: {wav_path, onset, offset, day} for every syllable segment found,
    with no label filtering."""
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


def extract_segment_spectral_sum(wav_path, onset, offset, config,
                                  nperseg, noverlap, nfft, freq_cutoffs,
                                  pad_seconds):
    """Load raw audio spanning [onset - pad_seconds, offset + pad_seconds],
    compute its spectrogram, restrict to freq_cutoffs, and collapse across
    frequency (sum) to get a 1D power-vs-time trace in dB. Time is returned
    ALIGNED TO ONSET: t=0 always corresponds to the true onset, so t=-pad
    is the start of the window and t=(offset-onset) is the true offset,
    regardless of whether the window had to be clipped at the start of the
    recording (clipping only shortens the pre-onset context shown, it never
    shifts the onset/offset positions themselves).

    Returns (t_aligned_to_onset, power_trace_db, offset_rel_to_onset,
    sampling_rate) or None if loading/processing fails.
    """
    file_path = {"file_name": os.path.basename(wav_path), "file_path": wav_path}
    try:
        display_dict = get_display_data(file_path, config)
    except Exception as e:
        print(f"  Skipping segment in {os.path.basename(wav_path)}: failed to load audio ({e})")
        return None

    sampling_rate = int(display_dict["sampling_rate"])
    rawsong = display_dict["song_data"]

    win_start = max(0.0, onset - pad_seconds)
    win_end = offset + pad_seconds
    start_idx = int(win_start * sampling_rate)
    end_idx = int(win_end * sampling_rate)
    end_idx = min(end_idx, len(rawsong))
    if start_idx >= end_idx:
        return None

    windowed_audio = rawsong[start_idx:end_idx]
    if len(windowed_audio) < nperseg:
        return None

    f, t, Sxx = spectrogram(windowed_audio, fs=sampling_rate,
                            nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    freq_mask = (f >= freq_cutoffs[0]) & (f <= freq_cutoffs[1])
    Sxx = Sxx[freq_mask, :]
    if Sxx.shape[0] == 0:
        return None

    power_trace_db = decibel(Sxx.sum(axis=0))

    # t is relative to win_start; shift so t=0 is the true onset. If the
    # window was clipped at the recording start, win_start > onset - pad,
    # so this shift still lands exactly on the true onset -- only the
    # pre-onset context is shorter for that trace, never misaligned.
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

    n_sample = max(1, int(round(total_found * SAMPLE_FRACTION)))
    n_sample = min(n_sample, total_found)
    sampled_segments = random.sample(all_segments, n_sample)
    print(f"Randomly sampled {n_sample} segments ({100 * n_sample / total_found:.1f}% "
          f"of total) for validation figure.")

    config = {"global_dir": BIRD_ROOT}

    print("Extracting onset-aligned spectral-sum traces for sampled segments...")
    traces = []
    for i, seg in enumerate(sampled_segments):
        result = extract_segment_spectral_sum(
            seg["wav_path"], seg["onset"], seg["offset"], config,
            NPERSEG, NOVERLAP, NFFT, FREQ_CUTOFFS, PAD_SECONDS,
        )
        if result is None:
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
              f"and were excluded from the figure.")

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
    trace_top = 0.9  # trace occupies [0, trace_top] within its row
    for row_idx, tr in enumerate(traces):
        power = tr["power"]
        power_norm = power - np.min(power)
        max_range = np.max(power_norm) if np.max(power_norm) > 0 else 1.0
        power_norm = power_norm / max_range * trace_top

        y_offset = row_idx * row_gap
        ax.plot(tr["t"], power_norm + y_offset, color="black", linewidth=0.6)

        # Onset: every trace's t=0 is its true onset, so this line is at
        # the exact same x-position for every row -- forms a single clean
        # vertical reference line down the whole figure.
        ax.plot([0, 0], [y_offset, y_offset + trace_top],
               color="tab:green", linewidth=0.8, zorder=3)

        # Offset: marked with a triangle marker sitting just above the
        # trace (rather than a full vertical line), so the *marker shape*
        # -- not a line blending into neighboring rows -- is what the eye
        # tracks across rows. Because traces are sorted by duration, these
        # markers form a monotonically-rightward "staircase" when
        # segmentation is consistent; a marker that breaks the staircase
        # pattern relative to its neighbors, or a trace whose energy
        # visibly continues past its marker, flags a likely segmentation
        # problem for that syllable.
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
