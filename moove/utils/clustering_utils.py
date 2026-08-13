import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import threading
import warnings
import evfuncs
from matplotlib import cm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from scipy import interpolate
from scipy.signal import spectrogram
from sklearn.cluster import KMeans
from umap import UMAP

from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout

from moove.qt_helpers import invoke_in_main_thread, show_info, show_confirm_action_window

warnings.filterwarnings('ignore')

PAD_SAMPLE_FRACTION = 0.10   # fraction of syllables sampled to estimate padding width
PAD_PERCENTILE = 99          # percentile of (sampled) syllable duration used to set padding width
PAD_MARGIN = 1.10            # multiply the percentile duration by this margin for the final pad width
FLAGGED_COL = "duration_truncated"  # column marking syllables truncated to fit the pad width


def _set_cluster_running(app_state, running):
    """Store running state for the cluster dialog."""
    win = getattr(app_state, 'cluster_window', None)
    if win is not None:
        win._task_running = bool(running)
        if running:
            win._task_cancel_requested = False


def _cluster_cancel_requested(app_state):
    """Return True if user requested cancellation via dialog close."""
    win = getattr(app_state, 'cluster_window', None)
    return bool(win is not None and getattr(win, '_task_cancel_requested', False))


def start_create_cluster_dataset_thread(app_state, dataset_name, use_selected_files, selection, batch_file, bird,
                                        experiment, day, parent):
    """Start a thread to create a cluster dataset based on selected files and criteria."""
    from moove.utils import get_files_for_day, get_files_for_experiment, get_files_for_bird, filter_segmented_files

    if selection == "current_day":
        files = get_files_for_day(app_state, bird, experiment, day, batch_file)
    elif selection == "current_experiment":
        files = get_files_for_experiment(app_state, bird, experiment, batch_file)
    elif selection == "current_bird":
        files = get_files_for_bird(app_state, bird, batch_file)

    if use_selected_files:
        files = filter_segmented_files(files)

    app_state.logger.debug(
        "Creating training dataset with parameters: Use selected files: %s, Selection: %s, Batch file: %s",
        use_selected_files, selection, batch_file
    )

    if len(dataset_name) < 1:
        show_info(parent, "Error", "Dataset name not valid! A dataset name needs to contain at least one character.")
    else:
        win = app_state.cluster_window
        if getattr(win, '_task_running', False):
            show_info(win, "Info", "A cluster job is already running.")
            return

        progressbar = win.progressbar
        progressbar.setMaximum(len(files))
        progressbar.setValue(0)
        progressbar.show()
        _set_cluster_running(app_state, True)

        def thread_wrapper():
            current_thread = threading.current_thread()
            try:
                create_cluster_dataset(app_state, dataset_name, progressbar, len(files), files, parent)
            finally:
                _set_cluster_running(app_state, False)
                app_state.remove_thread(current_thread)

        thread = threading.Thread(target=thread_wrapper, name="CreateClusterDatasetThread")
        app_state.add_thread(thread)
        thread.start()


def _get_onset_offset_info(file_path):
    notmat_file = file_path + ".not.mat"
    if os.path.exists(notmat_file):
        notmat_dict = evfuncs.load_notmat(notmat_file)
        return {"onsets": notmat_dict.get("onsets", []), "offsets": notmat_dict.get("offsets", [])}
    return {"onsets": [], "offsets": []}


def _estimate_max_width_frames(all_files, app_state, sample_fraction=PAD_SAMPLE_FRACTION,
                                percentile=PAD_PERCENTILE, margin=PAD_MARGIN, random_seed=42):
    """Estimate the number of time-bin frames to zero-pad every syllable spectrogram to.

    Rather than resampling every syllable onto a fixed 40-bin time axis (which
    destroys duration information -- disastrous for e.g. stack calls, whose
    classes differ primarily in duration rather than spectral content), we
    zero-pad every syllable's spectrogram out to a fixed width in frames.
    That width is chosen as `margin` x the `percentile`-th percentile of
    syllable duration (in frames), estimated from a random `sample_fraction`
    of all syllables in the dataset (for efficiency on large datasets).

    Returns (max_width_frames, nperseg, noverlap, nfft, sampling_rate) so the
    caller can reuse the same STFT parameters when actually building spectrograms.
    """
    from moove.utils import get_display_data, seconds_to_index

    nperseg = int(app_state.spec_params['nperseg'].get())
    noverlap = int(app_state.spec_params['noverlap'].get())
    nfft = int(app_state.spec_params['nfft'].get())
    hop = nperseg - noverlap

    durations_frames = []
    rng = random.Random(random_seed)
    sample_files = all_files if sample_fraction >= 1.0 else rng.sample(
        all_files, max(1, int(len(all_files) * sample_fraction)))

    for file_i in sample_files:
        info = _get_onset_offset_info(file_i)
        onsets, offsets = info["onsets"], info["offsets"]
        if len(onsets) == 0 or len(offsets) == 0:
            continue
        try:
            file_path = {"file_name": os.path.basename(file_i), "file_path": os.path.join(os.getcwd(), file_i)}
            display_dict = get_display_data(file_path, app_state.config)
        except Exception:
            continue
        sampling_rate = int(display_dict["sampling_rate"])
        for onset, offset in zip(onsets, offsets):
            onset_index = int(seconds_to_index(onset, sampling_rate))
            offset_index = int(seconds_to_index(offset, sampling_rate))
            n_samples = max(0, offset_index - onset_index)
            n_frames = max(1, 1 + (n_samples - nperseg) // hop) if n_samples >= nperseg else 1
            durations_frames.append(n_frames)

    if not durations_frames:
        return 40, nperseg, noverlap, nfft, None

    p99 = float(np.percentile(durations_frames, percentile))
    max_width = int(np.ceil(p99 * margin))
    max_width = max(max_width, 1)
    return max_width, nperseg, noverlap, nfft, None


def create_cluster_dataset(app_state, dataset_name, progressbar, max_value, all_files, parent):
    """Generate and save a cluster dataset, tracking progress with a progress bar.

    Syllable spectrograms are zero-padded (not time-interpolated) to a fixed
    frame width estimated from the dataset, preserving duration as a real
    feature of the flattened spectrogram rather than normalizing it away.
    Syllables longer than the padding width are truncated and flagged via the
    'duration_truncated' column; the fraction of flagged syllables is printed
    to the terminal once dataset creation completes.
    """
    from moove.utils import get_display_data, seconds_to_index, decibel, plot_data

    original_data_dir = app_state.data_dir
    original_song_files = app_state.song_files.copy() if app_state.song_files else []
    original_current_file_index = app_state.current_file_index
    cancelled = False

    if dataset_name:
        going_prod_df = pd.DataFrame(columns=['file', 'onset_no', 'cluster_flattend_spectrogram', 'label',
                                              FLAGGED_COL])
        entry_no = 0

    invoke_in_main_thread(progressbar.hide)

    def _show_looking():
        if hasattr(app_state.cluster_window, 'status_label'):
            app_state.cluster_window.status_label.setText("Looking for segments...")
            app_state.cluster_window.status_label.show()

    invoke_in_main_thread(_show_looking)

    num_segs = 0
    for file_path in all_files:
        info = _get_onset_offset_info(file_path)
        if len(info["onsets"]) > 0 and len(info["offsets"]) > 0:
            num_segs += min(len(info["onsets"]), len(info["offsets"]))

    if num_segs < 10:
        invoke_in_main_thread(lambda: (
            app_state.cluster_window.status_label.hide() if hasattr(app_state.cluster_window, 'status_label') else None,
            show_info(parent, "Error", "Not enough segments given. Need at least 10 segments to form clusters.")))
        return

    def _show_estimating():
        if hasattr(app_state.cluster_window, 'status_label'):
            app_state.cluster_window.status_label.setText("Estimating padding width from sample...")
            app_state.cluster_window.status_label.show()

    invoke_in_main_thread(_show_estimating)

    max_width, nperseg, noverlap, nfft, _ = _estimate_max_width_frames(all_files, app_state)
    app_state.logger.info(
        "Zero-padding syllable spectrograms to %d time frames (%.0f%% margin over p%d duration, "
        "estimated from a %.0f%% random sample).",
        max_width, (PAD_MARGIN - 1.0) * 100, PAD_PERCENTILE, PAD_SAMPLE_FRACTION * 100
    )
    print(f"[Cluster Dataset] Zero-padding syllables to {max_width} time frames "
          f"(p{PAD_PERCENTILE} duration x{PAD_MARGIN} margin, from {PAD_SAMPLE_FRACTION*100:.0f}% sample).")

    def _hide_show_progress():
        if hasattr(app_state.cluster_window, 'status_label'):
            app_state.cluster_window.status_label.hide()
        progressbar.show()

    invoke_in_main_thread(_hide_show_progress)

    n_truncated = 0
    n_total_syllables = 0

    for i in range(max_value):
        if _cluster_cancel_requested(app_state):
            cancelled = True
            break
        invoke_in_main_thread(progressbar.setValue, i)
        file_i = all_files[i]
        file_path = {"file_name": os.path.basename(file_i), "file_path": os.path.join(os.getcwd(), file_i)}
        try:
            display_dict = get_display_data(file_path, app_state.config)
        except Exception as e:
            app_state.logger.error("Skipping file '%s' in clustering dataset creation: %s", file_i, e)
            print(f"Skipped file: {file_i}")
            continue
        app_state.data_dir = os.path.dirname(file_i)

        sampling_rate = int(display_dict["sampling_rate"])
        rawsong = display_dict["song_data"]

        freq_cutoffs = tuple(map(int, app_state.spec_params['freq_cutoffs'].get().split(',')))
        onsets, offsets = display_dict["onsets"], display_dict["offsets"]

        if dataset_name:
            for syllable_no, (onset, offset) in enumerate(zip(onsets, offsets)):
                entry_no += 1
                n_total_syllables += 1
                onset_index = int(seconds_to_index(onset, sampling_rate))
                offset_index = int(seconds_to_index(offset, sampling_rate))
                cutted_raw_song = rawsong[onset_index:offset_index]
                f, t, Sxx_cluster = spectrogram(cutted_raw_song, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap,
                                                nfft=nfft)

                Sxx_cluster = Sxx_cluster[(f >= freq_cutoffs[0]) & (f <= freq_cutoffs[1]), :]
                Sxx_cluster = decibel(Sxx_cluster)

                n_freq_bins, n_time_frames = Sxx_cluster.shape
                is_truncated = False
                if n_time_frames >= max_width:
                    if n_time_frames > max_width:
                        is_truncated = True
                        n_truncated += 1
                    Sxx_cluster = Sxx_cluster[:, :max_width]
                else:
                    pad_width = max_width - n_time_frames
                    Sxx_cluster = np.pad(Sxx_cluster, ((0, 0), (0, pad_width)), mode='constant',
                                         constant_values=Sxx_cluster.min() if Sxx_cluster.size else 0.0)

                going_prod_df.loc[entry_no] = [file_i, syllable_no, Sxx_cluster.flatten(), "x", is_truncated]

    app_state.data_dir = original_data_dir
    app_state.song_files = original_song_files
    app_state.current_file_index = original_current_file_index

    if cancelled:
        invoke_in_main_thread(progressbar.hide)
        invoke_in_main_thread(lambda: show_info(parent, "Info", "Cluster dataset creation aborted."))
        return

    if dataset_name:
        file_path = os.path.join(app_state.config['global_dir'], 'cluster_data', f'{dataset_name}_clus.pkl')
        going_prod_df.to_pickle(file_path)
        app_state.update_cluster_datasets_combobox()

    flagged_fraction = (n_truncated / n_total_syllables) if n_total_syllables else 0.0
    app_state.logger.info(
        "Cluster dataset creation complete: %d/%d syllables (%.2f%%) were truncated to fit the %d-frame pad width.",
        n_truncated, n_total_syllables, flagged_fraction * 100, max_width
    )
    print(f"[Cluster Dataset] Flagged (truncated) syllables: {n_truncated}/{n_total_syllables} "
          f"({flagged_fraction * 100:.2f}%). See '{FLAGGED_COL}' column in the saved dataset.")

    invoke_in_main_thread(progressbar.setValue, max_value)
    invoke_in_main_thread(progressbar.hide)
    invoke_in_main_thread(lambda: show_info(
        parent, "Info", f"Cluster dataset '{dataset_name}' created successfully!\n\n"
                        f"Truncated syllables: {n_truncated}/{n_total_syllables} ({flagged_fraction * 100:.2f}%)"))


def start_clustering_thread(parent, app_state, dataset_name_entry):
    """Start the clustering process in a separate thread."""
    dataset_name = dataset_name_entry
    if dataset_name == "Select Cluster Dataset":
        show_info(parent, "Error", "Selected cluster dataset not valid! Perhaps you forgot to pick a dataset?")
        return
    else:
        if getattr(app_state.cluster_window, '_task_running', False):
            show_info(app_state.cluster_window, "Info", "A cluster job is already running.")
            return
        if not show_confirm_action_window(parent, "Info", "Clustering started. "
                                                          "This may take a while, please wait!"):
            # stop execution if closed with [Close]
            return

    _set_cluster_running(app_state, True)

    def thread_wrapper():
        current_thread = threading.current_thread()
        try:
            run_clustering(parent, app_state, dataset_name)
        finally:
            _set_cluster_running(app_state, False)
            app_state.remove_thread(current_thread)

    thread = threading.Thread(target=thread_wrapper, name="ClusteringThread")
    app_state.add_thread(thread)
    thread.start()


def run_clustering(parent, app_state, dataset_name):
    """Run the clustering process using UMAP and KMeans."""
    dataset_name_pkl = f"{dataset_name}.pkl"
    n_syllables = int(app_state.umap_k_means_params['n_clusters'].get())
    n_neighbors = int(app_state.umap_k_means_params['n_neighbors'].get())
    min_dist = float(app_state.umap_k_means_params['min_dist'].get())

    def _show_running():
        if hasattr(app_state.cluster_window, 'status_label'):
            app_state.cluster_window.status_label.setText("Running...")
            app_state.cluster_window.status_label.show()

    def _hide_running():
        if hasattr(app_state.cluster_window, 'status_label'):
            app_state.cluster_window.status_label.hide()

    invoke_in_main_thread(_show_running)

    if _cluster_cancel_requested(app_state):
        invoke_in_main_thread(_hide_running)
        invoke_in_main_thread(lambda: show_info(parent, "Info", "Clustering aborted."))
        return

    dataset_path = os.path.join(app_state.config['global_dir'], 'cluster_data', dataset_name_pkl)
    if not os.path.exists(dataset_path):
        app_state.logger.error("Dataset %s not found in cluster_data folder.", dataset_name_pkl)
        invoke_in_main_thread(_hide_running)
        return

    df = pd.read_pickle(dataset_path)
    spectrogram_feature_array = np.array([np.array(x) for x in df['cluster_flattend_spectrogram'].values])

    umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric='euclidean', random_state=42)
    low_dimensional_data = umap_model.fit_transform(spectrogram_feature_array)

    if _cluster_cancel_requested(app_state):
        invoke_in_main_thread(_hide_running)
        invoke_in_main_thread(lambda: show_info(parent, "Info", "Clustering aborted."))
        return

    kmeans = KMeans(n_clusters=n_syllables, random_state=42)
    labels = kmeans.fit_predict(low_dimensional_data)

    label_mapping = {i: chr(97 + i) for i in range(n_syllables)}
    alphabet_labels = [label_mapping[label] for label in labels]
    df['clustered_label'] = alphabet_labels

    df['UMAP1'] = low_dimensional_data[:, 0]
    df['UMAP2'] = low_dimensional_data[:, 1]

    output_path = os.path.join(app_state.config['global_dir'], 'cluster_data', dataset_name_pkl)
    df.to_pickle(output_path)

    invoke_in_main_thread(plot_generic_clusters, parent, app_state, low_dimensional_data, alphabet_labels,
                          output_path, "UMAP1", "UMAP2", "UMAP Clustering")

    invoke_in_main_thread(_hide_running)

    app_state.logger.debug("Clustering complete. Results saved to %s", output_path)
    invoke_in_main_thread(lambda: show_info(
        parent, "Info", f"Clustering complete! Results saved to {output_path}"))


def plot_clusters(parent, app_state, low_dimensional_data, labels, output_path):
    """Backward-compatible wrapper around plot_generic_clusters for UMAP+KMeans output."""
    plot_generic_clusters(parent, app_state, low_dimensional_data, labels, output_path,
                          "UMAP1", "UMAP2", "UMAP Clustering")


def replace_labels_from_df(app_state, dataset_name, parent=None):
    """Replace labels in the dataset based on clustering results (UMAP+KMeans only).
    Kept for backward compatibility; replace_labels_from_df_method() below is the
    method-aware version used by the current Cluster dialog."""
    replace_labels_from_df_method(app_state, dataset_name, method="umap_kmeans", parent=parent)


# ======================================================================
# VAE + GMM clustering (parallel alternative to UMAP + KMeans, above)
# ======================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

VAE_LABEL_COL = "vae_gmm_label"
VAE_COORD_COLS = ("VAE1", "VAE2")

# Default hyperparameters (also exposed/overridable via app_state.vae_gmm_params
# in the Cluster dialog). Kept here so this module works standalone/scriptable.
VAE_LATENT_DIM = 32
VAE_EPOCHS = 200
VAE_BATCH_SIZE = 64
VAE_LEARNING_RATE = 1e-3
VAE_BIC_MIN_COMPONENTS = 2
VAE_BIC_MAX_COMPONENTS = 20


def infer_time_and_freq_bins(spectrogram_feature_array, df=None):
    """Infer the flattened row length from a spectrogram feature array,
    validating that every row shares the same length (i.e. the dataset was
    created with one consistent padding width). Unlike the old fixed-40-bin
    assumption, the padded time width now varies per dataset (chosen at
    creation time from the data itself via _estimate_max_width_frames), so
    recovering the actual (n_freq_bins, n_time_frames) split requires
    re-deriving n_freq_bins from a real syllable -- see run_vae_encoding."""
    row_lengths = {len(row) for row in spectrogram_feature_array}
    if len(row_lengths) != 1:
        raise ValueError(
            f"Inconsistent flattened spectrogram lengths found in dataset: {row_lengths}. "
            "This dataset may combine rows generated with different padding widths / "
            "spectrogram parameters. Re-create the cluster dataset so all rows share one width."
        )
    return row_lengths.pop()


class ConvVAE(nn.Module):
    """Convolutional VAE operating on (1, n_freq_bins, n_time_frames) spectrogram patches."""

    def __init__(self, n_freq_bins, n_time_frames, latent_dim=VAE_LATENT_DIM):
        super().__init__()
        self.n_freq_bins = n_freq_bins
        self.n_time_frames = n_time_frames
        self.latent_dim = latent_dim

        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_freq_bins, n_time_frames)
            enc_out = self.enc_conv(dummy)
            self._enc_shape = enc_out.shape[1:]  # (C, H, W)
            enc_flat_dim = enc_out.numel()

        self.fc_mu = nn.Linear(enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, enc_flat_dim)

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, *self._enc_shape)
        out = self.dec_conv(h)
        # Guard against off-by-one size mismatches from transposed convs;
        # crop/pad to exactly (n_freq_bins, n_time_frames).
        out = out[:, :, :self.n_freq_bins, :self.n_time_frames]
        if out.shape[2] < self.n_freq_bins or out.shape[3] < self.n_time_frames:
            pad_h = self.n_freq_bins - out.shape[2]
            pad_w = self.n_time_frames - out.shape[3]
            out = nn.functional.pad(out, (0, max(pad_w, 0), 0, max(pad_h, 0)))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    # NOTE (duration conditioning hook): if zero-padding alone proves
    # insufficient to let the VAE separate duration-differentiated classes
    # (e.g. stack calls), a conditioning input can be added here by
    # concatenating a z-scored duration scalar onto the flattened encoder
    # output before fc_mu/fc_logvar, and onto z before fc_dec, then widening
    # those Linear layers by 1 input/output feature accordingly. Not
    # implemented by default -- zero-padding is the primary fix; conditioning
    # is a documented extension point, not a default behavior.
    # ------------------------------------------------------------------


def vae_loss_function(recon_x, x, mu, logvar):
    """Reconstruction (MSE, summed) + KL divergence to standard normal prior."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div, recon_loss, kl_div


def train_vae(spectrogram_feature_array, n_freq_bins, n_time_frames, latent_dim=VAE_LATENT_DIM,
              epochs=VAE_EPOCHS, batch_size=VAE_BATCH_SIZE, learning_rate=VAE_LEARNING_RATE,
              device=None, cancel_check=None, progress_callback=None):
    """Train a ConvVAE on the full dataset (no validation split, per design).
    cancel_check: optional callable returning True if training should abort.
    progress_callback: optional callable(epoch, epochs, train_loss) for GUI updates."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor(spectrogram_feature_array, dtype=torch.float32)
    x = x.view(-1, 1, n_freq_bins, n_time_frames)

    mean, std = x.mean(), x.std()
    std = std if std > 1e-8 else 1.0
    x = (x - mean) / std

    dataset = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConvVAE(n_freq_bins=n_freq_bins, n_time_frames=n_time_frames, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        if cancel_check is not None and cancel_check():
            return model, mean.item(), std.item(), True
        model.train()
        running_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss_function(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(dataset)
        if progress_callback is not None:
            progress_callback(epoch, epochs, train_loss)

    return model, mean.item(), std.item(), False


def encode_to_latent(model, spectrogram_feature_array, n_freq_bins, n_time_frames, mean, std, device=None):
    """Encode a feature array to latent means (mu) using a trained ConvVAE."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x = torch.tensor(spectrogram_feature_array, dtype=torch.float32)
    x = x.view(-1, 1, n_freq_bins, n_time_frames)
    x = (x - mean) / std
    x = x.to(device)
    with torch.no_grad():
        mu, _ = model.encode(x)
    return mu.cpu().numpy()


def fit_gmm_with_bic(latent_vectors, min_components=VAE_BIC_MIN_COMPONENTS,
                      max_components=VAE_BIC_MAX_COMPONENTS, random_state=42,
                      logger=None):
    """Fit a GaussianMixture for each n_components in [min_components, max_components]
    and return the one minimizing BIC, along with the full BIC sweep for logging.
    Prints and (optionally) logs each component count's BIC as it's computed, so
    a monotonically-decreasing BIC that pins to max_components (a red flag that
    the component ceiling, not a real minimum, drove the selection) is visible
    immediately rather than silently accepted."""
    best_gmm, best_bic, bic_scores = None, np.inf, []
    max_components = min(max_components, latent_vectors.shape[0] - 1)
    for n_components in range(min_components, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                              random_state=random_state)
        gmm.fit(latent_vectors)
        bic = gmm.bic(latent_vectors)
        bic_scores.append((n_components, bic))
        msg = f"[GMM BIC sweep] n_components={n_components:3d}  BIC={bic:,.2f}"
        print(msg)
        if logger is not None:
            logger.info(msg)
        if bic < best_bic:
            best_bic, best_gmm = bic, gmm

    if best_gmm is not None and best_gmm.n_components == max_components:
        warn_msg = (f"[GMM BIC sweep] WARNING: selected n_components={max_components} equals the "
                    f"upper sweep bound. BIC may still be decreasing past this ceiling rather than "
                    f"having found a true minimum -- consider raising bic_max_components and/or "
                    f"using covariance_type='diag' if this keeps happening.")
        print(warn_msg)
        if logger is not None:
            logger.warning(warn_msg)

    return best_gmm, bic_scores


def start_vae_encoding_thread(parent, app_state, dataset_name_entry):
    """Start VAE training + latent encoding in a separate thread (Step 1 of 2
    for the VAE+GMM method). Latents are saved to disk so GMM fitting (Step 2)
    can be re-run cheaply with different BIC ranges without re-training the VAE."""
    dataset_name = dataset_name_entry
    if dataset_name == "Select Cluster Dataset":
        show_info(parent, "Error", "Selected cluster dataset not valid! Perhaps you forgot to pick a dataset?")
        return
    else:
        if getattr(app_state.cluster_window, '_task_running', False):
            show_info(app_state.cluster_window, "Info", "A cluster job is already running.")
            return
        if not show_confirm_action_window(parent, "Info", "VAE training/encoding started. "
                                                          "This may take a while, please wait!"):
            return

    _set_cluster_running(app_state, True)

    def thread_wrapper():
        current_thread = threading.current_thread()
        try:
            run_vae_encoding(parent, app_state, dataset_name)
        finally:
            _set_cluster_running(app_state, False)
            app_state.remove_thread(current_thread)

    thread = threading.Thread(target=thread_wrapper, name="VAEEncodingThread")
    app_state.add_thread(thread)
    thread.start()


def run_vae_encoding(parent, app_state, dataset_name):
    """Step 1 of 2 for VAE+GMM: train a ConvVAE on the cluster dataset's
    (zero-padded) spectrograms and encode every syllable to its latent mu.
    Saves latents + the trained VAE checkpoint to disk. Does NOT run GMM --
    see run_gmm_clustering() for that, which can be re-run cheaply against
    these saved latents with different BIC sweep ranges."""
    dataset_name_pkl = f"{dataset_name}.pkl"
    latent_dim = int(app_state.vae_gmm_params['latent_dim'].get())
    epochs = int(app_state.vae_gmm_params['epochs'].get())
    batch_size = int(app_state.vae_gmm_params['batch_size'].get())
    learning_rate = float(app_state.vae_gmm_params['learning_rate'].get())

    def _show_running(text="Training VAE..."):
        if hasattr(app_state.cluster_window, 'status_label'):
            app_state.cluster_window.status_label.setText(text)
            app_state.cluster_window.status_label.show()

    def _hide_running():
        if hasattr(app_state.cluster_window, 'status_label'):
            app_state.cluster_window.status_label.hide()

    invoke_in_main_thread(_show_running, "Training VAE...")

    if _cluster_cancel_requested(app_state):
        invoke_in_main_thread(_hide_running)
        invoke_in_main_thread(lambda: show_info(parent, "Info", "VAE encoding aborted."))
        return

    dataset_path = os.path.join(app_state.config['global_dir'], 'cluster_data', dataset_name_pkl)
    if not os.path.exists(dataset_path):
        app_state.logger.error("Dataset %s not found in cluster_data folder.", dataset_name_pkl)
        invoke_in_main_thread(_hide_running)
        return

    df = pd.read_pickle(dataset_path)
    spectrogram_feature_array = np.array([np.array(x) for x in df['cluster_flattend_spectrogram'].values])

    nperseg = int(app_state.spec_params['nperseg'].get())
    noverlap = int(app_state.spec_params['noverlap'].get())
    freq_cutoffs = tuple(map(int, app_state.spec_params['freq_cutoffs'].get().split(',')))
    nfft = int(app_state.spec_params['nfft'].get())
    flat_len = infer_time_and_freq_bins(spectrogram_feature_array)

    # n_freq_bins isn't independently recoverable from flat_len alone without
    # knowing the padding width used at dataset-creation time; recompute it
    # directly the same way create_cluster_dataset does, using one real syllable.
    sample_row_2d = None
    for file_i in df['file'].unique():
        info = _get_onset_offset_info(file_i)
        if len(info["onsets"]) > 0 and len(info["offsets"]) > 0:
            from moove.utils import get_display_data, seconds_to_index
            try:
                file_path = {"file_name": os.path.basename(file_i), "file_path": os.path.join(os.getcwd(), file_i)}
                display_dict = get_display_data(file_path, app_state.config)
                sampling_rate = int(display_dict["sampling_rate"])
                onset_index = int(seconds_to_index(info["onsets"][0], sampling_rate))
                offset_index = int(seconds_to_index(info["offsets"][0], sampling_rate))
                cutted = display_dict["song_data"][onset_index:offset_index]
                f, t, Sxx = spectrogram(cutted, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                Sxx = Sxx[(f >= freq_cutoffs[0]) & (f <= freq_cutoffs[1]), :]
                sample_row_2d = Sxx
                break
            except Exception:
                continue
    if sample_row_2d is None:
        err = ("Could not recover frequency-bin count for this dataset (no readable source files "
              "found). Re-create the cluster dataset.")
        app_state.logger.error(err)
        invoke_in_main_thread(_hide_running)
        invoke_in_main_thread(lambda: show_info(parent, "Error", err))
        return

    n_freq_bins = sample_row_2d.shape[0]
    if flat_len % n_freq_bins != 0:
        err = (f"Dataset's flattened spectrogram length ({flat_len}) is not divisible by the "
              f"recovered frequency-bin count ({n_freq_bins}). This dataset may have been created "
              f"with different spectrogram parameters than are currently set. Re-create it.")
        app_state.logger.error(err)
        invoke_in_main_thread(_hide_running)
        invoke_in_main_thread(lambda: show_info(parent, "Error", err))
        return
    n_time_frames = flat_len // n_freq_bins

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _progress_cb(epoch, total_epochs, train_loss):
        if (epoch + 1) % 10 == 0 or epoch == 0:
            app_state.logger.debug("VAE epoch %d/%d - train_loss: %.4f", epoch + 1, total_epochs, train_loss)
            invoke_in_main_thread(_show_running, f"Training VAE... epoch {epoch + 1}/{total_epochs}")

    model, mean, std, cancelled = train_vae(
        spectrogram_feature_array, n_freq_bins, n_time_frames, latent_dim=latent_dim,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        device=device, cancel_check=lambda: _cluster_cancel_requested(app_state),
        progress_callback=_progress_cb,
    )

    if cancelled:
        invoke_in_main_thread(_hide_running)
        invoke_in_main_thread(lambda: show_info(parent, "Info", "VAE encoding aborted."))
        return

    invoke_in_main_thread(_show_running, "Encoding latents...")
    latent_vectors = encode_to_latent(model, spectrogram_feature_array, n_freq_bins, n_time_frames, mean, std,
                                      device=device)

    latents_path = os.path.join(app_state.config['global_dir'], 'cluster_data', f'{dataset_name}_vae_latents.npz')
    np.savez(latents_path, latents=latent_vectors, n_freq_bins=n_freq_bins, n_time_frames=n_time_frames)

    model_save_path = os.path.join(app_state.config['global_dir'], 'trained_models',
                                   f"{dataset_name}_vae_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {'n_freq_bins': n_freq_bins, 'n_time_frames': n_time_frames, 'latent_dim': latent_dim},
        'norm_mean': mean,
        'norm_std': std,
        'metadata': {
            'latent_dim': latent_dim, 'epochs': epochs, 'batch_size': batch_size,
            'learning_rate': learning_rate, 'n_freq_bins': n_freq_bins, 'n_time_frames': n_time_frames,
        },
    }, model_save_path)
    app_state.logger.debug("VAE model saved to %s; latents saved to %s", model_save_path, latents_path)

    invoke_in_main_thread(_hide_running)
    invoke_in_main_thread(lambda: show_info(
        parent, "Info",
        f"VAE training and latent encoding complete!\nLatents saved for {len(latent_vectors)} syllables.\n"
        f"You can now run 'Fit Clusters (GMM)' -- it will reuse these latents and can be "
        f"re-run cheaply with different BIC settings without retraining the VAE."))


def start_gmm_clustering_thread(parent, app_state, dataset_name_entry):
    """Start GMM fitting in a separate thread (Step 2 of 2 for VAE+GMM).
    Requires that run_vae_encoding() has already been run for this dataset."""
    dataset_name = dataset_name_entry
    if dataset_name == "Select Cluster Dataset":
        show_info(parent, "Error", "Selected cluster dataset not valid! Perhaps you forgot to pick a dataset?")
        return
    else:
        if getattr(app_state.cluster_window, '_task_running', False):
            show_info(app_state.cluster_window, "Info", "A cluster job is already running.")
            return

    latents_path = os.path.join(app_state.config['global_dir'], 'cluster_data', f'{dataset_name}_vae_latents.npz')
    if not os.path.exists(latents_path):
        show_info(parent, "Error", "No saved VAE latents found for this dataset. Run 'Train VAE / Encode Latents' "
                                   "first.")
        return

    _set_cluster_running(app_state, True)

    def thread_wrapper():
        current_thread = threading.current_thread()
        try:
            run_gmm_clustering(parent, app_state, dataset_name)
        finally:
            _set_cluster_running(app_state, False)
            app_state.remove_thread(current_thread)

    thread = threading.Thread(target=thread_wrapper, name="GMMClusteringThread")
    app_state.add_thread(thread)
    thread.start()


def run_gmm_clustering(parent, app_state, dataset_name):
    """Step 2 of 2 for VAE+GMM: load previously-saved VAE latents (from
    run_vae_encoding) and fit a BIC-selected GaussianMixture, writing labels
    back to the cluster dataset pickle. Does NOT touch the VAE -- this can be
    re-run cheaply with different bic_min/max_components without retraining."""
    dataset_name_pkl = f"{dataset_name}.pkl"
    bic_min = int(app_state.vae_gmm_params['bic_min_components'].get())
    bic_max = int(app_state.vae_gmm_params['bic_max_components'].get())

    def _show_running(text="Fitting GMM (BIC sweep)..."):
        if hasattr(app_state.cluster_window, 'status_label'):
            app_state.cluster_window.status_label.setText(text)
            app_state.cluster_window.status_label.show()

    def _hide_running():
        if hasattr(app_state.cluster_window, 'status_label'):
            app_state.cluster_window.status_label.hide()

    invoke_in_main_thread(_show_running)

    if _cluster_cancel_requested(app_state):
        invoke_in_main_thread(_hide_running)
        invoke_in_main_thread(lambda: show_info(parent, "Info", "GMM clustering aborted."))
        return

    latents_path = os.path.join(app_state.config['global_dir'], 'cluster_data', f'{dataset_name}_vae_latents.npz')
    if not os.path.exists(latents_path):
        app_state.logger.error("No saved VAE latents found at %s.", latents_path)
        invoke_in_main_thread(_hide_running)
        invoke_in_main_thread(lambda: show_info(
            parent, "Error", "No saved VAE latents found. Run 'Train VAE / Encode Latents' first."))
        return

    latents_npz = np.load(latents_path)
    latent_vectors = latents_npz['latents']

    dataset_path = os.path.join(app_state.config['global_dir'], 'cluster_data', dataset_name_pkl)
    if not os.path.exists(dataset_path):
        app_state.logger.error("Dataset %s not found in cluster_data folder.", dataset_name_pkl)
        invoke_in_main_thread(_hide_running)
        return
    df = pd.read_pickle(dataset_path)

    if len(df) != len(latent_vectors):
        err = (f"Row count mismatch: dataset has {len(df)} syllables but saved latents have "
              f"{len(latent_vectors)}. The dataset may have changed since latents were computed -- "
              f"re-run 'Train VAE / Encode Latents'.")
        app_state.logger.error(err)
        invoke_in_main_thread(_hide_running)
        invoke_in_main_thread(lambda: show_info(parent, "Error", err))
        return

    app_state.logger.info("Starting GMM BIC sweep: n_components in [%d, %d].", bic_min, bic_max)
    print(f"[GMM BIC sweep] Starting sweep: n_components in [{bic_min}, {bic_max}] "
          f"over {len(latent_vectors)} latent vectors.")

    gmm, bic_scores = fit_gmm_with_bic(latent_vectors, min_components=bic_min, max_components=bic_max,
                                       logger=app_state.logger)

    if _cluster_cancel_requested(app_state):
        invoke_in_main_thread(_hide_running)
        invoke_in_main_thread(lambda: show_info(parent, "Info", "GMM clustering aborted."))
        return

    n_syllables = gmm.n_components
    app_state.logger.info("Selected GMM with %d components via BIC.", n_syllables)
    print(f"[GMM BIC sweep] Selected n_components={n_syllables} (lowest BIC).")

    labels = gmm.predict(latent_vectors)
    label_mapping = {i: chr(97 + i) for i in range(n_syllables)}
    alphabet_labels = [label_mapping[label] for label in labels]

    pca = PCA(n_components=2, random_state=42)
    latent_2d = pca.fit_transform(latent_vectors)

    df[VAE_LABEL_COL] = alphabet_labels
    df[VAE_COORD_COLS[0]] = latent_2d[:, 0]
    df[VAE_COORD_COLS[1]] = latent_2d[:, 1]

    output_path = os.path.join(app_state.config['global_dir'], 'cluster_data', dataset_name_pkl)
    df.to_pickle(output_path)

    gmm_save_path = os.path.join(app_state.config['global_dir'], 'trained_models',
                                 f"{dataset_name}_gmm_model.pth")
    torch.save({
        'gmm': gmm,
        'pca': pca,
        'metadata': {
            'bic_min_components': bic_min, 'bic_max_components': bic_max,
            'n_components_selected': n_syllables, 'bic_scores': bic_scores,
        },
    }, gmm_save_path)
    app_state.logger.debug("GMM model saved to %s", gmm_save_path)

    invoke_in_main_thread(plot_generic_clusters, parent, app_state, latent_2d, alphabet_labels,
                          output_path, VAE_COORD_COLS[0], VAE_COORD_COLS[1], "VAE Latent Space (PCA) Clustering")

    invoke_in_main_thread(_hide_running)

    app_state.logger.debug("GMM clustering complete. Results saved to %s", output_path)
    invoke_in_main_thread(lambda: show_info(
        parent, "Info",
        f"GMM clustering complete!\nSelected {n_syllables} clusters via BIC.\nResults saved to {output_path}"))


def plot_generic_clusters(parent, app_state, low_dimensional_data, labels, output_path,
                          xlabel="Dim1", ylabel="Dim2", title="Clustering"):
    """Generic version of plot_clusters that works for both UMAP+KMeans and
    VAE+GMM output, parameterized by axis labels/title."""
    unique_labels = sorted(set(labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_mapping[label] for label in labels]

    dlg = QDialog(parent)
    dlg.setWindowTitle("Cluster Plot")
    dlg.resize(800, 600)
    layout = QVBoxLayout(dlg)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(left=0.124, bottom=0.138, top=0.912, right=0.842, wspace=0.2, hspace=0.2)
    scatter = ax.scatter(low_dimensional_data[:, 0], low_dimensional_data[:, 1],
                         c=numeric_labels, s=5, cmap=matplotlib.colormaps['jet'])

    handles, _ = scatter.legend_elements()
    legend = ax.legend(handles, unique_labels, title="Labels", loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.add_artist(legend)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plot_suffix = "_clusters.png" if xlabel == "UMAP1" else "_vae_clusters.png"
    plot_path = output_path.replace('_clus.pkl', plot_suffix).replace('.pkl', plot_suffix)
    plt.savefig(plot_path)
    app_state.logger.debug("Cluster plot saved to %s", plot_path)

    canvas = FigureCanvasQTAgg(fig)
    toolbar = NavigationToolbar2QT(canvas, dlg)
    layout.addWidget(toolbar)
    layout.addWidget(canvas)

    dlg.show()


def replace_labels_from_df_method(app_state, dataset_name, method="umap_kmeans", parent=None):
    """Method-aware version of replace_labels_from_df: writes either the
    'clustered_label' (UMAP+KMeans) or 'vae_gmm_label' (VAE+GMM) column into
    each file's .not.mat, depending on `method`."""
    from moove.utils.file_utils import get_display_data
    from moove.utils.movefuncs_utils import save_notmat
    from moove.utils.plot_utils import plot_data

    label_col = VAE_LABEL_COL if method == "vae_gmm" else "clustered_label"

    if dataset_name == "Select Cluster Dataset":
        show_info(parent, "Error", "Selected cluster dataset not valid! Perhaps you forgot to pick a dataset?")
        return
    else:
        if not show_confirm_action_window(parent, "Info", "Replacement of syllables started. "
                                                          "This may take a while, please wait!"):
            return

    original_data_dir = app_state.data_dir
    original_song_files = app_state.song_files.copy() if app_state.song_files else []
    original_current_file_index = app_state.current_file_index
    _set_cluster_running(app_state, True)

    dataset_path = os.path.join(app_state.config['global_dir'], 'cluster_data', f'{dataset_name}.pkl')
    df = pd.read_pickle(dataset_path)
    files = df['file'].unique()

    app_state.logger.debug("Starting replacement of syllables with dataset %s (method=%s, label_col=%s)",
                           dataset_name, method, label_col)

    processed_count = 0
    failed_count = 0
    cancelled = False

    win = app_state.cluster_window
    progressbar = win.progressbar
    progressbar.setMaximum(len(files))
    progressbar.setValue(0)
    progressbar.show()

    for i, file in enumerate(files):
        QApplication.processEvents()
        if _cluster_cancel_requested(app_state):
            cancelled = True
            break
        try:
            invoke_in_main_thread(progressbar.setValue, i)
            if label_col not in df.columns:
                raise KeyError(
                    f"Dataset has not been clustered with the '{method}' method yet. "
                    f"Please run that clustering method first before replacing labels."
                )
            labels = df.loc[df['file'] == file][label_col].astype(str).str.cat(sep='')

            display_dict = get_display_data({"file_name": os.path.basename(file), "file_path": file}, app_state.config)
            display_dict["labels"] = labels

            app_state.data_dir = os.path.dirname(file)
            save_path = os.path.join(app_state.data_dir, f"{display_dict['file_name']}.not.mat")
            app_state.logger.debug("Saving labels to %s", save_path)
            save_notmat(save_path, display_dict)
            processed_count += 1

        except Exception as e:
            app_state.logger.error(f"File {file} could not be processed correctly: {e}. Check manually.")
            print(f"Skipped file: {file}")
            failed_count += 1
            continue

    app_state.data_dir = original_data_dir
    app_state.song_files = original_song_files
    app_state.current_file_index = original_current_file_index

    if cancelled:
        invoke_in_main_thread(progressbar.hide)
        _set_cluster_running(app_state, False)
        invoke_in_main_thread(lambda: show_info(parent, "Info", "Label replacement aborted."))
        return

    invoke_in_main_thread(progressbar.setValue, len(files))
    invoke_in_main_thread(progressbar.hide)
    invoke_in_main_thread(app_state.reset_edit_type)
    invoke_in_main_thread(plot_data, app_state)

    invoke_in_main_thread(lambda: show_info(
        parent, "Info", f"Replacement of syllables complete!\n\n"
                        f"Total: {len(files)}\n"
                        f"Processed: {processed_count}\n"
                        f"Failed: {failed_count}\n"))
    _set_cluster_running(app_state, False)
