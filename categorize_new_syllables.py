"""
MOOVE VAE+GMM New-Syllable Categorization Script
==================================================
Loads a trained VAE+GMM clustering model (saved by the "VAE + GMM" method
in MOOVE's Cluster dialog, as `<dataset_name>_vae_gmm_model.pth` under
trained_models/) and applies it to a NEW cluster dataset pickle (i.e. a
freshly created *_clus.pkl that has NOT yet been clustered).

For each new syllable, this:
  1) Encodes it into the trained VAE's latent space.
  2) Assigns a cluster label using the trained GMM.
  3) Projects it into the same 2D PCA space used for the training data.
  4) Plots the new syllables on top of the training-syllable point cloud,
     so you can visually inspect classification performance.

Usage
-----
    python categorize_new_syllables.py

You will be prompted for:
  - Path to the trained model (<dataset_name>_vae_gmm_model.pth)
  - Path to the NEW cluster dataset pickle to categorize
  - (Optional) path to the ORIGINAL training cluster dataset pickle, if you
    want the training point cloud plotted for comparison (recommended)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

N_TIMEBINS = 40  # must match the value hardcoded in create_cluster_dataset


# ---------------------------------------------------------------------------
# ConvVAE definition (must exactly match moove/utils/clustering_utils.py)
# ---------------------------------------------------------------------------
class ConvVAE(nn.Module):
    def __init__(self, n_freq_bins, latent_dim):
        super().__init__()
        self.n_freq_bins = n_freq_bins
        self.latent_dim = latent_dim

        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_freq_bins, N_TIMEBINS)
            enc_out = self.enc_conv(dummy)
            self._enc_shape = enc_out.shape[1:]
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


def infer_n_freq_bins(spectrogram_feature_array):
    """Same validation logic as clustering_utils.infer_n_freq_bins: confirms
    every row is divisible by 40 (timebins) and consistent across the dataset
    before trusting the reshape to (n_freq_bins, 40)."""
    row_lengths = {len(row) for row in spectrogram_feature_array}
    if len(row_lengths) != 1:
        raise ValueError(
            f"Inconsistent flattened spectrogram lengths found: {row_lengths}. "
            "Cannot safely infer a single (freq_bins, 40) shape."
        )
    row_len = row_lengths.pop()
    if row_len % N_TIMEBINS != 0:
        raise ValueError(
            f"Flattened spectrogram length {row_len} is not divisible by {N_TIMEBINS}."
        )
    return row_len // N_TIMEBINS


def encode_to_latent(model, spectrogram_feature_array, n_freq_bins, mean, std, device):
    model.eval()
    x = torch.tensor(spectrogram_feature_array, dtype=torch.float32)
    x = x.view(-1, 1, n_freq_bins, N_TIMEBINS)
    x = (x - mean) / std
    x = x.to(device)
    with torch.no_grad():
        mu, _ = model.encode(x)
    return mu.cpu().numpy()


def main():
    model_path = input("Path to trained VAE+GMM model (*_vae_gmm_model.pth): ").strip().strip('"')
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    new_data_path = input("Path to NEW cluster dataset pickle to categorize (*_clus.pkl): ").strip().strip('"')
    if not os.path.exists(new_data_path):
        print(f"ERROR: File not found: {new_data_path}")
        sys.exit(1)

    train_data_path = input(
        "Path to ORIGINAL training cluster dataset pickle (for point-cloud comparison, "
        "press Enter to skip): "
    ).strip().strip('"')
    if train_data_path and not os.path.exists(train_data_path):
        print(f"WARNING: Training dataset path not found, continuing without it: {train_data_path}")
        train_data_path = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']
    mean = checkpoint['norm_mean']
    std = checkpoint['norm_std']
    gmm = checkpoint['gmm']
    pca = checkpoint['pca']
    metadata = checkpoint['metadata']

    print(f"Loaded model: latent_dim={model_config['latent_dim']}, "
          f"n_freq_bins={model_config['n_freq_bins']}, "
          f"n_components (from BIC)={metadata['n_components_selected']}")

    model = ConvVAE(n_freq_bins=model_config['n_freq_bins'], latent_dim=model_config['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    df_new = pd.read_pickle(new_data_path)
    if not isinstance(df_new, pd.DataFrame):
        print("ERROR: New dataset pickle is not a DataFrame.")
        sys.exit(1)
    if 'cluster_flattend_spectrogram' not in df_new.columns:
        print("ERROR: New dataset is missing 'cluster_flattend_spectrogram' column. "
              "Was it created via MOOVE's 'Create Cluster Dataset' step?")
        sys.exit(1)

    new_feature_array = np.array([np.array(x) for x in df_new['cluster_flattend_spectrogram'].values])

    n_freq_bins_new = infer_n_freq_bins(new_feature_array)
    if n_freq_bins_new != model_config['n_freq_bins']:
        print(
            f"ERROR: New dataset's inferred frequency-bin count ({n_freq_bins_new}) does not "
            f"match the trained model's ({model_config['n_freq_bins']}). This usually means the "
            f"new dataset was created with different spectrogram/frequency-cutoff parameters "
            f"than the training dataset. Re-create the new dataset with matching parameters."
        )
        sys.exit(1)

    print(f"Encoding {len(df_new)} new syllable(s) into latent space...")
    new_latent = encode_to_latent(model, new_feature_array, n_freq_bins_new, mean, std, device)

    new_gmm_labels_int = gmm.predict(new_latent)
    label_mapping = {i: chr(97 + i) for i in range(gmm.n_components)}
    new_labels = [label_mapping[i] for i in new_gmm_labels_int]
    df_new['vae_gmm_label'] = new_labels

    new_latent_2d = pca.transform(new_latent)
    df_new['VAE1'] = new_latent_2d[:, 0]
    df_new['VAE2'] = new_latent_2d[:, 1]

    output_path = new_data_path  # overwrite in place, matching MOOVE's own convention
    df_new.to_pickle(output_path)
    print(f"Saved categorized labels to {output_path}")
    print(df_new['vae_gmm_label'].value_counts())

    # ---- Plot new syllables over the training point cloud ----
    fig, ax = plt.subplots(figsize=(10, 8))

    if train_data_path:
        df_train = pd.read_pickle(train_data_path)
        if 'VAE1' in df_train.columns and 'VAE2' in df_train.columns and 'vae_gmm_label' in df_train.columns:
            train_unique_labels = sorted(df_train['vae_gmm_label'].dropna().unique())
            train_label_map = {lab: idx for idx, lab in enumerate(train_unique_labels)}
            train_numeric = [train_label_map[l] for l in df_train['vae_gmm_label']]
            ax.scatter(df_train['VAE1'], df_train['VAE2'], c=train_numeric, cmap='jet',
                      s=5, alpha=0.25, label='Training syllables')
        else:
            print("WARNING: Training dataset does not have VAE1/VAE2/vae_gmm_label columns "
                  "(was it clustered with the VAE+GMM method?). Skipping training point cloud.")

    new_unique_labels = sorted(set(new_labels))
    new_label_map = {lab: idx for idx, lab in enumerate(new_unique_labels)}
    new_numeric = [new_label_map[l] for l in new_labels]
    scatter_new = ax.scatter(new_latent_2d[:, 0], new_latent_2d[:, 1], c=new_numeric, cmap='jet',
                             s=40, edgecolors='black', linewidths=0.8, marker='^',
                             label='New syllables')

    ax.set_xlabel("VAE1 (PCA of latent space)")
    ax.set_ylabel("VAE2 (PCA of latent space)")
    ax.set_title("New Syllable Categorization vs. Training Point Cloud")
    ax.legend(loc='best')

    plot_path = os.path.splitext(new_data_path)[0] + "_categorization_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
