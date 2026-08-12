import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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


def _set_cluster_running(app_state, running):
    win = getattr(app_state, 'cluster_window', None)
    if win is not None:
        win._task_running = bool(running)
        if running:
            win._task_cancel_requested = False


def _cluster_cancel_requested(app_state):
    win = getattr(app_state, 'cluster_window', None)
    return bool(win is not None and getattr(win, '_task_cancel_requested', False))
