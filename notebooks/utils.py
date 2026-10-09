from pathlib import Path

import numpy as np


def load():
    d = np.load(Path(__file__).resolve().parent.parent / "data" / "embeddings.npz")
    return d["I"], d["T"], d["Y"]


def center_normalize(X):
    X = X - X.mean(0)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# one fixed colour per modality, shared by all notebooks
COLOR_I = "#1f77b4"
COLOR_T = "#d95f02"
COLOR_IT = "#2ca02c"
COLOR_HIGHLIGHT = "#c0392b"


def label_colors(Y):
    """Fixed label -> colour dict (tab20 + tab20b); label 34 is light grey.

    Colours are assigned by rank among the sorted non-34 labels of the full
    label set, so the mapping is identical in every plot.
    """
    import matplotlib

    palette = list(matplotlib.colormaps["tab20"].colors) + list(matplotlib.colormaps["tab20b"].colors)
    labels = [int(l) for l in np.unique(Y) if l != 34]
    colors = {l: palette[i % len(palette)] for i, l in enumerate(labels)}
    colors[34] = "#cccccc"
    return colors


def get_split():
    """Load the fixed train/test split from results/split.npz.

    Returns train_idx, test_idx, groups (group id per row, from identical T rows).
    """
    d = np.load(Path(__file__).resolve().parent.parent / "results" / "split.npz")
    return d["train_idx"], d["test_idx"], d["groups"]


def fit_center(X_train):
    """Column mean of X_train. Fit on the train rows only to avoid test leakage."""
    return X_train.mean(0)


def apply_center_normalize(X, mean):
    """Subtract the given mean, then L2-normalise rows.

    The mean must come from fit_center on the train rows only, so no test
    information leaks into the preprocessing.
    """
    X = X - mean
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def load_split_data():
    """Return a dict with the fixed split, preprocessed with train-only statistics.

    I_tr, I_te, T_tr, T_te, Y_tr, Y_te: centred with the TRAIN mean only
    (separately for I and T) then row-normalised, so the mean is fitted on
    train only to avoid test leakage. Also contains raw arrays
    (I_tr_raw, I_te_raw, T_tr_raw, T_te_raw), the train means (I_mean, T_mean)
    and the split indices (train_idx, test_idx).
    """
    I, T, Y = load()
    tr, te, _ = get_split()
    I_mean, T_mean = fit_center(I[tr]), fit_center(T[tr])
    return {
        "I_tr": apply_center_normalize(I[tr], I_mean), "I_te": apply_center_normalize(I[te], I_mean),
        "T_tr": apply_center_normalize(T[tr], T_mean), "T_te": apply_center_normalize(T[te], T_mean),
        "Y_tr": Y[tr], "Y_te": Y[te],
        "I_tr_raw": I[tr], "I_te_raw": I[te], "T_tr_raw": T[tr], "T_te_raw": T[te],
        "I_mean": I_mean, "T_mean": T_mean,
        "train_idx": tr, "test_idx": te,
    }
