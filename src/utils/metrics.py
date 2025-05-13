import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def compute_confusion_matrix(preds, labels, num_classes, ignore_index=None):
    """
    preds, labels: 1D numpy arrays of shape (N,)
    ignore_index: integer label to skip (e.g. 0), or None.
    returns: num_classes x num_classes matrix, cm[i,j]=# true=i predicted=j
    """
    mask = np.ones_like(labels, dtype=bool)
    if ignore_index is not None:
        mask = labels != ignore_index

    valid_labels = labels[mask]
    valid_preds  = preds[mask]

    hist = np.bincount(
        num_classes * valid_labels.astype(int) + valid_preds.astype(int),
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    return hist


def pixel_accuracy(conf_mat):
    """
    Pixel Accuracy: total correct pixels / total pixels
    """
    correct = np.diag(conf_mat).sum()
    total   = conf_mat.sum()
    return float(correct) / max(total, 1)


def per_class_accuracy(conf_mat):
    """
    Accuracy per class: tp / total_true for each class
    returns array of shape (num_classes,)
    """
    tp = np.diag(conf_mat).astype(float)
    total_true = conf_mat.sum(axis=1).astype(float)
    return tp / np.maximum(total_true, 1)


def mean_accuracy(conf_mat):
    """
    Mean Accuracy: average of per-class accuracies
    """
    return float(np.nanmean(per_class_accuracy(conf_mat)))


def per_class_iou(conf_mat):
    tp = np.diag(conf_mat)
    fp = conf_mat.sum(axis=0) - tp
    fn = conf_mat.sum(axis=1) - tp
    denom = tp + fp + fn
    return tp / np.maximum(denom, 1)


def mean_iou(conf_mat):
    """
    Mean Intersection over Union across classes
    """
    return float(np.nanmean(per_class_iou(conf_mat)))


def frequency_weighted_iou(conf_mat):
    """
    Frequency Weighted IoU: weights each class IoU by its frequency
    """
    freq = conf_mat.sum(axis=1).astype(float)
    iou  = per_class_iou(conf_mat)
    total = freq.sum()
    return float((freq * iou).sum() / max(total, 1))


def worst_class_iou(conf_mat):
    return float(np.nanmin(per_class_iou(conf_mat)))


def expected_calibration_error(probs, correctness, n_bins=10, ignore_mask=None):
    """
    probs: 1D array of predicted confidences [0…1]
    correctness: 1D binary array (1 if pred==gt, 0 else)
    ignore_mask: optional boolean mask (same shape) where True means keep
    """
    if ignore_mask is not None:
        probs = probs[ignore_mask]
        correctness = correctness[ignore_mask]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bin_edges[i]) & (probs <= bin_edges[i+1])
        if mask.any():
            acc_bin  = correctness[mask].mean()
            conf_bin = probs[mask].mean()
            ece += np.abs(acc_bin - conf_bin) * mask.mean()
    return float(ece)


def pixel_auroc(probs, correctness, ignore_mask=None):
    """
    probs: 1D confidences, correctness: 1D binary (1=correct,0=wrong)
    ignore_mask: optional boolean mask to drop void pixels
    """
    if ignore_mask is not None:
        probs       = probs[ignore_mask]
        correctness = correctness[ignore_mask]

    try:
        return float(roc_auc_score(correctness, probs))
    except ValueError:
        # if only one class present in `correctness`
        return float('nan')

