from typing import Iterable, List

from sklearn.metrics import roc_auc_score


def impression_auc(pred_lists: List[List[float]], label_lists: List[List[int]]) -> float:
    """
    Compute AUC per impression then average.
    """
    aucs = []
    for preds, labels in zip(pred_lists, label_lists):
        if len(set(labels)) < 2:
            continue  # skip if only one class present
        aucs.append(roc_auc_score(labels, preds))
    return float(sum(aucs) / len(aucs)) if aucs else 0.0
