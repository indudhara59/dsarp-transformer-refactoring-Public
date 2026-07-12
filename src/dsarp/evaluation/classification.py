"""Comprehensive classification metrics."""
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, average_precision_score, roc_auc_score


def classification_metrics(y_true: Any, y_pred: Any, probabilities: Any | None = None) -> dict[str, Any]:
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); macro=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0); weighted=precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0); per=precision_recall_fscore_support(y_true,y_pred,average=None,zero_division=0)
    result={"macro_precision":float(macro[0]),"macro_recall":float(macro[1]),"macro_f1":float(macro[2]),"weighted_f1":float(weighted[2]),"accuracy":float(accuracy_score(y_true,y_pred)),"balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),"per_class_precision":per[0].tolist(),"per_class_recall":per[1].tolist(),"confusion_matrix":confusion_matrix(y_true,y_pred).tolist()}
    if probabilities is not None and len(np.unique(y_true))==2:
        scores=np.asarray(probabilities); scores=scores[:,1] if scores.ndim==2 else scores; result.update(roc_auc=float(roc_auc_score(y_true,scores)),pr_auc=float(average_precision_score(y_true,scores)))
    return result

