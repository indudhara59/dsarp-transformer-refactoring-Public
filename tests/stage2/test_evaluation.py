import numpy as np
from dsarp.evaluation.calibration import TemperatureScaler,calibration_metrics
from dsarp.evaluation.classification import classification_metrics
from dsarp.evaluation.regression import regression_metrics


def test_metrics_and_calibration():
    classification=classification_metrics([0,1,1],[0,1,0],[.1,.9,.4]); assert "macro_f1" in classification and "pr_auc" in classification
    regression=regression_metrics([.1,.5,.9],[.2,.4,.8]); assert regression["mae"]>=0
    scaler=TemperatureScaler().fit(np.array([-2.,2.,1.]),np.array([0,1,1])); probabilities=scaler.predict_proba([-2.,2.,1.]); metrics=calibration_metrics([0,1,1],probabilities); assert 0<=metrics["ece"]<=1 and 0<=metrics["brier_score"]<=1
