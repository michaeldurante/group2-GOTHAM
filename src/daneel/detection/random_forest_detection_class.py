import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ExoplanetTransitAnalyzer:
    """Loads a pre-trained Random Forest model and uses it to analyze light curves for exoplanet transits"""

    def __init__(self, model_path="tess_rf_model.joblib", threshold_path="rf_optimal_threshold.npy", n_bins=1000):
        
        self.n_bins = n_bins
        self.flux_cols = [f"flux_{i:04d}" for i in range(n_bins)]
        self.flux_err_cols = [f"flux_err_{i:04d}" for i in range(n_bins)]
        self.model = joblib.load(model_path)
        self.threshold = float(np.load(threshold_path)) if threshold_path else 0.5