import numpy as np
import pandas as pd

class DistanceNormalizer:
    def __init__(self, metrics_df: pd.DataFrame, method='zscore'):
        self.df = metrics_df
        self.method = method
        self.params = {}  # Store mean/std or min/max per metric

        self._compute_params()

    def _compute_params(self):
        for col in self.df.columns:
            col_data = self.df[col].values
            if self.method == 'zscore':
                mean = np.mean(col_data)
                std = np.std(col_data)
                self.params[col] = {"mean": mean, "std": std}
            elif self.method == 'minmax':
                min_val = np.min(col_data)
                max_val = np.max(col_data)
                self.params[col] = {"min": min_val, "max": max_val}

    def normalize(self, metric_name: str, value: float) -> float:
        param = self.params[metric_name]

        if self.method == 'zscore':
            normalized_value = (value - param["mean"]) / param["std"]
        elif self.method == 'minmax':
            normalized_value = (value - param["min"]) / (param["max"] - param["min"])
        else:
            raise ValueError("Unsupported normalization method.")

        return normalized_value
