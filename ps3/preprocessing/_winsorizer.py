import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric features at quantiles learned from the training data.

    Parameters
    ----------
    lower_quantile : float
        Quantile (between 0 and 1) below which values are set to that quantile.
    upper_quantile : float
        Quantile (between 0 and 1) above which values are set to that quantile.
    """

    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        """Compute the clipping bounds of each column from ``X``."""
        X = np.asarray(X)
        self.lower_quantile_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        """Clip ``X`` at the bounds computed in ``fit``."""
        check_is_fitted(self)
        return np.clip(np.asarray(X), self.lower_quantile_, self.upper_quantile_)
