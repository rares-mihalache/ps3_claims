import numpy as np
import pandas as pd
from sklearn.metrics import auc, mean_tweedie_deviance


def _lorenz_curve(actuals, predictions, sample_weight):
    """Return the Lorenz curve used in ps3_script.

    Policies are ordered from lowest to highest predicted risk. The x-axis is
    the share of policies and the y-axis the share of total claims they cause.
    """
    ranking = np.argsort(predictions)
    ranked_claims = actuals[ranking] * sample_weight[ranking]
    cumulated_claims = np.cumsum(ranked_claims) / ranked_claims.sum()
    cumulated_samples = np.linspace(0, 1, len(cumulated_claims))
    return cumulated_samples, cumulated_claims


def evaluate_predictions(actuals, predictions, sample_weight=None, tweedie_power=1.5):
    """Compute performance metrics for a model's predictions.

    Parameters
    ----------
    actuals : array-like
        True outcome values (here the pure premium).
    predictions : array-like
        The model's predictions of the outcome.
    sample_weight : array-like, optional
        Weight of each observation (here the exposure). Equal weights if None.
    tweedie_power : float
        Power of the Tweedie distribution used for the deviance.

    Returns
    -------
    pd.DataFrame
        One row per metric, with the metric names as index:

        - bias: weighted mean prediction relative to the weighted mean actual
          (0 = unbiased, -0.07 = predictions are 7% too low on average)
        - deviance: weighted mean Tweedie deviance
        - mae: weighted mean absolute error
        - rmse: weighted root mean squared error
        - gini: Gini coefficient from the Lorenz curve (higher = better ranking
          of risks)
    """
    actuals = np.asarray(actuals, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if sample_weight is None:
        sample_weight = np.ones_like(actuals)
    sample_weight = np.asarray(sample_weight, dtype=float)

    errors = predictions - actuals

    mean_prediction = np.average(predictions, weights=sample_weight)
    mean_actual = np.average(actuals, weights=sample_weight)
    bias = mean_prediction / mean_actual - 1

    deviance = mean_tweedie_deviance(
        actuals, predictions, sample_weight=sample_weight, power=tweedie_power
    )
    mae = np.average(np.abs(errors), weights=sample_weight)
    rmse = np.sqrt(np.average(errors**2, weights=sample_weight))

    lorenz_x, lorenz_y = _lorenz_curve(actuals, predictions, sample_weight)
    gini = 1 - 2 * auc(lorenz_x, lorenz_y)

    return pd.DataFrame(
        {"value": [bias, deviance, mae, rmse, gini]},
        index=pd.Index(["bias", "deviance", "mae", "rmse", "gini"], name="metric"),
    )