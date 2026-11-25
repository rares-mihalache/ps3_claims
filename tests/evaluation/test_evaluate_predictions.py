import numpy as np
import pytest

from ps3.evaluation import evaluate_predictions


def test_output_format():

    metrics = evaluate_predictions([1, 2, 3], [1, 2, 3])

    assert list(metrics.index) == ["bias", "deviance", "mae", "rmse", "gini"]
    assert list(metrics.columns) == ["value"]


def test_perfect_predictions_have_no_error():

    actuals = np.array([0.5, 1.0, 2.0, 4.0])
    metrics = evaluate_predictions(actuals, actuals, sample_weight=[1, 2, 3, 4])

    for metric in ["bias", "deviance", "mae", "rmse"]:
        assert metrics.loc[metric, "value"] == pytest.approx(0)


def test_known_values():

    # errors are [1, 0, -1, 2]; mean prediction 3, mean actual 2.5
    metrics = evaluate_predictions([1, 2, 3, 4], [2, 2, 2, 6])

    assert metrics.loc["bias", "value"] == pytest.approx(3 / 2.5 - 1)
    assert metrics.loc["mae", "value"] == pytest.approx(1)
    assert metrics.loc["rmse", "value"] == pytest.approx(np.sqrt(6 / 4))


def test_weight_equals_repeating_a_row():

    # a weight of 2 should count the same as having that observation twice
    weighted = evaluate_predictions([1, 2, 3], [2, 2, 5], sample_weight=[2, 1, 1])
    repeated = evaluate_predictions([1, 1, 2, 3], [2, 2, 2, 5])

    for metric in ["bias", "deviance", "mae", "rmse"]:
        assert weighted.loc[metric, "value"] == pytest.approx(
            repeated.loc[metric, "value"]
        )


def test_no_weights_means_equal_weights():

    actuals, predictions = [1, 2, 3, 4], [1.5, 2, 2.5, 5]

    np.testing.assert_allclose(
        evaluate_predictions(actuals, predictions).values,
        evaluate_predictions(actuals, predictions, sample_weight=[1, 1, 1, 1]).values,
    )


def test_gini_rewards_good_ranking():

    actuals = np.array([0, 0, 1, 2, 5, 10], dtype=float)

    good = evaluate_predictions(actuals, actuals + 1).loc["gini", "value"]
    reversed_order = evaluate_predictions(actuals, -actuals + 20).loc["gini", "value"]

    assert good > 0
    assert reversed_order < 0