import numpy as np
import pytest

from ps3.preprocessing import Winsorizer


@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)

    winsorizer = Winsorizer(
        lower_quantile=lower_quantile, upper_quantile=upper_quantile
    )
    X_clipped = winsorizer.fit_transform(X)

    lower = np.quantile(X, lower_quantile)
    upper = np.quantile(X, upper_quantile)

    # fit stores the quantiles of the data
    assert winsorizer.lower_quantile_ == pytest.approx(lower)
    assert winsorizer.upper_quantile_ == pytest.approx(upper)

    # transform keeps the shape and puts every value inside the bounds
    assert X_clipped.shape == X.shape
    assert X_clipped.min() >= lower
    assert X_clipped.max() <= upper

    # values already inside the bounds are left unchanged
    inside = (X >= lower) & (X <= upper)
    np.testing.assert_array_equal(X_clipped[inside], X[inside])


def test_winsorizer_uses_bounds_from_fit():

    X_train = np.arange(101)  # 0, 1, ..., 100
    X_new = np.array([-50, 50, 500])

    winsorizer = Winsorizer(lower_quantile=0.1, upper_quantile=0.9).fit(X_train)

    # new data is clipped at the training quantiles (10 and 90), not its own
    np.testing.assert_array_equal(winsorizer.transform(X_new), [10, 50, 90])
