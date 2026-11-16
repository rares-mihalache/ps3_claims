import hashlib

import numpy as np


def create_sample_column(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str or list of str
        Name(s) of ID column(s)
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    if isinstance(id_column, str):
        id_column = [id_column]

    if len(id_column) == 1 and np.issubdtype(df[id_column[0]].dtype, np.integer):
        # Integer ID: use modulo 100 directly to get 100 groups
        buckets = df[id_column[0]] % 100
    else:
        # String ID or multiple columns: use a stable hash, then modulo 100
        key = df[id_column].astype(str).agg("_".join, axis=1)
        buckets = key.apply(
            lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 100
        )

    # Groups below the cutoff go to train, the rest to test
    df["sample"] = np.where(buckets < training_frac * 100, "train", "test")

    return df