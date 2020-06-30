import numpy as np
import pandas as pd


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(
    train_sr=None, test_sr=None, target_sr=None, min_samples_leaf=1, smoothing=1, noise_level=0
):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    train_sr : training categorical feature as a pd.Series
    test_sr : test categorical feature as a pd.Series
    target_sr : target_sr data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(train_sr) == len(target_sr)
    assert train_sr.name == test_sr.name

    temp = pd.concat([train_sr, target_sr], axis=1)
    # Compute target_sr mean
    averages = temp.groupby(by=train_sr.name)[target_sr.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target_sr data
    prior = target_sr.mean()

    # The bigger the count the less full_avg is taken into account
    averages[target_sr.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply averages to test series
    ft_test_sr = pd.merge(
        test_sr.to_frame(test_sr.name),
        averages.reset_index().rename(columns={
            'index': target_sr.name,
            target_sr.name: 'average'
        }), on=test_sr.name, how='left'
    )['average'].rename(train_sr.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it

    ft_test_sr.index = test_sr.index

    return add_noise(ft_test_sr, noise_level)
