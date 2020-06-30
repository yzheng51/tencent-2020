import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from utils.encoder import target_encode


def kfold_stack_binary(kfold, classifier, x_train, y_train, x_test):
    """k fold stacking x_train and y_train for binary classification task
    given estimator `classifier` from sklearn api

    Args:
        kfold (sklearn.model_selection.StratifiedKFold): k fold spliter
        classifier (sklearn.base.BaseEstimator): classifer such as LogisticRegression
            from sklearn api
        x_train (numpy.ndarray): train features
        y_train (numpy.ndarray): train labels
        x_test (numpy.ndarray): test features

    Returns:
        stack_train (numpy.ndarray): stacked train feature
        stack_test (numpy.ndarray): stacked test feature
    """
    print(f'{classifier.__class__.__name__} Stacking...')
    stack_train = np.zeros(x_train.shape[0])
    stack_test = np.zeros(x_test.shape[0])

    for i, (train_idx, valid_idx) in enumerate(kfold.split(x_train, y_train)):
        print(f'Stacking: {i + 1}/{5}... ', end='')
        classifier.fit(x_train[train_idx], y_train[train_idx])
        try:
            valid_score = classifier.predict_proba(x_train[valid_idx])[:, 1]
            test_score = classifier.predict_proba(x_test)[:, 1]
        except AttributeError:
            valid_score = classifier._predict_proba_lr(x_train[valid_idx])[:, 1]
            test_score = classifier._predict_proba_lr(x_test)[:, 1]
        pred = np.where(valid_score <= 0.5, 0, 1)
        print(
            f'auc: {roc_auc_score(y_train[valid_idx], valid_score):.6f}... '
            f'acc: {accuracy_score(y_train[valid_idx], pred):.6f}... '
            f'log loss: {log_loss(y_train[valid_idx], valid_score):.6f}'
        )
        stack_train[valid_idx] = valid_score
        stack_test += test_score
    stack_test /= 5

    return stack_train, stack_test


def kfold_stack_multi(output_size, kfold, classifier, x_train, y_train, x_test):
    """k fold stacking x_train and y_train for multi-classification task
    given estimator `classifier` from sklearn api

    Args:
        output_size (int): number of classes of classification task
        kfold (sklearn.model_selection.StratifiedKFold): k fold spliter
        classifier (sklearn.base.BaseEstimator): classifer such as LogisticRegression
            from sklearn api
        x_train (numpy.ndarray): train features
        y_train (numpy.ndarray): train labels
        x_test (numpy.ndarray): test features

    Returns:
        stack_train (numpy.ndarray): stacked train feature
        stack_test (numpy.ndarray): stacked test feature
    """
    print(f'{classifier.__class__.__name__} Stacking...')
    stack_train = np.zeros((x_train.shape[0], output_size))
    stack_test = np.zeros((x_test.shape[0], output_size))

    for i, (train_idx, valid_idx) in enumerate(kfold.split(x_train, y_train)):
        print(f'Stacking: {i + 1}/{5}... ', end='')
        classifier.fit(x_train[train_idx], y_train[train_idx])
        try:
            valid_score = classifier.predict_proba(x_train[valid_idx])
            test_score = classifier.predict_proba(x_test)
        except AttributeError:
            valid_score = classifier._predict_proba_lr(x_train[valid_idx])
            test_score = classifier._predict_proba_lr(x_test)
        pred = valid_score.argmax(axis=1)
        print(
            f'acc: {accuracy_score(y_train[valid_idx], pred):.6f}... '
            f'log loss: {log_loss(y_train[valid_idx], valid_score):.6f}'
        )
        stack_train[valid_idx] = valid_score
        stack_test += test_score
    stack_test /= 5

    return stack_train, stack_test


def kfold_target_mean(
    kfold, train_sr, target_sr, test_sr, min_samples_leaf=1, smoothing=1, noise_level=0
):
    """k fold cv to create target encoding feature

    Args:
        kfold (sklearn.model_selection.StratifiedKFold): k fold spliter
        train_sr (pd.Series): train categorical feature
        target_sr (pd.Series): label
        test_sr (pd.Series): test categorical feature
        min_samples_leaf (int, optional): minimum samples to take category average
            into account. Defaults to 1.
        smoothing (int, optional): smoothing effect to balance categorical average
            vs prior. Defaults to 1.
        noise_level (int, optional): mean of the gaussian distribution. Defaults to 0.

    Returns:
        [type]: [description]
    """
    print(f'Target encoding {train_sr.name} with {target_sr.name}...')
    stack_train = np.zeros(train_sr.shape[0])
    stack_test = np.zeros(test_sr.shape[0])

    for train_idx, valid_idx in kfold.split(train_sr, target_sr):
        val_tf = target_encode(
            train_sr=train_sr[train_idx], target_sr=target_sr[train_idx], test_sr=train_sr[valid_idx],
            min_samples_leaf=min_samples_leaf, smoothing=smoothing, noise_level=noise_level
        )
        test_tf = target_encode(
            train_sr=train_sr[train_idx], target_sr=target_sr[train_idx], test_sr=test_sr,
            min_samples_leaf=min_samples_leaf, smoothing=smoothing, noise_level=noise_level
        )
        stack_train[valid_idx] = val_tf
        stack_test += test_tf
    stack_test /= 5

    return stack_train, stack_test
