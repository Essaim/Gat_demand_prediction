from collections import defaultdict
import numpy as np
import torch.nn as nn
from utils.util import get_loss
import torch

def evaluate(predictions: np.ndarray, targets: np.ndarray):
    """
    evaluate model performance
    :param predictions: [n_samples, 12, n_nodes, n_features]
    :param targets: np.ndarray, shape [n_samples, 12, n_nodes, n_features]
    :return: a dict [str -> float]
    """
    assert targets.shape == predictions.shape and targets.shape[1] == 12, f'{targets.shape}/{predictions.shape}'
    n_samples = targets.shape[0]
    scores = defaultdict(dict)
    for horizon in range(12):
        y_true = np.reshape(targets[:, horizon], (n_samples, -1))
        y_pred = np.reshape(predictions[:, horizon], (n_samples, -1))
        scores['masked MAE'][f'horizon-{horizon}'] = masked_mae_np(y_pred, y_true, null_val=0.0)
        scores['masked RMSE'][f'horizon-{horizon}'] = masked_rmse_np(y_pred, y_true, null_val=0.0)
        scores['masked MAPE'][f'horizon-{horizon}'] = masked_mape_np(y_pred, y_true, null_val=0.0) * 100.0

    return scores

def nomask_evaluate(predictions, targets):
    assert targets.shape == predictions.shape and targets.shape[1] == 12, f'{targets.shape}/{predictions.shape}'
    predictions = torch.from_numpy(np.asarray(predictions)).float()
    targets = torch.from_numpy(np.asarray(targets)).float()
    n_samples = targets.shape[0]
    scores = defaultdict(dict)
    loss = get_loss('rmse')
    for horizon in range(12):
        y_true = np.reshape(targets[:, horizon], (n_samples, -1))
        y_pred = np.reshape(predictions[:, horizon], (n_samples, -1))
        # scores['masked MAE'][f'horizon-{horizon}'] = masked_mae_np(y_pred, y_true, null_val=0.0)
        scores['masked RMSE'][f'horizon-{horizon}'] = loss(y_true,y_pred).cpu().numpy()
        # scores['masked MAPE'][f'horizon-{horizon}'] = masked_mape_np(y_pred, y_true, null_val=0.0) * 100.0
    return scores


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)
