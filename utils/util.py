from torch import optim
import torch.nn as nn
import os
import pickle
import torch
import numpy as np

def get_optimizer(opt_name,par, lr):
    if opt_name == 'Adam':
        return optim.Adam(par, lr)
    else:
        print("no optimizer")

def get_loss(loss_name):
    if loss_name == 'rmse':
        return RMSELoss()


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, truth, predict):
        return self.mse_loss(truth, predict) ** 0.5

def save_model(path: str, **save_dict):

    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data
def get_number_of_parameters(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def get_scheduler(name, optimizer, **kwargs):
    return getattr(optim.lr_scheduler, name)(optimizer, **kwargs)