import argparse
import json
import os
import shutil

import h5py
import torch
import yaml
import torch.nn as nn
import torch

from models import create_model
from utils.train import train_model, train_baseline, train_decompose
from utils.util import get_optimizer, get_loss, get_scheduler
from utils.data_container import get_data_loader, get_data_loader_base
from utils.preprocess import preprocessing


# 250， 279
def train(conf, data_category):
    print(json.dumps(conf, indent=4))

    # device = torch.device(conf['device'])
    # print(device.index)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf['device'])
    device = torch.device(0)

    model_name = conf['model']['name']
    optimizer_name = conf['optimizer']['name']
    data_set = conf['data']['dataset']
    graph = h5py.File(os.path.join('data', data_set, 'all_graph.h5'))
    scheduler_name = conf['scheduler']['name']
    loss = get_loss(conf['loss']['name'])
    # data_category = conf['data']['data_category']

    loss.to(device)
    encoder, decoder = None, None
    if model_name == 'Costnet':
        base_model_name = conf['Base']['name']
        encoder, decoder = preprocessing(base_model_name, conf, loss, graph, data_category, device, data_set,
                                         optimizer_name, scheduler_name)

    model, trainer = create_model(model_name,
                                  loss,
                                  conf['model'][model_name],
                                  data_category,
                                  device,
                                  graph,
                                  encoder,
                                  decoder)

    optimizer = get_optimizer(optimizer_name, model.parameters(), conf['optimizer'][optimizer_name]['lr'])
    scheduler = get_scheduler(scheduler_name, optimizer, **conf['scheduler'][scheduler_name])
    if torch.cuda.device_count() > 1:
        print("use ", torch.cuda.device_count(), "GPUS")
        model = nn.DataParallel(model)
    else:
        model.to(device)

    save_folder = os.path.join('save', conf['name'], f'{data_set}_{"".join(data_category)}')
    run_folder = os.path.join('run', conf['name'], f'{data_set}_{"".join(data_category)}')

    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder)
    shutil.rmtree(run_folder, ignore_errors=True)
    os.makedirs(run_folder)

    with open(os.path.join(save_folder, 'config.yaml'), 'w+') as _f:
        yaml.safe_dump(conf, _f)

    data_loader, normal = get_data_loader(**conf['data'], data_category=data_category, device=device, model_name=model_name)

    if data_category == 2:
        train_model(model=model,
                    dataloaders=data_loader,
                    trainer=trainer,
                    optimizer=optimizer,
                    normal=normal,
                    scheduler=scheduler,
                    folder=save_folder,
                    tensorboard_folder=run_folder,
                    device=device,
                    **conf['train'])
    else:
        train_baseline(model=model,
                       dataloaders=data_loader,
                       trainer=trainer,
                       optimizer=optimizer,
                       normal=normal,
                       scheduler=scheduler,
                       folder=save_folder,
                       tensorboard_folder=run_folder,
                       device=device,
                       **conf['train'])


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', required=True, type=str,
    #                     help='Configuration filename for restoring the model.')
    # parser.add_argument('--resume', required=False, type=bool, default=False,
    #                     help='Resume.')
    # parser.add_argument('--test', required=False, type=bool, default=False,
    #                     help='Test.')
    #
    # args = parser.parse_args()
    con = 'costnet-config'
    data = ['bike']
    with open(os.path.join('config', f'{con}.yaml')) as f:
        conf = yaml.safe_load(f)
    train(conf, data)
