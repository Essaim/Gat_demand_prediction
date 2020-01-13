import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import Dataset
from tqdm import tqdm
import copy, time
from utils.util import save_model, get_number_of_parameters
from collections import defaultdict
from utils.evaluate import nomask_evaluate


def train_model(model: nn.Module,
                   dataloaders,
                   optimizer,
                   normal,
                   scheduler,
                   folder: str,
                   trainer,
                   tensorboard_folder,
                   epochs: int,
                   device,
                   max_grad_norm: float = None,
                   early_stop_steps: float = None):
    save_path = os.path.join(folder, 'best_model.pkl')
    bike_normal, taxi_normal = normal[0], normal[1]
    if os.path.exists(save_path):
        save_dict = torch.load(save_path)

        model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])

        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0

    phases = ['train', 'validate', 'test']

    writer = SummaryWriter(tensorboard_folder)

    since = time.perf_counter()

    model = model.to(device)
    print(model)
    print(f'Trainable parameters: {get_number_of_parameters(model)}.')

    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):

            running_loss, running_metrics = defaultdict(float), dict()
            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                steps, predictions, running_targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(dataloaders[phase]))
                for step, (inputs, targets) in tqdm_loader:
                    running_targets.append(targets.numpy())

                    with torch.no_grad():
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, loss = trainer.train(inputs, targets, phase)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            if max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    with torch.no_grad():
                        predictions.append(outputs.cpu().numpy())

                    running_loss[phase] += loss * len(targets)
                    steps += len(targets)

                    tqdm_loader.set_description(
                        f'{phase:5} epoch: {epoch:3}, {phase:5} loss: {running_loss[phase] / steps:3.6}')

                    # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                    torch.cuda.empty_cache()
                # 性能
                running_metrics[phase] = nomask_evaluate(np.concatenate(predictions), np.concatenate(running_targets))

                if phase == 'validate':
                    if running_loss['validate'] <= best_val_loss:
                        best_val_loss = running_loss['validate']
                        save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         best_val_loss=best_val_loss,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        print(f'Better model at epoch {epoch} recorded.')
                    elif epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')

            scheduler.step(running_loss['train'])

            for metric in running_metrics['train'].keys():
                for phase in phases:
                    for key, val in running_metrics[phase][metric].items():
                        writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)
            writer.add_scalars('Loss', {
                f'{phase} loss': running_loss[phase] / len(dataloaders[phase].dataset) for phase in phases},
                               global_step=epoch)
    except (ValueError, KeyboardInterrupt):
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")

        model.load_state_dict(save_dict['model_state_dict'])

        save_model(save_path, **save_dict)
        print(f'model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')

    return model


def train_baseline(model: nn.Module,
                   dataloaders,
                   optimizer,
                   normal,
                   scheduler,
                   folder: str,
                   trainer,
                   tensorboard_folder,
                   epochs: int,
                   device,
                   max_grad_norm: float = None,
                   early_stop_steps: float = None):
    # dataloaders = get_dataloaders(datasets, batch_size)
    # scaler = ZScoreScaler(datasets['train'].mean[0], datasets['train'].std[0])

    save_path = os.path.join(folder, 'best_model.pkl')

    if os.path.exists(save_path):
        save_dict = torch.load(save_path)

        model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])

        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0

    phases = ['train', 'validate', 'test']

    writer = SummaryWriter(tensorboard_folder)

    since = time.perf_counter()

    model = model.to(device)
    print(model)
    print(f'Trainable parameters: {get_number_of_parameters(model)}.')

    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):

            running_loss, running_metrics = defaultdict(float), dict()
            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                steps, predictions, running_targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(dataloaders[phase]))
                for step, (inputs, targets) in tqdm_loader:
                    running_targets.append(targets.numpy())

                    with torch.no_grad():
                        # inputs[..., 0] = scaler.transform(inputs[..., 0])
                        inputs = inputs.to(device)
                        # targets[..., 0] = scaler.transform(targets[..., 0])
                        targets = targets.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, loss = trainer.train(inputs, targets, phase)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            if max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    with torch.no_grad():
                        # predictions.append(scaler.inverse_transform(outputs).cpu().numpy())
                        predictions.append(outputs.cpu().numpy())

                    running_loss[phase] += loss * len(targets)
                    steps += len(targets)

                    tqdm_loader.set_description(
                        f'{phase:5} epoch: {epoch:3}, {phase:5} loss: {running_loss[phase] / steps:3.6}')

                    # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                    torch.cuda.empty_cache()
                # 性能
                running_metrics[phase] = nomask_evaluate(np.concatenate(predictions), np.concatenate(running_targets))

                if phase == 'validate':
                    if running_loss['validate'] <= best_val_loss:
                        best_val_loss = running_loss['validate']
                        save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         best_val_loss=best_val_loss,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        print(f'Better model at epoch {epoch} recorded.')
                    elif epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')

            scheduler.step(running_loss['train'])

            for metric in running_metrics['train'].keys():
                for phase in phases:
                    for key, val in running_metrics[phase][metric].items():
                        writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)
            writer.add_scalars('Loss', {
                f'{phase} loss': running_loss[phase] / len(dataloaders[phase].dataset) for phase in phases},
                               global_step=epoch)
    except (ValueError, KeyboardInterrupt):
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")

        model.load_state_dict(save_dict['model_state_dict'])

        save_model(save_path, **save_dict)
        print(f'model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')

    return model


def train_decompose(model: nn.Module,
                    dataloaders,
                    optimizer,
                    scheduler,
                    folder: str,
                    trainer,
                    tensorboard_floder,
                    epochs: int,
                    device,
                    max_grad_norm: float = None,
                    early_stop_steps: float = None):
    # dataloaders = get_dataloaders(datasets, batch_size)
    # scaler = ZScoreScaler(datasets['train'].mean[0], datasets['train'].std[0])

    save_path = os.path.join(folder, 'best_model.pkl')

    if os.path.exists(save_path):
        save_dict = torch.load(save_path)

        model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])

        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0

    phases = ['train', 'validate', 'test']

    writer = SummaryWriter(tensorboard_floder)

    since = time.perf_counter()

    model = model.to(device)

    print(model)
    print(f'Trainable parameters: {get_number_of_parameters(model)}.')

    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):

            running_loss, running_metrics = defaultdict(float), dict()
            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                steps, predictions, running_targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(dataloaders[phase]))
                for step, (inputs, targets) in tqdm_loader:
                    running_targets.append(targets.numpy())

                    with torch.no_grad():
                        # inputs[..., 0] = scaler.transform(inputs[..., 0])
                        inputs = inputs.to(device)
                        # targets[..., 0] = scaler.transform(targets[..., 0])
                        targets = targets.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, loss = trainer.train(inputs, targets, phase)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            if max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    with torch.no_grad():
                        # predictions.append(scaler.inverse_transform(outputs).cpu().numpy())
                        predictions.append(outputs.cpu().numpy())
                    running_loss[phase] += loss * len(targets)
                    steps += len(targets)

                    tqdm_loader.set_description(
                        f'{phase:5} epoch: {epoch:3}, {phase:5} loss: {running_loss[phase] / steps:3.6}')

                    # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                    torch.cuda.empty_cache()
                # 性能
                # running_metrics[phase] = trainer.loss(torch.cat(predictions), torch.cat(running_targets)).cpu().numpy()

                if phase == 'validate':
                    if running_loss['validate'] < best_val_loss:
                        best_val_loss = running_loss['validate']
                        save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         best_val_loss=best_val_loss,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        print(f'Better model at epoch {epoch} recorded.')
                    elif epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')

            scheduler.step(running_loss['train'])

            # for metric in running_metrics['train'].keys():
            #     for phase in phases:
            #         for key, val in running_metrics[phase][metric].items():
            #             writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)
            # writer.add_scalars('Loss', {
            #     f'{phase} loss': running_loss[phase] / len(dataloaders[phase].dataset) for phase in phases},
            #                    global_step=epoch)
    except (ValueError, KeyboardInterrupt):
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")

        model.load_state_dict(save_dict['model_state_dict'])
        print(save_path)
        save_model(save_path, **save_dict)
        print(f'model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')

    return model
