import os
import torch
import shutil
from utils.data_container import get_data_loader_base
from models.model import create_model
from utils.util import get_optimizer,get_scheduler
from utils.train import train_decompose
from utils.normalization import Standard


def preprocessing(base_model_name,
                  conf,
                  loss,
                  graph,
                  data_category,
                  device,
                  data_set,
                  optimizer_name,
                  scheduler_name):

    if base_model_name == 'LinearDecompose':
        data_loader = get_data_loader_base(base_model_name = base_model_name,dataset=conf['data']['dataset'], batch_size=conf['batch_size_base'],
                                           _len=conf['data']['_len'], data_category=data_category, device=device)
        model, trainer = create_model(base_model_name, loss, conf['Base'][base_model_name], data_category, device,
                                      graph)
        save_folder = os.path.join('saves', f"{conf['name']}_{base_model_name}", f'{data_set}_{"".join(data_category)}')
        run_folder = os.path.join('run', f"{conf['name']}_{base_model_name}", f'{data_set}_{"".join(data_category)}')
        optimizer = get_optimizer(optimizer_name, model.parameters(), conf['optimizerbase'][optimizer_name]['lr'])
        scheduler = get_scheduler(scheduler_name, optimizer, **conf['scheduler'][scheduler_name])
        shutil.rmtree(save_folder, ignore_errors=True)
        os.makedirs(save_folder)
        shutil.rmtree(run_folder, ignore_errors=True)
        os.makedirs(run_folder)
        model = train_decompose(model=model,
                                dataloaders=data_loader,
                                trainer=trainer,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                folder=save_folder,
                                tensorboard_floder=run_folder,
                                device=device,
                                **conf['train'])
        model.load_state_dict(torch.load(f"{os.path.join(save_folder, 'best_model.pkl')}")['model_state_dict'])
        return model.encoder, model.decoder
    if base_model_name == 'SvdDecompose':
        data = get_data_loader_base(base_model_name = base_model_name,dataset=conf['data']['dataset'], batch_size=conf['batch_size_base'],
                                           _len=conf['data']['_len'], data_category=data_category, device=device)
        data = torch.from_numpy(data).float().to(device)
        save_folder = os.path.join('saves', f"{conf['name']}_{base_model_name}", f'{data_set}_{"".join(data_category)}')
        run_folder = os.path.join('run', f"{conf['name']}_{base_model_name}", f'{data_set}_{"".join(data_category)}')
        model,trainer = create_model(base_model_name, loss, conf['Base'][base_model_name], data_category, device,
                                      graph)
        shutil.rmtree(save_folder, ignore_errors=True)
        os.makedirs(save_folder)
        shutil.rmtree(run_folder, ignore_errors=True)
        os.makedirs(run_folder)
        model.decompose(data)
        return model.encoder,model.decoder