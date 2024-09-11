import os
import anndata
import pickle

from networkx import draw
project_path = '/media/workspace/caoguangshuo/scPlantGPT'
os.chdir(f'{project_path}/s03_scPlantGPT/trainer')
import copy
import torch 
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import h5py
import sys
import pandas as pd
from torch import nn
import time
import datetime
import wandb
from sklearn.metrics.pairwise import cosine_similarity
import scipy.cluster.hierarchy as sch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json

sys.path.insert(0, "../")
from utils import *
from scplantgpt.model import TransformerModel
from moudules import  train, evaluate, test, data_loader, masked_mse_loss


start_time = time.time()
config = load_config('./config.json')
seed = config.seed
set_seed(seed)

species = 'cross' # Ara cross Maize Rice ## cls_decoder
data_name = 'Ara' #Ara Zea Rice

data_path = '/media/workspace/caoguangshuo/scPlantGPT/s05.bgi_test_data/plantcell_shanni/minor_cell_data'
result_path = f'{project_path}/s03_scPlantGPT/result/clean_label/scAraGPT/plantcell_shanni'

hyperparameter_defaults = dict(
    parallel=False,
    epochs=15, #pretrain 30
    batch_size=780, # 64 
    lr=1e-4,
    ntoken= 185622, #Ara 45416, cross 185622
    nctype= 44,
    nbatch_effect= 238,#156,
    ecs_threshold=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    layer_size=512,
    hlayer_size=512,
    nlayers=6,
    nhead=8,
    nlayers_cls=3,
    dropout=0.5,
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    fast_transformer=True,
    explicit_zero_prob=False,
    pre_norm=True,
    Species=species,
)

current_time = datetime.datetime.now()
timestamp = current_time.strftime("%YY%mM%dD%HH%MM%SS")
run = wandb.init(
    config=hyperparameter_defaults,
    project="scPlantGPT_test",
    entity="aibio",
    group=f"{config.train_strategy}_{config.input_emb_style}_{timestamp}",
)
model_config = wandb.config

if model_config.parallel:   
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)

    world_size = torch.cuda.device_count()
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        logger = setup_custom_logger(f"L01.{config.train_strategy}_{config.input_emb_style}_{timestamp}", './../Log/')
    else:
        class EmptyLogger:
            def info(self, *args, **kwargs):
                pass
            def debug(self, *args, **kwargs):
                pass
            def error(self, *args, **kwargs):
                pass
        logger = EmptyLogger()
    device = torch.device("cuda", local_rank)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    device = torch.device('cuda', 0)
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%YY%mM%dD%HH%MM%SS")
    logger = setup_custom_logger(f"L01.{config.train_strategy}_{config.input_emb_style}_{timestamp}", './../Log/')
    
log_fig(logger)
logger.info(f"Begin to **{config.train_strategy}** model... ")

if config.input_emb_style == "category":
    n_input_bins = config.n_bins  + 2 # pad_value:-2, cls_value:0, masked_value:-1
else:
    n_input_bins = config.n_bins
    
model = TransformerModel(
    ntoken=model_config.ntoken, 
    d_model=model_config.layer_size, 
    nhead=model_config.nhead, 
    d_hid=model_config.hlayer_size,
    nlayers=model_config.nlayers, 
    nlayers_cls=model_config.nlayers_cls, 
    n_cls=model_config.nctype, 
    dropout=model_config.dropout, 
    pad_value=int(config.pad_value),
    pad_token_id=config.pad_token_id,  
    num_batch_labels=model_config.nbatch_effect, 
    input_emb_style=config.input_emb_style, 
    n_input_bins=n_input_bins, 
    use_fast_transformer=model_config.fast_transformer, 
    pre_norm=model_config.pre_norm,)

model.to(device)

load_model = True
freeze = False
if load_model:
    
    model_name=f"./model_param/best_model.pth"
    try:
        model.load_state_dict(torch.load(model_name))
        logger.info(f"Loading all model params from {model_name}")
        if freeze:
            for name, param in model.named_parameters():
                if 'decoder' not in name:  # 冻结
                    param.requires_grad = False
                    logger.info(f"Freeze {name}")
                
            for name, param in model.named_parameters():
                logger.info(f'{name}: requires_grad={param.requires_grad}')

    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_name)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if freeze:
            for name, param in model.named_parameters():
                if 'decoder' not in name:  # 冻结
                    param.requires_grad = False
                    logger.info(f"Freeze {name}")
            for name, param in model.named_parameters():
                logger.info(f'{name}: requires_grad={param.requires_grad}')

if model_config.parallel:
    logger.info("Using DistributedDataParallel!")
    model = DDP(model, find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)
        
logger.info(f"loading data from {data_path}")
train_sampler, train_loader, train_metadata = data_loader(data_path, data_type='train', start_chunk=1, end_chunk=1, batch_size=model_config.batch_size, logger=logger, append_cls=True, parallel=model_config.parallel)
valid_sampler, valid_loader, valid_metadata = data_loader(data_path, data_type='valid', start_chunk=1, num_chunks=1, batch_size=model_config.batch_size,logger=logger, append_cls=True, parallel=model_config.parallel)
test_sampler, test_loader, test_metadata = data_loader(data_path,  data_type='test',start_chunk=1, num_chunks=1, batch_size=model_config.batch_size,logger=logger, append_cls=True, parallel=model_config.parallel)


def cal_result(data_name, species, model, train_loader, train_metadata, valid_loader, valid_meatadata, test_loader, test_metadata, device, config, logger,result_path,fine_tune,):
    result_path = os.path.join(result_path, data_name)
    os.makedirs(result_path, exist_ok=True)
    logger.info(f"test on the {data_name} dataset:")
    
    if fine_tune:
        criterion_cls = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config.lr, eps=1e-4 if config.amp else 1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=model_config.schedule_ratio)
        scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

        best_val_loss = float("inf")
        for epoch in range(model_config.epochs):
            
            epoch_start_time = time.time()
            train(model, valid_loader, criterion_cls, scaler, optimizer, scheduler, device, config, logger, epoch, model_config.parallel)
            epoch_end_time = time.time()
            logger.info(f"Epoch {epoch} time: {epoch_end_time - epoch_start_time}")
    
            val_loss = evaluate(model, test_loader, criterion_cls, device, config, logger, species, epoch)
            torch.save(model.state_dict(), os.path.join(result_path,f"model_{data_name}_finetune.pth"))
            logger.info(f"Save the model in {result_path}/model_{data_name}_finetune.pth")
    cell_types_predictions, cell_types_labels, cell_names, probabilities, cell_embeddings, batch_labels_list = test(model, train_loader, train_metadata, device, config, logger)
    
    predict_end_time = time.time()
    logger.info(f"Using time to predict: {predict_end_time - start_time}")

    umap_embedding = compute_umap_embeddings(cell_embeddings)
    
    reducer = TSNE()
    tsne_embedding = reducer.fit_transform(cell_embeddings)

    result_df = pd.DataFrame({
        'cell_types_predictions': cell_types_predictions,
        'cell_types_labels': cell_types_labels,
        'cell_names': cell_names,
        'batch_labels': batch_labels_list,
        })

    result_df.to_csv(os.path.join(result_path, f'{data_name}_finetune_{fine_tune}_predict_results_{os.path.basename (model_name)}.csv'), index=False)

    with open(os.path.join(result_path,f'cell_embeddings_finetune_{fine_tune}_{data_name}.pkl'), 'wb') as f:
        pickle.dump(cell_embeddings, f)
    with open(os.path.join(result_path,f'umap_embeddings_finetune_{fine_tune}_{data_name}.pkl'), 'wb') as f:
        pickle.dump(umap_embedding, f)
    with open(os.path.join(result_path,f'tsne_embeddings_finetune_{fine_tune}_{data_name}.pkl'), 'wb') as f:
        pickle.dump(tsne_embedding, f)
        

    logger.info(f"Save the adata in {result_path}/adata_without_X_finetune_{fine_tune}.h5ad")

cal_result(data_name, model_config.Species, model, train_loader, train_metadata, valid_loader, valid_metadata, test_loader, test_metadata, device, config, logger, result_path, fine_tune=False, cal_similarity=False)
# cal_result(data_name, model_config.Species, model, train_loader, train_metadata, valid_loader, valid_metadata, test_loader, test_metadata, device, config, logger, result_path, fine_tune=True, cal_similarity=False)

if model_config.parallel:
    torch.distributed.destroy_process_group()
wandb.finish()
plt.close('all')

end_time = time.time()
logger.info(f"Train time: {end_time - start_time}")
logger.info("Train finished!")