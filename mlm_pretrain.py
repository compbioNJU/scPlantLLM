import os
project_path = '/media/workspace/caoguangshuo/scPlantGPT'
os.chdir(f'{project_path}/s03_scPlantGPT/trainer')
import copy
import torch 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import sys
import pandas as pd
from torch import nn
import time
from loss import masked_mse_loss
from s01_pretrain import data_loader, pretrain, evaluate, generation_evaluate, pretrain_generation
import datetime
import wandb
sys.path.insert(0, "../")
from utils import set_seed, setup_custom_logger, load_config
from model.generation_model import TransformerModel

start_time = time.time()
# config = load_config('../Util/config.json')
config = load_config('../Util/pretrain_config.json')
seed = config.seed
set_seed(seed)

hyperparameter_defaults = dict(
    parallel=True,
    epochs=20, #pretrain 30
    batch_size=64,
    lr=1e-4,
    ntoken= 185622, #Ara 45416, cross 185622
    nctype= 100, #54,
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
)
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%YY%mM%dD%HH%MM%SS")
run = wandb.init(
    config=hyperparameter_defaults,
    project="scPlantGPT",
    entity="aibio",
    group=f"{config.train_strategy}_{config.input_emb_style}_{timestamp}",
)
model_config = wandb.config

if model_config.parallel:   
    torch.cuda.set_device(0)
    device = torch.device('cuda', 1)

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
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%YY%mM%dD%HH%MM%SS")
    logger = setup_custom_logger(f"L01.{config.train_strategy}_{config.input_emb_style}_{timestamp}", './../Log/')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# log_fig(logger)
logger.info(f"Begin to **{config.train_strategy}** model... ")

if config.input_emb_style == "category":
    n_input_bins = config.n_bins + 2 # pad_value:-2, cls_value:0, masked_value:-1
else:
    n_input_bins = config.n_bins
# logger.info(f"model init")
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
    use_batch_labels=config.use_batch_labels, 
    num_batch_labels=model_config.nbatch_effect, 
    input_emb_style=config.input_emb_style, 
    n_input_bins= n_input_bins, 

    use_fast_transformer=model_config.fast_transformer, 
    pre_norm=model_config.pre_norm,)

model.to(device)


        
if model_config.parallel:
    logger.info("Using DistributedDataParallel!")
    model = DDP(model, find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)


data_path = f'{project_path}/s03_scPlantGPT/cross_data/independent_clean_label/Ara_train'

logger.info(f"loading data from {data_path}")
train_sampler_12, train_loader_12, _ = data_loader(data_path, data_type='train', start_chunk=1, end_chunk=2, batch_size=model_config.batch_size, logger=logger, append_cls=True, parallel=model_config.parallel)
train_sampler_34, train_loader_34, _ = data_loader(data_path, data_type='train', start_chunk=3, end_chunk=3, batch_size=model_config.batch_size, logger=logger, append_cls=True, parallel=model_config.parallel)
valid_sampler, valid_loader, _ = data_loader(data_path, data_type='valid', start_chunk=1, num_chunks=1, batch_size=model_config.batch_size,logger=logger, append_cls=True, parallel=model_config.parallel)
test_sampler, test_loader, _ = data_loader(data_path,  data_type='test',start_chunk=1, num_chunks=1, batch_size=model_config.batch_size,logger=logger, append_cls=True, parallel=model_config.parallel)

criterion_gep_gepc = masked_mse_loss
optimizer = torch.optim.Adam(
    model.parameters(), lr=model_config.lr, eps= 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=model_config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

best_val_loss = float("inf")
for epoch in range(model_config.epochs):
    epoch_start_time = time.time()
    if model_config.parallel:
        dist.barrier()
        train_sampler_12.set_epoch(epoch)
        train_sampler_34.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)

    pretrain_generation(model, train_loader_12, criterion_gep_gepc, scaler, optimizer, scheduler, device, config, logger, epoch)
    pretrain_generation(model, train_loader_34, criterion_gep_gepc, scaler, optimizer, scheduler, device, config, logger, epoch)
    epoch_end_time = time.time()
    logger.info(f"Epoch {epoch} time: {epoch_end_time - epoch_start_time}")
    
    with torch.no_grad():
        val_loss = generation_evaluate(model, valid_loader, criterion_gep_gepc, device, config, logger, epoch)

    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%YY%mM%dD%HH%MM%SS")
    save_path = f'./model_param/{config.train_strategy}'
    os.makedirs(save_path, exist_ok=True)
    if model_config.parallel and local_rank != 0:
        pass 
    else:
        checkpoint_path = os.path.join(save_path, f"{timestamp}_{config.input_emb_style}_model_{epoch}.pth")
        torch.save(model.module.state_dict(), checkpoint_path)
        logger.info(f"Saving model at {timestamp}, saving path: {save_path}")
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        best_model_name = f"best_model_{config.input_emb_style}_{best_model_epoch}_{timestamp}.pth"
        logger.info(f"Best model with valid loss {best_val_loss:5.4f} at epoch {best_model_epoch}, Current time: {current_time}")

if model_config.parallel and local_rank != 0:
    pass 
else:
    best_model_path = f'{save_path}/best_model'
    os.makedirs(best_model_path, exist_ok=True)
    logger.info(f"Best model with valid loss {best_val_loss:5.4f}")
    torch.save(best_model.module.state_dict(), os.path.join(best_model_path, best_model_name))
    logger.info(f"Saving best model at {best_model_epoch}, saving path: {best_model_path}")

if model_config.parallel:
    torch.distributed.destroy_process_group()
wandb.finish()

end_time = time.time()
logger.info(f"Train time: {end_time - start_time}")
logger.info("Train finished!")

