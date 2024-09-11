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
from moudules import data_loader, train, evaluate
import datetime
import wandb
sys.path.insert(0, "../")
from utils import set_seed, setup_custom_logger, load_config, log_fig
from scplantgpt.model import TransformerModel
from torch import distributed

start_time = time.time()
config = load_config('../Util/config.json')
seed = config.seed
set_seed(seed)

hyperparameter_defaults = dict(
    parallel=True,
    epochs=20, #pretrain 30
    batch_size=64,
    lr=1e-4,
    ntoken= 185622, #Ara 45416, cross 185622
    nctype= 44, #54,
    nbatch_effect= 238,#156,
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
    project="scPlantGPT_clean_label",
    entity="aibio",
    group=f"{config.train_strategy}_{config.input_emb_style}_{timestamp}",
)
model_config = wandb.config

if model_config.parallel:   
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)

    import torch.distributed as dist

    world_size = torch.cuda.device_count()
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    if local_rank == 0:
        logger = setup_custom_logger(f"L02.{config.train_strategy}_{config.input_emb_style}_{timestamp}", './../Log/')
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
    use_batch_labels=config.use_batch_labels, 
    num_batch_labels=model_config.nbatch_effect, 
    input_emb_style=config.input_emb_style, 
    n_input_bins=n_input_bins, 
    cell_emb_style="cls", 
    use_fast_transformer=model_config.fast_transformer, 
    pre_norm=model_config.pre_norm,)

model.to(device)

load_model = True
freeze = False
if load_model:
    model_name = f"{project_path}/s03_scPlantGPT/trainer/model_param/pretrain_clean_label_nlayer_6_mask0.15/best_model/best_model_category_12_2024Y07M19D23H45M09S.pth"
    logger.info(f"Loading model from {model_name}")
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
        logger.info(f"Because of the model structure change, only load params that are in the model and match the size!")
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

data_path = f'{project_path}/s03_scPlantGPT/cross_data/independent_clean_label/Rice_train'

logger.info(f"loading data from {data_path}")
train_sampler_12, train_loader_12, _ = data_loader(data_path, data_type='train', start_chunk=1, end_chunk=1, batch_size=model_config.batch_size, logger=logger, append_cls=True, parallel=model_config.parallel)

valid_sampler, valid_loader, _ = data_loader(data_path, data_type='valid', start_chunk=1, num_chunks=1, batch_size=model_config.batch_size,logger=logger, append_cls=True, parallel=model_config.parallel)
test_sampler, test_loader, _ = data_loader(data_path,  data_type='test',start_chunk=1, num_chunks=1, batch_size=model_config.batch_size,logger=logger, append_cls=True, parallel=model_config.parallel)

criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=model_config.lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=model_config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

best_val_loss = float("inf")
logger.info(f"Begin to train model...")
for epoch in range(model_config.epochs):
    epoch_start_time = time.time()
    if model_config.parallel:
        dist.barrier()
        if train_sampler_12 is not None:  # Check if train_sampler is not None
            train_sampler_12.set_epoch(epoch)
        if valid_sampler is not None:  # Check if valid_sampler is not None
            valid_sampler.set_epoch(epoch)
        if test_sampler is not None:  # Check if test_sampler is not None
            test_sampler.set_epoch(epoch)

    train(model, train_loader_12, criterion_cls, scaler, optimizer, scheduler, device, config, logger, epoch, model_config.parallel)
    epoch_end_time = time.time()
    logger.info(f"Epoch {epoch} time: {epoch_end_time - epoch_start_time}")
    
    val_loss = evaluate(model, test_loader, criterion_cls, device, config, logger, epoch)

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
    distributed.destroy_process_group()
wandb.finish()

end_time = time.time()
logger.info(f"Train time: {end_time - start_time}")
logger.info("Train finished!")