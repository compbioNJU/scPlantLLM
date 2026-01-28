from collections import Counter
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import time
import traceback
import h5py
import numpy as np
from typing import Dict, Iterable, List, Optional, Tuple, Union
from loss import masked_relative_error
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import wandb
import warnings
import torch.distributed as dist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import random
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import scipy.cluster.hierarchy as sch
import umap
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime

def setup_custom_logger(logger_name, log_path):
    '''
    logger_name: str, name of the logger
    log_path: str, path to save the log file
    return: logger
    '''
    timestamp = datetime.now().strftime("%yY%mM%dD%HH%MM")
    log_file = os.path.join(log_path, f"{logger_name}_{timestamp}.log")
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    if not os.path.isfile(log_file):
        open(log_file, "w").close()
    fhlr = logging.FileHandler(log_file)
    fhlr.setFormatter(formatter)
    
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(fhlr)
    return logger


def set_seed(seed=1234):
    '''
    seed: int, random seed
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

def cell_type_metrics(predictions, celltypes_labels):
    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro",  zero_division=1)
    recall = recall_score(celltypes_labels, predictions, average="macro",  zero_division=1)
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")
    micro_f1 = f1_score(celltypes_labels, predictions, average="micro")
    return accuracy, precision, recall, macro_f1, micro_f1


def reduce_loss(single_loss, parallel=False, reduction="mean"):
    if parallel:
        if reduction == "sum":
            single_loss = single_loss.sum()
        elif reduction == "mean":
            single_loss = single_loss.mean()
        else:
            raise ValueError("Unsupported reduction type. Use 'sum' or 'mean'.")
            
    return single_loss


def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        values (array-like):
            A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of genes to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.
        pad_value (int): The value of padding in the values, will be kept unchanged.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()
    
    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()


def read_data_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as h5file:
        data = {key: h5file[key][:] for key in h5file.keys()}
    return data


def load_all_chunks(output_path, dtype, start_chunk, num_chunks, end_chunk=None, logger=None):

    combined_data = {}
    
    # Determine end_chunk if not specified
    if end_chunk is None:
        end_chunk = start_chunk + num_chunks - 1
    
    for i in range(start_chunk, end_chunk + 1):
        if logger is not None:
            logger.info(f"Loading {dtype} data from chunk {i}...")
        filename = os.path.join(output_path, f'{dtype}_chunk_{i}.h5')
        chunk_data = read_data_from_hdf5(filename)

        for key, value in chunk_data.items():
            if key not in combined_data:
                combined_data[key] = []
            combined_data[key].append(value)

    # Combine data from all chunks
    if logger is not None:
        logger.info(f"Combining {dtype} data from chunks {start_chunk} to {end_chunk}...")
    for key in combined_data:
        combined_data[key] = np.concatenate(combined_data[key], axis=0)
    if logger is not None:        
        logger.info(f"Combined {dtype} data from chunks {start_chunk} to {end_chunk}. Done!")
    return combined_data


class CustomDataset(Dataset):
    def __init__(self, file_path, data_type='train', start_chunk=1, num_chunks=None, end_chunk=None, vocab=None, append_cls=True, cls_token='<cls>', logger=None):
        
        if num_chunks is None and end_chunk is None:
            raise ValueError("At least one of num_chunks or end_chunk must be provided.")
        # if logger is None:
        #     raise ValueError("logger must be provided.")
        self.data, self.metadata = self._load_data(file_path, data_type, start_chunk, num_chunks, end_chunk, vocab, append_cls, cls_token, logger)

        self.cell_name_to_index = {name: idx for idx, name in enumerate(self.metadata['cell_names'])}
        self.index_to_cell_name = {idx: name for name, idx in self.cell_name_to_index.items()}
        
    def _load_data(self, file_path, data_type, start_chunk, num_chunks, end_chunk, vocab, append_cls, cls_token, logger):
#         with h5py.File(file_path, 'r') as h5file:
        h5file = load_all_chunks(file_path, data_type, start_chunk, num_chunks, end_chunk, logger)
        data =  {
                'expressions': np.array(h5file[f'ex']).astype(np.float32),
                'gene_ids': np.array(h5file[f'gid']).astype(np.float32),
                'cell_types': np.array(h5file[f'major_ctype']).astype(np.float32) if 'major_ctype' in h5file else None,
                'batch_effects': np.array(h5file[f'batch']).astype(np.float32),
            }
        if logger is not None:
            logger.info(f"Transformed {data_type} data from h5file to numpy array.")    
        metadata ={
            'cell_names' : np.array(h5file[f'cell_index']),
            # 'gene_names' : np.array(h5file[f'gname_{data_type}']),
            'cell_types' : np.array(h5file[f'major_ctype']).astype(np.float32) if 'major_ctype' in h5file else None,
            }
            
        if append_cls:
            if vocab is not None:
                cls_id = vocab[cls_token]
            else:
                cls_id = 185621 
            data['expressions'] = np.insert(data['expressions'], 0, 0, axis=1)        
            data['gene_ids'] = np.insert(data['gene_ids'], 0, cls_id, axis=1)
        return data, metadata
    
    def __len__(self):
        return len(self.data['expressions'])

    def __getitem__(self, index):
        data_item = {}
        for key, value in self.data.items():
            if value is not None: 
                data_item[key] = torch.tensor(value[index])
        cell_name = self.metadata['cell_names'][index]
        cell_index = self.cell_name_to_index[cell_name]
        return data_item , cell_index


def data_loader(file_path, data_type='train', gene_vocab =None, start_chunk=1, num_chunks=None, end_chunk=None, batch_size=64, num_workers=1, pin_memory=True,  shuffle=False, drop_last=True, logger=None, append_cls=True, parallel=False):
 
    if num_chunks is None and end_chunk is None:
        raise ValueError("At least one of num_chunks or end_chunk must be provided.")
        
    if data_type not in ['train', 'valid', 'test']:
        raise ValueError("Invalid data_type. It must be 'train', 'valid', or 'test'.")
        
    dataset = CustomDataset(file_path, data_type, start_chunk, num_chunks, end_chunk, vocab=gene_vocab, append_cls=append_cls, logger=logger)
    
    if logger:
        logger.info(f"Load {data_type} data from {file_path}. The number of samples is {len(dataset)}.")
    if not parallel:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory)
        sampler = None
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory)
    
    return sampler, dataloader, dataset.metadata


def pretrain_generation(
    model: nn.Module,
    loader: DataLoader,
    criterion_gep_gepc,
    scaler,
    optimizer,
    scheduler,
    device,
    config,
    epoch,
):
    """
    Pretrain the model on the training data.
    """
    model.train()
    total_loss = 0.0
    total_gep = 0.0
    total_err = 0.0
    total_acc = 0.0 
    total_num = 0
    log_interval = config.log_interval

    start_time = time.time()
    num_batches = len(loader)
    for batch, data in enumerate(loader):
        batch_data = data[0]
        input_gene_ids = batch_data["gene_ids"].to(device).long()
        target_values = batch_data["expressions"]#.to(device)
        if config.input_emb_style == "category":
            pad_value = config.n_bins
            mask_value = config.mask_value + 1
            target_values[target_values == config.pad_value] = pad_value
            n_bins = config.n_bins + 2
        else:
            pad_value = config.pad_value 
            mask_value = config.mask_value
            
        input_values = random_mask_value(target_values, mask_ratio=config.mask_ratio, mask_value=mask_value, pad_value=pad_value)
        
        masked_positions = input_values.eq(mask_value)#.to(device)
        target_values = target_values#.to(device)
        input_values = input_values.to(device)
        src_key_padding_mask = input_gene_ids.eq(config.pad_token_id) # pad position
        
        loss_to_log = {}
        with torch.cuda.amp.autocast(enabled=config.amp):

            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=False  # pretraining does not need CLS or CCE
            )
            
            loss = 0.0
            
            if config.input_emb_style == "else":#"category":
                # logger.info(output_values.shape)
                output_values = output_dict["mlm_output"].cpu()
                output_values = F.softmax(output_values, dim=-1)#.cpu() #B,L,Bins
                masked_predictions = torch.masked_select(output_values, masked_positions.unsqueeze(-1)).view(-1, n_bins)
                masked_target = torch.masked_select(target_values, masked_positions).to(torch.int64)
                masked_predicted_labels = torch.masked_select(torch.argmax(output_values, dim=-1), masked_positions).numpy()
                masked_target_labels = masked_target.numpy()
                
                accuracy, precision, recall, macro_f1, micro_f1 = cell_type_metrics(masked_predicted_labels, masked_target_labels)
                error_rate = 1 - (
                    (masked_predicted_labels == masked_target_labels)
                    .sum()
                    .item()
                    ) / masked_target_labels.shape[0]
                total_err += error_rate
                total_acc += accuracy
                # logger.info(f"masked_target_labels: {masked_target_labels.shape}, masked_predicted_labels: {masked_predicted_labels.shape}" )
                loss_gep = F.cross_entropy(masked_predictions, masked_target)
                loss_to_log.update({"pretrain/accuracy": accuracy, "pretrain/gep": loss_gep.item()})
            else:
                output_values = output_dict["mlm_output"].cpu()
                loss_gep = criterion_gep_gepc(
                    output_values, target_values, masked_positions
                )
                mre = masked_relative_error(
                    output_values, target_values, masked_positions
                )

                total_err += mre.item()
                loss_to_log = {"pretrain/gep": loss_gep.item(), "pretrain/mre": mre.item()}
                
            total_gep += loss_gep.item()
            loss = loss + loss_gep
            
        model.zero_grad()
        loss = scaler.scale(loss)
        loss.backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss
        total_num += target_values.shape[0]  
        
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval / scaler.get_scale()
            cur_gep = total_gep / log_interval if config.GEP else 0.0
            cur_err = total_err / log_interval
            if config.input_emb_style == "else":#"category":
                cur_acc = total_acc / log_interval
                
                print(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.5f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | Scale Factor: {scaler.get_scale()} | "
                    f"real loss:{cur_loss:5.2f} | "
                    f"curl gep: {cur_gep:5.2f} |"
                    f"train/accuracy: {cur_acc:5.2f}, train/error_rate: {cur_err:5.2f}"
                    )
                
                total_err = 0.0
                total_acc = 0.0 
            else:
                print(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.5f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | Scale Factor: {scaler.get_scale()} | "
                    f"real loss:{cur_loss:5.2f} | "
                    f"curl gep: {cur_gep:5.2f} | mre {cur_err:5.2f}"
                    )
            total_loss = 0.0
            total_err = 0.0
            total_gep = 0.0
            start_time = time.time()

        wandb.log(loss_to_log)


def generation_evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_gep_gepc,
    device,
    config,
    epoch,
) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.train()
    total_loss = 0.0
    total_gep = 0.0
    total_err = 0.0
    total_acc = 0.0 
    total_num = 0
    log_interval = config.log_interval

    start_time = time.time()
    num_batches = len(loader)
    for batch, data in enumerate(loader):
        batch_data = data[0]
        input_gene_ids = batch_data["gene_ids"].to(device).long()
        target_values = batch_data["expressions"]#.to(device)
        
        if config.input_emb_style == "category":
            pad_value = config.n_bins
            mask_value = config.mask_value + 1
            # cls=0 目前
            target_values[target_values == config.pad_value] = pad_value
            n_bins = config.n_bins + 2
        else:
            pad_value = config.pad_value 
            mask_value = config.mask_value
        
        # has_zero = (target_values == 0).any()
        # logger.info(f"target_values Has value 0:{ has_zero}, maskvaule:{mask_value}")
        input_values = random_mask_value(target_values, mask_ratio=config.mask_ratio, mask_value=mask_value, pad_value=pad_value)
        
        masked_positions = input_values.eq(mask_value)#.to(device)
        target_values = target_values#.to(device)
        input_values = input_values.to(device)
        src_key_padding_mask = input_gene_ids.eq(config.pad_token_id) # pad position
        
        # has_zero = ( target_values[src_key_padding_mask.cpu()] ==pad_value).any()
        # logger.info(f"target_values Has value :{ has_zero}")

        loss_to_log = {}
        with torch.cuda.amp.autocast(enabled=config.amp):

            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=False  # pretraining does not need CLS or CCE

            )
            
            loss = 0.0
            if config.input_emb_style == "else":#"category":
                output_values = output_dict["mlm_output"].cpu()
                output_values = F.softmax(output_values, dim=-1).cpu() #B,L,Bins
                masked_predictions = torch.masked_select(output_values, masked_positions.unsqueeze(-1)).view(-1, n_bins)
                masked_target = torch.masked_select(target_values, masked_positions).to(torch.int64)
                masked_predicted_labels = torch.masked_select(torch.argmax(output_values, dim=-1), masked_positions).numpy()
                masked_target_labels = masked_target.numpy()
                
                accuracy, precision, recall, macro_f1, micro_f1 = cell_type_metrics(masked_predicted_labels, masked_target_labels)
                
                error_rate = 1 - (
                    (masked_predicted_labels == masked_target_labels)
                    .sum()
                    .item()
                    ) / masked_target_labels.shape[0]
                total_err += error_rate
                total_acc += accuracy
                loss_gep = F.cross_entropy(masked_predictions, masked_target)
                loss_to_log.update({"pretrain/accuracy": accuracy, "pretrain/gep": loss_gep.item()})
            else:
                target_values=target_values.to(device)
                masked_positions = masked_positions.to(device)
                output_values = output_dict["mlm_output"]#.cpu()
                loss_gep = criterion_gep_gepc(
                    output_values, target_values, masked_positions
                )
                mre = masked_relative_error(
                    output_values, target_values, masked_positions
                )

                total_err += mre.item()
                loss = loss + loss_gep
                loss_to_log = {"pretrain/gep": loss_gep.item(), "pretrain/mre": mre.item()}
                
            total_gep += loss_gep.item()
            loss = loss + loss_gep
        
        total_loss += loss
        total_num += target_values.shape[0]
        
        if batch % log_interval == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_gep = total_gep / log_interval if config.GEP else 0.0
            cur_err = total_err / log_interval
            if config.input_emb_style =="else":# "category":
                cur_acc = total_acc / log_interval

                print(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f" ms/batch {ms_per_batch:5.2f} | "
                    f"curl gep: {cur_gep:5.2f} |"
                    f"valid/accuracy: {cur_acc:5.2f}, valid/error_rate: {cur_err:5.2f}"
                    )
                
                total_err = 0.0
                total_acc = 0.0 
            else:
                print(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"ms/batch {ms_per_batch:5.2f} | "
                    f"curl gep: {cur_gep:5.2f} | mre {cur_err:5.2f}"
                    )
            total_err = 0.0
            total_loss = 0.0
            total_gep = 0.0
            start_time = time.time()

        wandb.log(loss_to_log)

    return total_loss / total_num


def compute_umap_embeddings(embeddings):
    umap_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(embeddings)
    return umap_embeddings

class ConfigObject:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_config(config_file_path):
    """
    Load configuration from a JSON file.

    Args:
        config_file_path (str): Path to the JSON configuration file.

    Returns:
        ConfigObject: Object containing configuration.
    """
    def object_decoder(obj):
        return ConfigObject(**obj)

    with open(config_file_path, 'r') as f:
        config = json.load(f, object_hook=object_decoder)
    
    return config


def train(
    model: nn.Module,
    loader: DataLoader,
    criterion_gep_gepc,
    criterion_dab,
    criterion_cls,
    scaler,
    optimizer,
    scheduler,
    device,
    config,
    epoch,
    parallel=False,
) -> None:
    """
    Train the model for one epoch.
    """
    
    model.train()
    total_loss, total_gep, total_cls, total_gepc, total_ecs, total_dab = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    total_err = 0.0
    total_acc = 0.0 
    total_zero_log_prob, total_gepc_zero_log_prob = 0.0, 0.0
    # total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()
    num_batches = len(loader)
    for batch, data in enumerate(loader):
        batch_data = data[0]
        input_gene_ids = batch_data["gene_ids"].to(device).long()
        target_values = batch_data["expressions"]
        batch_labels = batch_data["batch_effects"].to(device).long()
        batch_labels = torch.squeeze(batch_labels)
        if config.input_emb_style == "category":
                pad_value = config.n_bins
                mask_value = config.mask_value + 1
                target_values[target_values == config.pad_value] = pad_value
                n_bins = config.n_bins + 2
        else:
                pad_value = config.pad_value
                mask_value = config.mask_value

        input_values = random_mask_value(target_values, mask_ratio=config.mask_ratio, mask_value=mask_value, pad_value=pad_value)

        if config.task == "annotation":
            celltype_labels = batch_data["cell_types"].to(device).long()
            celltype_labels =  torch.squeeze(celltype_labels)
            # celltype_labels -= 1
            input_values = target_values
        
        target_values = target_values.to(device)
        masked_positions = input_values.eq(mask_value).to(device)
        input_values = input_values.to(device)

        src_key_padding_mask = input_gene_ids.eq(config.pad_token_id)

        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels
                if config.use_batch_labels or config.DSBN
                else None,
                CLS=config.CLS
            )

            loss = 0.0
            metrics_to_log = {}
            if config.input_emb_style == "else": #category":
                output_values = output_dict["mlm_output"]
                output_values = F.softmax(output_values, dim=-1).cpu() #B,L,Bins
                masked_predictions = torch.masked_select(output_values, masked_positions.unsqueeze(-1)).view(-1, n_bins)
                masked_target = torch.masked_select(target_values, masked_positions).to(torch.int64) 
                masked_predicted_labels = torch.masked_select(torch.argmax(output_values, dim=-1), masked_positions).numpy()
                masked_target_labels = masked_target.numpy()
                
                accuracy, precision, recall, macro_f1, micro_f1 = cell_type_metrics(masked_predicted_labels, masked_target_labels)
                error_rate = 1 - (
                    (masked_predicted_labels == masked_target_labels)
                    .sum()
                    .item()
                    ) / masked_target_labels.shape[0]
                total_err += error_rate
                total_acc += accuracy
                loss_gep = F.cross_entropy(masked_predictions, masked_target)
                metrics_to_log.update({"train_gep/accuracy": accuracy, "train/gep": loss_gep.item()})
            else:
                if config.GEP:
                    loss_gep = criterion_gep_gepc(
                        output_dict["mlm_output"], target_values, masked_positions
                    )
                    loss = loss + loss_gep
                    metrics_to_log = {"train/gep": loss_gep.item()}
                if config.GEP and config.explicit_zero_prob:
                    loss_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_zero_log_prob
                    metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})

            if config.GEPC:
                loss_gepc = criterion_gep_gepc(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
            if config.GEPC and config.explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_gepc_zero_log_prob
                metrics_to_log.update(
                    {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                )

            if config.CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                output_value = output_dict["cls_output"].argmax(1) 
                cell_types_prediction = output_value.cpu().numpy()
                cell_types_label = list(celltype_labels.cpu().numpy())
                metrics_to_log.update({"train/cls": loss_cls.item()})  
                loss += loss_cls
                accuracy, precision, recall, macro_f1, micro_f1 = cell_type_metrics(cell_types_prediction, cell_types_label)
                error_rate = 1 - (
                    (output_value== celltype_labels)
                    .sum()
                    .item()
                    ) / celltype_labels.size(0)
                metrics_to_log.update({"train/accuracy": accuracy, "train/error_rate": error_rate}) 
                total_err += error_rate
                total_acc += accuracy
                if batch % log_interval == 0 and batch > 0:
                    print(
                        f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                        f"train/accuracy: {total_acc / log_interval}, train/error_rate: {total_err / log_interval}") 
                    total_err = 0.0
                    total_acc = 0.0 

            if config.ESC:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss_ecs = reduce_loss(loss_ecs, parallel=parallel)
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})

            if config.DAR:
                # try weighting and separate optimizer
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + config.dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})
        

        model.zero_grad()
        loss = scaler.scale(loss)
        loss.backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )

        scaler.step(optimizer)
        scaler.update()

        wandb.log(metrics_to_log)

        total_loss += loss.item()
        total_gep += loss_gep.item() if config.GEP else 0.0
        total_cls += loss_cls.item() if config.CLS else 0.0
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_ecs += loss_ecs.item() if config.ESC else 0.0
        total_dab += loss_dab.item() if config.DAR else 0.0
        total_zero_log_prob += (
            loss_zero_log_prob.item()
            if config.GEP and config.explicit_zero_prob
            else 0.0
        )
        total_gepc_zero_log_prob += (
            loss_gepc_zero_log_prob.item()
            if config.GEPC and config.explicit_zero_prob
            else 0.0
        )
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_gep = total_gep / log_interval if config.GEP else 0.0
            cur_cls = total_cls / log_interval if config.CLS else 0.0
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_ecs = total_ecs / log_interval if config.ESC else 0.0
            cur_dab = total_dab / log_interval if config.DAR else 0.0
            cur_zero_log_prob = (
                total_zero_log_prob / log_interval if config.explicit_zero_prob else 0.0
            )
            cur_gepc_zero_log_prob = (
                total_gepc_zero_log_prob / log_interval
                if config.GEPC and config.explicit_zero_prob
                else 0.0
            )
            # cur_error = total_error / log_interval

            print(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.5f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | scale factor: {scaler.get_scale()} |"
                + (f"scaled loss {cur_loss / scaler.get_scale():5.2f} |")
                + (f"gep {cur_gep:5.2f} |" if config.GEP else "")
                + (f"cls {cur_cls:5.2f} | " if config.CLS else "")
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                + (f"ecs {cur_ecs:5.2f} |" if config.ESC else "")
                + (f"dar {cur_dab:5.2f} |" if config.DAR else "")       
            )
            
            total_loss = 0
            total_gep = 0
            total_cls = 0
            total_gepc = 0
            total_ecs = 0
            total_dab = 0
            total_zero_log_prob = 0
            total_gepc_zero_log_prob = 0
            # total_error = 0
            start_time = time.time()
            
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_gep_gepc,
    criterion_dab,
    criterion_cls,
    device,
    config,
    epoch,
) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_gep = 0.0
    total_cls = 0.0
    total_dab = 0.0
    total_err = 0.0
    total_acc = 0.0
    total_num = 0
    cell_types_predictions = []
    cell_types_labels = []
    log_interval = 100
    with torch.no_grad():
        num_batches = len(loader)
        for batch, data in enumerate(loader):
            batch_data = data[0]
            input_gene_ids = batch_data["gene_ids"].to(device).long()
            target_values = batch_data["expressions"]#.to(device)
            batch_labels = batch_data["batch_effects"].to(device).long()
            batch_labels = torch.squeeze(batch_labels)
            if config.input_emb_style == "category":
                pad_value = config.n_bins
                mask_value = config.mask_value + 1
                target_values[target_values == config.pad_value] = pad_value
                n_bins = config.n_bins + 2
            else:
                pad_value = config.pad_value
                mask_value = config.mask_value
            
            input_values = random_mask_value(target_values, mask_ratio=config.mask_ratio, mask_value=mask_value, pad_value=pad_value)
            
            if config.task == "annotation":
                celltype_labels = batch_data["cell_types"].to(device).long()
                celltype_labels =  torch.squeeze(celltype_labels)
                # celltype_labels -= 1
                input_values = target_values
                
            target_values = target_values.to(device)
            masked_positions = input_values.eq(mask_value).to(device) 
            input_values = input_values.to(device)
            src_key_padding_mask = input_gene_ids.eq(config.pad_token_id)
            
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels
                    if config.use_batch_labels or config.DSBN
                    else None,
                    CLS=config.CLS # True or False
                    # MVC=config.GEPC, 
                    # ECS=config.ESC
                )
                
                loss = 0.0
                metrics_to_log = {}
                # when fine-tuning, meeds setting CLS or GEP to False
                if config.task == "annotation" or config.CLS:
                    loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                    loss = loss + loss_cls
                    output_value = output_dict["cls_output"].argmax(1) 
                    cell_types_prediction = output_value.cpu().numpy()
                    cell_types_label = list(celltype_labels.cpu().numpy())
                    cell_types_predictions.extend(cell_types_prediction)
                    cell_types_labels.extend(cell_types_label)
                    total_cls += loss_cls.item() 
                    
                    accuracy, precision, recall, macro_f1, micro_f1 = cell_type_metrics(cell_types_prediction, cell_types_label)
                    error_rate = 1 - (
                    (output_value== celltype_labels)
                    .sum()
                    .item()
                    ) / celltype_labels.size(0)
                    metrics_to_log.update({"valid/accuracy": accuracy, "valid/error_rate": error_rate}) 
                    total_err += error_rate
                    total_acc += accuracy
                    if batch % log_interval == 0 and batch > 0:
                        print(
                            f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                            f"valid/accuracy: {total_acc / log_interval}, valid/error_rate: {total_err / log_interval}") 
                        total_err = 0.0
                        total_acc = 0.0

                    else:
                        output_values = output_dict["mlm_output"]
                        loss_gep = criterion_gep_gepc(
                            output_values, target_values, masked_positions
                        )      
                        total_gep += loss_gep.item()
                        loss = loss + loss_gep
                        mre = masked_relative_error(output_values, target_values, masked_positions).item() * len(input_gene_ids)
                        total_err += mre
                        metrics_to_log.update({"evaluate/mre": mre, "evaluate/gep": loss_gep.item()})
                        if batch % log_interval == 0 and batch > 0:
                            print(
                                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                                f" valid/mre: {total_err / log_interval}") 
                            total_err = 0.0

                if config.DAR: #integration is true
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss * len(input_gene_ids)
            

            if config.DAR:
                total_dab += (
                    loss_dab.item() * len(input_gene_ids) if config.DAR else 0.0
                )
            else:
                total_dab = 0

            total_num += len(input_gene_ids)
            
            wandb.log(metrics_to_log)
    
    if config.task == "annotation":
        accuracy, precision, recall, macro_f1, micro_f1 = cell_type_metrics(cell_types_predictions, cell_types_labels)
        wandb.log(
            {
                "valid/loss": total_loss + config.dab_weight * total_dab / total_num,
                "valid/cls": total_cls / total_num,
                "valid/accuracy": accuracy,
                "valid/precision": precision,
                "valid/recall": recall,
                "valid/macro_f1": macro_f1,
                "valid/micro_f1": micro_f1,
            },
        )
        print(f"valid/loss: {total_loss + config.dab_weight * total_dab / total_num}, valid/cls: {total_cls / total_num}, valid/accuracy: {accuracy}, valid/precision: {precision}, valid/recall: {recall}, valid/macro_f1: {macro_f1}, valid/micro_f1: {micro_f1}")
        
    elif config.task in ["integration", "multiomic"]:
        wandb.log(
            {
                "valid/loss": total_loss + config.dab_weight * total_dab / total_num,
                "valid/gep": total_gep / total_num,
            },
        )
        print(f"valid/loss: {total_loss + config.dab_weight * total_dab / total_num}, valid/gep: {total_gep / total_num}")

    return total_loss / total_num


def test(
    model: nn.Module,
    loader: DataLoader,
    metadata,
    device,
    config,
    epoch=0,
):
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_err = 0.0
    total_acc = 0.0
    total_num = 0
    cell_types_predictions = []
    cell_types_labels = []
    cell_names = []
    probabilities = []
    cell_embeddings_list = []
    batch_labels_list = []
    log_interval = 100
    with torch.no_grad():
        num_batches = len(loader)
        for batch, data in enumerate(loader):
            batch_data = data[0]
            cell_index = data[1]
            input_gene_ids = batch_data["gene_ids"].to(device).long()
            target_values = batch_data["expressions"]#.to(device)
            batch_labels = batch_data["batch_effects"].to(device).long()
            batch_labels = torch.squeeze(batch_labels)
            batch_labels_list.extend(batch_labels.cpu().numpy())
            
            cell_name = metadata['cell_names'][cell_index]
            cell_name = [byte_string.decode('utf-8') for byte_string in cell_name]
            cell_names.extend(cell_name)
            
            if config.input_emb_style == "category":
                pad_value = config.n_bins
                mask_value = config.mask_value + 1
                target_values[target_values == config.pad_value] = pad_value
            else:
                pad_value = config.pad_value
                mask_value = config.mask_value
  
            if config.task == "annotation":
                celltype_labels = batch_data["cell_types"].to(device).long()
                celltype_labels =  torch.squeeze(celltype_labels)
                # celltype_labels -= 1
                input_values = target_values
                
            target_values = target_values#.to(device)
            input_values = input_values.to(device)
            src_key_padding_mask = input_gene_ids.eq(config.pad_token_id)
            
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels
                    if config.use_batch_labels or config.DSBN
                    else None,
                    CLS=config.CLS,  # evaluation does not need CLS or CCE
                    # MVC=False,
                    # ECS=False,
                )
            
            with torch.cuda.amp.autocast(enabled=config.amp):
                cell_embeddings = model.encode_batch(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_size=input_gene_ids.shape[0],
                batch_labels=torch.from_numpy(batch_labels).long() if config.DSBN else None,
                time_step=0,
                return_np=True,
            )
                cell_embeddings = cell_embeddings / np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
                cell_embeddings_list.append(cell_embeddings)
                
                loss = 0.0
                metrics_to_log = {}
                # when fine-tuning, meeds setting CLS or GEP to False
                if config.task == "annotation" or config.CLS:
                    output_value = output_dict["cls_output"] #torch.Size([B, n_ctype])
                    probability = F.softmax(output_value, dim=1)
                    probabilities.append(probability.detach().cpu().numpy()) 
                    output_value = output_value.argmax(1) 
                    cell_types_prediction = output_value.cpu().numpy()
                    cell_types_label = list(celltype_labels.cpu().numpy())
                    cell_types_predictions.extend(cell_types_prediction)
                    cell_types_labels.extend(cell_types_label)

                    accuracy, precision, recall, macro_f1, micro_f1 = cell_type_metrics(cell_types_prediction, cell_types_label)
                    error_rate = 1 - (
                    (output_value== celltype_labels)
                    .sum()
                    .item()
                    ) / celltype_labels.size(0)
                    metrics_to_log.update({"test/accuracy": accuracy, "test/error_rate": error_rate}) 
                    total_err += error_rate
                    total_acc += accuracy
                    if batch % log_interval == 0 and batch > 0:
                        print(
                            f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                            f"test/accuracy: {total_acc / log_interval}, test/error_rate: {total_err / log_interval}") 
                        total_err = 0.0
                        total_acc = 0.0


            total_num += len(input_gene_ids)
        
            wandb.log(metrics_to_log)
    cell_embeddings = np.concatenate(cell_embeddings_list, axis=0)
    if config.task == "annotation":
        accuracy, precision, recall, macro_f1, micro_f1 = cell_type_metrics(cell_types_predictions, cell_types_labels)
        wandb.log(
            {
                "test/accuracy": accuracy,
                "test/precision": precision,
                "test/recall": recall,
                "test/macro_f1": macro_f1,
                "test/micro_f1": micro_f1,
            },
        )
        print(f"test/accuracy: {accuracy}, test/precision: {precision}, test/recall: {recall}, test/macro_f1: {macro_f1}, test/micro_f1: {micro_f1}")
        probabilities = np.concatenate(probabilities, axis=0)
        
        return cell_types_predictions, cell_types_labels, cell_names, probabilities, cell_embeddings, batch_labels_list
    
def inference(
    model: nn.Module,
    loader: DataLoader,
    metadata,
    device,
    config,
    epoch=0,
):
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_err = 0.0
    total_acc = 0.0
    total_num = 0
    cell_types_predictions = []
    cell_types_labels = []
    cell_names = []
    probabilities = []
    cell_embeddings_list = []
    batch_labels_list = []
    log_interval = 100
    with torch.no_grad():
        num_batches = len(loader)
        for batch, data in enumerate(loader):
            batch_data = data[0]
            cell_index = data[1]
            input_gene_ids = batch_data["gene_ids"].to(device).long()
            target_values = batch_data["expressions"]#.to(device)
            batch_labels = batch_data["batch_effects"].to(device).long()
            batch_labels = torch.squeeze(batch_labels)
            batch_labels_list.extend(batch_labels.cpu().numpy())
            
            cell_name = metadata['cell_names'][cell_index]
            cell_name = [byte_string.decode('utf-8') for byte_string in cell_name]
            cell_names.extend(cell_name)
            
            if config.input_emb_style == "category":
                pad_value = config.n_bins
                mask_value = config.mask_value + 1
                target_values[target_values == config.pad_value] = pad_value
            else:
                pad_value = config.pad_value
                mask_value = config.mask_value
  
            if config.task == "annotation":
                input_values = target_values
                
            target_values = target_values#.to(device)
            input_values = input_values.to(device)
            src_key_padding_mask = input_gene_ids.eq(config.pad_token_id)
            
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels
                    if config.use_batch_labels or config.DSBN
                    else None,
                    CLS=config.CLS,  # evaluation does not need CLS or CCE
                    # MVC=False,
                    # ECS=False,
                )
            
            with torch.cuda.amp.autocast(enabled=config.amp):
                cell_embeddings = model.encode_batch(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_size=input_gene_ids.shape[0],
                batch_labels=torch.from_numpy(batch_labels).long() if config.DSBN else None,
                time_step=0,
                return_np=True,
            )
                cell_embeddings = cell_embeddings / np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
                cell_embeddings_list.append(cell_embeddings)
                
                loss = 0.0
                metrics_to_log = {}
                # when fine-tuning, meeds setting CLS or GEP to False
                if config.task == "annotation" or config.CLS:
                    output_value = output_dict["cls_output"] #torch.Size([B, n_ctype])
                    probability = F.softmax(output_value, dim=1)
                    probabilities.append(probability.detach().cpu().numpy()) 
                    output_value = output_value.argmax(1) 
                    cell_types_prediction = output_value.cpu().numpy()
                    cell_types_predictions.extend(cell_types_prediction)



            total_num += len(input_gene_ids)
        
            wandb.log(metrics_to_log)
    cell_embeddings = np.concatenate(cell_embeddings_list, axis=0)
    if config.task == "annotation":
        probabilities = np.concatenate(probabilities, axis=0)
        
        return cell_types_predictions, cell_types_labels, cell_names, probabilities, cell_embeddings, batch_labels_list
