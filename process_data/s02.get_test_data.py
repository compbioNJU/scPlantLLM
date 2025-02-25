from math import log
import pandas as pd
import h5py
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union
import time
import sys
import argparse
sys.path.insert(0, "../")
from Util.utils import setup_custom_logger, set_seed, load_config

seed = 1234
set_seed(seed)

def save_data_to_hdf5(file_path, **data):
    with h5py.File(file_path, 'w') as h5file:
        for key, value in data.items():
            h5file.create_dataset(key, data=value)
            
            
def split_and_save_data(output_path, stage, chunk_size, **data):
    total_size = len(next(iter(data.values())))
    num_chunks = (total_size // chunk_size) + (1 if total_size % chunk_size != 0 else 0)
    
    for i in range(num_chunks):
        chunk_data = {}
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_size)
        
        for key, value in data.items():
            chunk_data[key] = value[start_idx:end_idx]

        filename = os.path.join(output_path, f'{stage}_chunk_{i+1}.h5')
        save_data_to_hdf5(filename, **chunk_data)


def split_train_test_valid_save_hdf5(expression_file, gene_id_file, gene_name_file, cell_type_file, batch_effect_file, output_path, seed=1234, test_size=0.2, valid_size=0.5, logger=None):
    
    # Read data
    ex_df = pd.read_csv(expression_file, index_col=0)
    logger.info(f"loaded expression data!")
    gid_df = pd.read_csv(gene_id_file, index_col=0)
    logger.info(f"loaded gene id data!")
    gname_df = pd.read_csv(gene_name_file, index_col=0)
    logger.info(f"loaded gene name data!")
    ctype_df = pd.read_csv(cell_type_file, index_col=0)
    logger.info(f"loaded cell type data!")
    batch_df = pd.read_csv(batch_effect_file, index_col=0)
    logger.info(f"loaded batch effect data!")
    logger.info(f"loaded  all data!")
    
    ctype_df = ctype_df[['cell_label']]
    batch_df = batch_df[['batch_id']]
    
    logger.info(f"Expression shape: {ex_df.shape}")
    ex_train, ex_remain, gid_train, gid_remain, gname_train, gname_remain, ctype_train, ctype_remain, batch_train, batch_remain = train_test_split(
        ex_df, gid_df, gname_df, ctype_df, batch_df, test_size=test_size, stratify=ctype_df['cell_label'], shuffle=True, random_state=seed)
    logger.info(f"Train shape: {ex_train.shape}")
    
    ex_test, ex_valid, gid_test, gid_valid, gname_test, gname_valid, ctype_test, ctype_valid, batch_test, batch_valid = train_test_split(
        ex_remain, gid_remain, gname_remain, ctype_remain, batch_remain, test_size=valid_size, stratify=ctype_remain['cell_label'], shuffle=True, random_state=seed)
    logger.info(f"Test shape: {ex_test.shape}")
    
    ex_test.to_csv(os.path.join(output_path, 'test.csv'), index=True)
    ex_valid.to_csv(os.path.join(output_path, 'valid.csv'), index=True)
    ex_train.to_csv(os.path.join(output_path, 'train.csv'), index=True)
    logger.info(f"Saved test.csv, valid.csv, train.csv!")
    
    ctype_test.to_csv(os.path.join(output_path, 'ctype_test.csv'), index=True)
    ctype_valid.to_csv(os.path.join(output_path, 'ctype_valid.csv'), index=True)
    ctype_train.to_csv(os.path.join(output_path, 'ctype_train.csv'), index=True)
    logger.info(f"Saved ctype_test.csv, ctype_valid.csv, ctype_train.csv!")

    try:
        split_and_save_data(output_path, stage='test', chunk_size=500000, ex_test=ex_test.values, gid_test=gid_test.values, gname_test=gname_test.values, ctype_test=ctype_test.values, batch_test=batch_test.   values, cell_index_test=ex_test.index.values)
        logger.info(f"Saved test.h5 in chunks!")
        split_and_save_data(output_path, stage='valid', chunk_size=500000, ex_valid=ex_valid.values, gid_valid=gid_valid.values, gname_valid=gname_valid.values, ctype_valid=ctype_valid.values, batch_valid=batch_valid.values, cell_index_valid=ex_valid.index.values)
        logger.info(f"Saved valid.h5 in chunks!")
#         ex_test, ex_valid, gid_test, gid_valid, gname_test, gname_valid = _ , _, _, _, _, _
#         save_data_to_hdf5(os.path.join(output_path, 'train.h5'), ex_train=ex_train.values, gid_train=gid_train.values, gname_train=gname_train.values, ctype_train=ctype_train.values, batch_train=batch_train.values, cell_index_train=ex_train.index.values)
        split_and_save_data(output_path, stage='train', chunk_size=500000, ex_train=ex_train.values, gid_train=gid_train.values, gname_train=gname_train.values, ctype_train=ctype_train.values, batch_train=batch_train.values, cell_index_train=ex_train.index.values)
        logger.info("Saved train.h5 in chunks!")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    config = load_config('../Util/config.json')
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_path', type=str, help='Path to the data folder')
    parser.add_argument("--output_path", type=str, help="Path to save the output files")
    parser.add_argument('--log_name', type=str, help='log name')
    
    args = parser.parse_args()
    home_path = args.home_path
    log_name = args.log_name
    output_path = args.output_path
    
    logger = setup_custom_logger(f'L02.{log_name}', log_path='./log/')
    logger.info("Start to split train, test and valid data.")
    # logger.info("Home path: %s" % home_path)
    # home_path = '/mnt/public6/caoguangshuo/scPlantGPT/s03.scPlantGPT/data/processed_data_last'

    expression_file = os.path.join(home_path, 'gene_expression_matrix.csv')
    gene_id_file = os.path.join(home_path, 'gene_id_df.csv')
    gene_name_file = os.path.join(home_path, 'gene_names_df.csv')
    cell_type_file = os.path.join(home_path, 'cell_type_df.csv')
    batch_effect_file = os.path.join(home_path, 'batch_effect_df.csv')
    
    # output_path = os.path.join(home_path)
    logger.info(f"Out path: {output_path}")
    split_train_test_valid_save_hdf5(expression_file, gene_id_file, gene_name_file, cell_type_file, batch_effect_file, output_path, seed=config.seed, test_size=0.1, valid_size=0.5, logger=logger)
    
    logger.info("Split train, test and valid data finished. Saved to %s." % output_path)
    
    end_time = time.time()
    logger.info("Time used: %.2f seconds." % (end_time - start_time))
    logger.info("All done!")