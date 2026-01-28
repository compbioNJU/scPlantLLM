from ast import parse
import h5py
import pandas as pd
import numpy as np
import json
import argparse
import os
import random
from sklearn.model_selection import train_test_split
import time
import sys


# sys.path.insert(0, "../")
from utils import setup_custom_logger, set_seed, load_config

set_seed(1234)
pd.set_option('future.no_silent_downcasting', True)


def save_data_to_hdf5(file_path, compression="gzip", compression_level=4, **data):
    with h5py.File(file_path, 'w') as h5file:
        for key, value in data.items():
            try:
                h5file.create_dataset(key, 
                                      data=value, 
                                      compression=compression,        
                                      compression_opts=compression_level 
                )
            except TypeError as e:
                # 如果发生 TypeError，检查是否因为 object 类型
                if value.dtype == 'O':
                    raise TypeError(f"Column '{key}' has object type, which is not supported by HDF5.") from e
                else:
                    raise  # 如果是其他错误，重新抛出
            
            
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


def split_train_test_valid_save_hdf5_has_celltype(ex_df, gid_df, batch_df, ctype_df=None, output_path='./', seed=1234, test_size=0.1, valid_size=0.5, logger=None):
    
    if test_size > 0:
        major_ctype_df = ctype_df[['celltype_label']]

        logger.info(f"Expression shape: {ex_df.shape}")
        ex_train, ex_remain, gid_train, gid_remain, major_ctype_train, major_ctype_remain, batch_train, batch_remain = train_test_split(
            ex_df, gid_df, major_ctype_df, batch_df, test_size=test_size, stratify=major_ctype_df['celltype_label'], shuffle=True, random_state=seed)
        logger.info(f"Train shape: {ex_train.shape}")

        remaining_counts = major_ctype_remain['celltype_label'].value_counts()

        for label, count in remaining_counts.items():
            if count == 1:
                logger.info(f"Found class {label} with only 1 sample in remaining data")
                train_class_indices = major_ctype_train[major_ctype_train['celltype_label'] == label].index
                if len(train_class_indices) > 0:
                    index_to_move = train_class_indices[0]
                    ex_remain = pd.concat([ex_remain, ex_train.loc[[index_to_move]]])
                    ex_train = ex_train.drop(index_to_move)

                    gid_remain = pd.concat([gid_remain, gid_train.loc[[index_to_move]]])
                    gid_train = gid_train.drop(index_to_move)

                    major_ctype_remain = pd.concat([major_ctype_remain, major_ctype_train.loc[[index_to_move]]])
                    major_ctype_train = major_ctype_train.drop(index_to_move)

                    logger.info(f"Moved sample from train to remain for class {label}")

        ex_test, ex_valid, gid_test, gid_valid, major_ctype_test, major_ctype_valid, batch_test, batch_valid = train_test_split(ex_remain, gid_remain, major_ctype_remain,  batch_remain, test_size=valid_size, stratify=major_ctype_remain['celltype_label'], shuffle=True, random_state=seed)
        logger.info(f"Test shape: {ex_test.shape}")

        try:
            split_and_save_data(output_path, stage='test', chunk_size=500000, ex=ex_test.values, gid=gid_test.values, batch=batch_test.values ,major_ctype=major_ctype_test.values,cell_index=ex_test.index.values)
            logger.info(f"Saved test.h5 in chunks!")
            split_and_save_data(output_path, stage='valid', chunk_size=500000, ex=ex_valid.values, gid=gid_valid.values, batch=batch_valid.values,major_ctype=major_ctype_valid.values, cell_index=ex_valid.index.values)
            logger.info(f"Saved valid.h5 in chunks!")

            split_and_save_data(output_path, stage='train', chunk_size=500000, ex=ex_train.values, gid=gid_train.values, batch=batch_train.values, major_ctype=major_ctype_train.values,  cell_index=ex_train.index.values)
            logger.info("Saved train.h5 in chunks!")
        except Exception as e:
            logger.error(f"Error: {e}")
    else:
        try:
            split_and_save_data(output_path, stage='test', chunk_size=500000, ex=ex_df.values, gid=gid_df.values, batch=batch_df.values,major_ctype=ctype_df.values, cell_index=ex_df.index.values)
            logger.info("Saved test.h5 in chunks!")
        except Exception as e:
            logger.error(f"Error: {e}")
            
def split_train_test_valid_save_hdf5(ex_df, gid_df, batch_effect_df, output_path='./', seed=1234, test_size=0.1, valid_size=0.5, logger=None):
    if test_size > 0:
        logger.info(f"Expression shape: {ex_df.shape}")
        ex_train, ex_remain, gid_train, gid_remain, batch_train, batch_remain = train_test_split(ex_df, gid_df, batch_effect_df,test_size=test_size, shuffle=True, random_state=seed)
        logger.info(f"Train shape: {ex_train.shape}")

        ex_test, ex_valid, gid_test, gid_valid, batch_test, batch_valid = train_test_split(ex_remain, gid_remain, batch_remain, test_size=valid_size, shuffle=True, random_state=seed)
        logger.info(f"Test shape: {ex_test.shape}")

        try:
            split_and_save_data(output_path, stage='test', chunk_size=500000, ex=ex_test.values, gid=gid_test.values, batch=batch_test.values, cell_index=ex_test.index.values)
            logger.info(f"Saved test.h5 in chunks!")
            split_and_save_data(output_path, stage='valid', chunk_size=500000, ex=ex_valid.values, gid=gid_valid.values, batch=batch_valid.values, cell_index=ex_valid.index.values)
            logger.info(f"Saved valid.h5 in chunks!")

            split_and_save_data(output_path, stage='train', chunk_size=500000, ex=ex_train.values, gid=gid_train.values, batch=batch_train.values, cell_index=ex_train.index.values)
            logger.info("Saved train.h5 in chunks!")
        except Exception as e:
            logger.error(f"Error: {e}")
    else:
        try:
            split_and_save_data(output_path, stage='test', chunk_size=500000, ex=ex_df.values, gid=gid_df.values,batch=batch_effect_df.values, cell_index=ex_df.index.values)
            logger.info("Saved train.h5 in chunks!")
        except Exception as e:
            logger.error(f"Error: {e}")
            

def ensure_directory_exists(directory, logger):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Directory {directory} created.")
    else:
        logger.info(f"Directory {directory} already exists.")
        

def binning_dataframe(df, bins):
    min_value = df[df > 0].min().min()
    max_value = df.max().max()

    bin_edges = np.linspace(min_value, max_value, bins)

    def map_to_bins(x):
        if x == 0:
            return 0
        else:
            return np.digitize(x, bin_edges)

    binned_df = df.map(map_to_bins)
    
    return binned_df


def process_gene_expression(gene_counts_file_path, cell_type_file, gene_vocab_file, celltype_vocab_file, batch_effect_file, batch_effect_vocab_file, pad_token_id=0, pad_value=-2.0, seq_length=1200, binned=False, n_bins=100, include_zero_gene = False, out_path='./', has_celltype=True, logger=None, test_size=0.0):
    '''
    gene_counts_file_path: str, path to the gene expression matrix
    cell_type_file: str, path to the cell type file
    batch_effect_file: str, path to the batch effect file
    gene_vocab_file: str, path to the gene vocabulary file
    celltype_vocab_file: str, path to the cell type vocabulary file
    batch_effect_vocab_file: str, path to the batch effect vocabulary file
    pad: str, padding token
    seq_length: int, sequence length
    out_path: str, path to save the processed data
    
    This function processes the gene expression matrix and saves the processed data.
    '''
    
    start_time = time.time()  
    
    logger.debug("Start processing gene expression data...")

    gene_expression_list = []
    cell_type_list = []
    gene_id_list = []
    idx_list = []
    batch_effect_list = []

    to_remove = [
    "Unknow", "S phase", "G2/M phase", "G1/G0 phase", "G1/S phase", "G1 phase",
    "Contaminating nuclei", "Contamination", "Stress response", "Transitory",
    "Outer cell layer", "Inner cell layer", "Middle cell layer", "Vascular tissue"
    ]
    
    with open(batch_effect_vocab_file) as f:
        batch_effect_vocab = json.load(f)
    logger.info(f"max batch effect label: {max(batch_effect_vocab.values())}, min batch effect label: {min(batch_effect_vocab.values())}")
    batch_effect_df_origin = pd.read_csv(batch_effect_file).set_index('cell')
    
    if has_celltype:
        cell_type_df_origin = pd.read_csv(cell_type_file).set_index('cell')
        cell_type_df_origin = cell_type_df_origin[cell_type_df_origin['celltype'] != 'Unknow']
        cell_type_df_origin = cell_type_df_origin[~cell_type_df_origin['celltype'].isin(to_remove)]
    
    with open(gene_vocab_file) as f:
        gene_vocab = json.load(f)
    logger.info(f"max gene label: {max(gene_vocab.values())}, min gene label: {min(gene_vocab.values())}")
    
    if has_celltype:
        with open(celltype_vocab_file) as f:
            celltype_vocab = json.load(f)
        logger.info(f"max cell type label: {max(celltype_vocab.values())}, min cell type label: {min(celltype_vocab.values())}")

    for csv_file in os.listdir(gene_counts_file_path):
        if csv_file.endswith(".h5"):
            with h5py.File(os.path.join(gene_counts_file_path, csv_file), 'r') as f:
                    data = f['count/data'][:]  
                    cell_names = f['count/cell_names'][:]  
                    gene_names = f['count/gene_names'][:] 

                    cell_names = [cell.decode('utf-8') for cell in cell_names]
                    gene_names = [gene.decode('utf-8') for gene in gene_names]

            df = pd.DataFrame(data, index=cell_names, columns=gene_names)
            
            logger.info(f"start processing sample {csv_file}...{df.shape}")
            if binned:
                df = binning_dataframe(df, n_bins)
                logger.info(f"after binning, df max: {df.max().max()}, df min: {df.min().min()}")
                
            if has_celltype:
                df = df.loc[df.index.isin(cell_type_df_origin.index)]
                df['cell_type'] = cell_type_df_origin.loc[df.index, 'celltype']
            df['batch_effect'] = batch_effect_df_origin.loc[df.index, 'orig.ident']
            logger.info(f"The number of cell in file: {df.shape[0]}")
            
            flag = 1
            for idx, row in df.iterrows():
                flag += 1
                if flag % 1000 == 0:
                    logger.debug(f"Processing cell: {flag}")
                if not include_zero_gene:
                    if has_celltype:
                        non_zero_genes = row.drop(['cell_type', 'batch_effect']).replace(0, np.nan).dropna().index.tolist()
                    else:
                        non_zero_genes = row.drop(['batch_effect']).replace(0, np.nan).dropna().index.tolist()
                    if len(non_zero_genes) > seq_length:
                        proportions = [0.75]
                        for i, proportion in enumerate(proportions):
                            sampled_genes = random.sample(non_zero_genes, int(seq_length * proportion))
                            gene_expression = np.full(seq_length, pad_value)
                            gene_expression[:len(sampled_genes)] = row[sampled_genes].values
                            gene_expression_list.append(gene_expression)

                            gene_ids = [gene_vocab.get(gene_name, pad_token_id) for gene_name in sampled_genes]
                            gene_id_array = np.full(seq_length, pad_token_id)
                            gene_id_array[:len(gene_ids)] = gene_ids
                            gene_id_list.append(gene_id_array)

                            if has_celltype:
                                cell_type_list.append(row['cell_type'])
                            batch_effect_list.append(row['batch_effect'])
                            idx_list.append(idx)
                    else:
                        gene_expression = np.full(seq_length, pad_value)
                        gene_expression[:len(non_zero_genes)] = row[non_zero_genes].values
                        gene_expression_list.append(gene_expression)

                        gene_ids = [gene_vocab.get(gene_name, pad_token_id) for gene_name in non_zero_genes]
                        gene_id_array = np.full(seq_length, pad_token_id)
                        gene_id_array[:len(gene_ids)] = gene_ids
                        gene_id_list.append(gene_id_array)

                        if has_celltype:
                            cell_type_list.append(row['cell_type'])
                        batch_effect_list.append(row['batch_effect'])
                        idx_list.append(idx)
                else:
                    if(len(row) > seq_length):
                        logger.info(f"Sample {idx} has more than {seq_length} genes!")
                    gene_expression = np.full(seq_length, pad_value)
                    if has_celltype:
                        gene_expression[:len(row)] = row.drop(['cell_type', 'batch_effect']).values
                    else:
                        gene_expression[:len(row)] = row.drop(['batch_effect']).values
                    gene_expression_list.append(gene_expression)
                    if has_celltype:
                        gene_ids = [gene_vocab.get(gene_name, pad_token_id) for gene_name in row.drop(['cell_type']).index]
                    else:
                        gene_ids = [gene_vocab.get(gene_name, pad_token_id) for gene_name in row.index]
                    
                    gene_id_array = np.full(seq_length, pad_token_id)
                    gene_id_array[:len(gene_ids)] = gene_ids
                    gene_id_list.append(gene_id_array)

                    if has_celltype:
                        cell_type_list.append(row['cell_type'])
                    batch_effect_list.append(row['batch_effect'])

                    idx_list.append(idx)
                    
            logger.info(f"processing sample {csv_file} done!")
    
    logger.info("Start converting data to DataFrame...")
    
    gene_expression_df = pd.DataFrame(gene_expression_list, index=idx_list)
    
    batch_effect_df = pd.DataFrame({'batch_effect': batch_effect_list}, index=idx_list)
    batch_effect_df["batch"] = batch_effect_df["batch_effect"].map(batch_effect_vocab)
    batch_effect_df = batch_effect_df[['batch']]
    
    if has_celltype:
        cell_type_df = pd.DataFrame({'cell_type': cell_type_list}, index=idx_list)
    
    gene_id_df = pd.DataFrame(gene_id_list, index=idx_list)
    
    if has_celltype:
        cell_type_df["celltype_label"] = cell_type_df["cell_type"].map(celltype_vocab)
        split_train_test_valid_save_hdf5_has_celltype(gene_expression_df, gene_id_df, batch_effect_df, cell_type_df, output_path=out_path, seed=config.seed, test_size=test_size, valid_size=0.5, logger=logger)
    else:
        split_train_test_valid_save_hdf5(gene_expression_df, gene_id_df, batch_effect_df, output_path=out_path, seed=config.seed, test_size=test_size, valid_size=0.5, logger=logger)
    
    end_time = time.time()  
    total_time = end_time - start_time  
    logger.info(f"Total processing time: {total_time} seconds \n")


if __name__ == '__main__':

    config=load_config('setting.json')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to the data folder')
    parser.add_argument('--output_path', type=str, help="Path to the output")
    # parser.add_argument('--binned',  action='store_true', help="whether the data is binned or not")
    parser.add_argument('--has_celltype', action='store_true', help="whether the data has cell type or not")
    parser.add_argument('--test_size', type=float, default=0.0, help='Total ratio for both test and validation sets (will be split 50%%/50%%). Example: 0.2 means 10%% test + 10%% valid. Default: 0.0 (no split)')
    parser.add_argument('--cell_type_file', type=str, default=None, help='cell type file')
    parser.add_argument('--cell_type_vocab_file', type=str, default=None, help='cell type vocab path')
    parser.add_argument('--gene_vocab_file', type=str, help='gene vocab path')
    parser.add_argument('--batch_effect_file', type=str, help='batch effect file')
    parser.add_argument('--batch_effect_vocab_file', type=str, help='batch effect vocab path')

    args = parser.parse_args()
    gene_counts_file_path = args.input_path
    output_dir = args.output_path
    binned = True #args.binned
    has_celltype = args.has_celltype
    test_size = args.test_size
    cell_type_file = args.cell_type_file
    celltype_vocab_path = args.cell_type_vocab_file
    gene_vocab_path = args.gene_vocab_file
    batch_effect_file = args.batch_effect_file
    batch_effect_vocab_path = args.batch_effect_vocab_file
    
    os.makedirs('./log/', exist_ok=True)
    logger = setup_custom_logger(f'prepare_data', log_path='./log/')
    
    logger.info(f"config: {config}")
    n_bins=-1
    if binned:
        n_bins = config.n_bins - 1
   
    ensure_directory_exists(output_dir, logger)
    
    process_gene_expression(gene_counts_file_path, cell_type_file,  gene_vocab_file=gene_vocab_path, celltype_vocab_file=celltype_vocab_path, batch_effect_file=batch_effect_file, batch_effect_vocab_file=batch_effect_vocab_path, pad_token_id=config.pad_token_id, seq_length=config.seq_length, include_zero_gene=config.include_zero_gene, binned=binned, n_bins=n_bins,pad_value=config.pad_value, out_path=output_dir, has_celltype=has_celltype ,logger=logger, test_size = test_size)



