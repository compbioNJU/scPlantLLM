from matplotlib.style import use
import pandas as pd
import numpy as np
import json
#import torch
import argparse
from typing import Union
import os
import random
from concurrent.futures import ThreadPoolExecutor
import time
import sys

sys.path.insert(0, "../")
from moudules import  load_config

#set_seed(1234)
pd.set_option('future.no_silent_downcasting', True)

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


def process_gene_expression(gene_counts_file_path, cell_type_file, batch_effect_file, gene_vocab_file, celltype_vocab_file, batch_effect_vocab_file, pad_token="<PAD>", pad_token_id=0, pad_value=-2.0, seq_length=1200, binned=False, n_bins=100, include_zero_gene = False, out_path='./', logger=None):
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
    
    start_time = time.time()  # 记录开始时间
    
    logger.debug("Start processing gene expression data...")

    gene_expression_list = []
    gene_names_list = []
    cell_type_list = []
    batch_effect_list = []
    gene_id_list = []
    idx_list = []
    
    cell_type_df_origin = pd.read_csv(cell_type_file).set_index('cell')
    
    ## 去除unkonw细胞类型
    to_remove = [
    "Unknow", "S phase", "G2/M phase", "G1/G0 phase", "G1/S phase", "G1 phase",
    "Contaminating nuclei", "Contamination", "Stress response", "Transitory",
    "Outer cell layer", "Inner cell layer", "Middle cell layer", "Vascular tissue"
    ]
    
    cell_type_df_origin = cell_type_df_origin[cell_type_df_origin['celltype'] != 'Unknow']
    cell_type_df_origin = cell_type_df_origin[~cell_type_df_origin['celltype'].isin(to_remove)]

    batch_effect_df_origin = pd.read_csv(batch_effect_file).set_index('cell')
    
    with open(gene_vocab_file) as f:
        gene_vocab = json.load(f)
    logger.info(f"max gene label: {max(gene_vocab.values())}, min gene label: {min(gene_vocab.values())}")
    
    with open(celltype_vocab_file) as f:
        celltype_vocab = json.load(f)
    logger.info(f"max cell type label: {max(celltype_vocab.values())}, min cell type label: {min(celltype_vocab.values())}")

    with open(batch_effect_vocab_file) as f:
        batch_effect_vocab = json.load(f)
    logger.info(f"max batch effect label: {max(batch_effect_vocab.values())}, min batch effect label: {min(batch_effect_vocab.values())}")
    
    for csv_file in os.listdir(gene_counts_file_path):
        if csv_file.endswith(".csv"):
            df = pd.read_csv(os.path.join(gene_counts_file_path, csv_file), index_col=0)
            
            logger.info(f"start processing sample {csv_file}...{df.shape}")
            if binned:
                df = binning_dataframe(df, n_bins)
                logger.info(f"after binning, df max: {df.max().max()}, df min: {df.min().min()}")
                
            df = df.loc[df.index.isin(cell_type_df_origin.index)]
            # df = df[df.index != "Unknow"]
            df['cell_type'] = cell_type_df_origin.loc[df.index, 'celltype']
            df['batch_effect'] = batch_effect_df_origin.loc[df.index, 'orig.ident']
            
            logger.info(f"The number of cell in celltype.meta: {df.shape[0]}")
            
            flag = 1
            for idx, row in df.iterrows():
                flag += 1
                if flag % 1000 == 0:
                    logger.debug(f"Processing cell: {flag}")
                if not include_zero_gene:
                    non_zero_genes = row.drop(['cell_type', 'batch_effect']).replace(0, np.nan).dropna().index.tolist()
                    if len(non_zero_genes) > seq_length:
                        proportions = [0.25, 0.75, 0.5]
                        for i, proportion in enumerate(proportions):
                            sampled_genes = random.sample(non_zero_genes, int(seq_length * proportion))
                            gene_expression = np.full(seq_length, pad_value)
                            gene_expression[:len(sampled_genes)] = row[sampled_genes].values
                            gene_expression_list.append(gene_expression)

                            gene_names_list.append(sampled_genes + [pad_token] * (seq_length - len(sampled_genes)))

                            gene_ids = [gene_vocab.get(gene_name, pad_token_id) for gene_name in sampled_genes]
                            gene_id_array = np.full(seq_length, pad_token_id)
                            gene_id_array[:len(gene_ids)] = gene_ids
                            gene_id_list.append(gene_id_array)

                            cell_type_list.append(row['cell_type'])
                            batch_effect_list.append(row['batch_effect'])
                            idx_list.append(idx + f'_{i}')
                    else:
                        gene_expression = np.full(seq_length, pad_value)
                        gene_expression[:len(non_zero_genes)] = row[non_zero_genes].values
                        gene_expression_list.append(gene_expression)
                        gene_names_list.append(non_zero_genes + [pad_token] * (seq_length - len(non_zero_genes)))

                        gene_ids = [gene_vocab.get(gene_name, pad_token_id) for gene_name in non_zero_genes]
                        gene_id_array = np.full(seq_length, pad_token_id)
                        gene_id_array[:len(gene_ids)] = gene_ids
                        gene_id_list.append(gene_id_array)

                        cell_type_list.append(row['cell_type'])
                        batch_effect_list.append(row['batch_effect'])
                        idx_list.append(idx)
                else:
                    if(len(row) > seq_length):
                        logger.info(f"Sample {idx} has more than {seq_length} genes!")
                    gene_expression = np.full(seq_length, pad_value)
                    gene_expression[:len(row)] = row.drop(['cell_type', 'batch_effect']).values
                    gene_expression_list.append(gene_expression)

                    gene_names_list.append(row.drop(['cell_type', 'batch_effect']).index.tolist() + [pad_token] * (seq_length - len(row)))

                    gene_ids = [gene_vocab.get(gene_name, pad_token_id) for gene_name in row.drop(['cell_type', 'batch_effect']).index]
                    gene_id_array = np.full(seq_length, pad_token_id)
                    gene_id_array[:len(gene_ids)] = gene_ids
                    gene_id_list.append(gene_id_array)

                    cell_type_list.append(row['cell_type'])
                    batch_effect_list.append(row['batch_effect'])
                    idx_list.append(idx)
                    
            logger.info(f"processing sample {csv_file} done!")
    
    logger.info("Start converting data to DataFrame...")
    
    logger.info("Start converting gene expression to DataFrame...")
    gene_expression_df = pd.DataFrame(gene_expression_list, index=idx_list)
    
    logger.info("Start converting gene names to DataFrame...")
    gene_names_df = pd.DataFrame(gene_names_list, index=idx_list)
    
    logger.info("Start converting cell type and batch effect to DataFrame...")
    cell_type_df = pd.DataFrame({'cell_type': cell_type_list}, index=idx_list)
    batch_effect_df = pd.DataFrame({'batch_effect': batch_effect_list}, index=idx_list)
    
    logger.info("Start converting gene id to DataFrame...")
    gene_id_df = pd.DataFrame(gene_id_list, index=idx_list)
    
    logger.info("Start filling missing values in dataframe...")
    gene_names_df.fillna(pad_token, inplace=True)  # 填充缺失值
    gene_names_df = gene_names_df.apply(lambda x: x.astype(str), axis=1)  # 将基因名转换为字符串
    
    logger.info("Start mapping cell type and batch effect...")
    cell_type_df["cell_label"] = cell_type_df["cell_type"].map(celltype_vocab)
    batch_effect_df["batch_id"] = batch_effect_df["batch_effect"].map(batch_effect_vocab)
    
    logger.info("Start saving processed data...")
    gene_expression_df.to_csv(os.path.join(out_path, "gene_expression_matrix.csv"), index=True)
    gene_names_df.to_csv(os.path.join(out_path, "gene_names_df.csv"), index=True)
    cell_type_df.to_csv(os.path.join(out_path, "cell_type_df.csv"), index=True)
    batch_effect_df.to_csv(os.path.join(out_path, "batch_effect_df.csv"), index=True)
    gene_id_df.to_csv(os.path.join(out_path, "gene_id_df.csv"), index=True)

    logger.info("All samples processed!")
    logger.info(f"Saved processed data to the output directory: {out_path}!")
    
    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时间
    logger.info(f"Total processing time: {total_time} seconds \n")


if __name__ == '__main__':

    project_path = '/media/workspace/caoguangshuo/scPlantGPT'
    config = load_config(f'{project_path}/s03_scPlantGPT/cross_data/config.json')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene_counts_file_path', type=str, help='Path to the data folder')
    parser.add_argument('--output_path', type=str, help="Path to the output")
    parser.add_argument('--binned',  action='store_true', help="whether the data is binned or not")
    parser.add_argument('--species', type=str, help='species name')
    parser.add_argument('--log_name', type=str, help='log name')
    parser.add_argument('--use_species_vocab', action='store_true', help='whether to use species vocab or not')
    args = parser.parse_args()
    gene_counts_file_path = args.gene_counts_file_path
    output_dir = args.output_path
    binned = args.binned
    log_name = args.log_name
    species = args.species #'cross'
    use_species_vocab = args.use_species_vocab
    
    #logger = setup_custom_logger(f'L01.{log_name}', log_path='./log/')
    
    root_path = f'{project_path}/s03_scPlantGPT/cross_data'
    cell_type_file = os.path.join(root_path, f'{species}_celltype.meta') #'./../data/cell_type_ara.meta'
    batch_effect_file = os.path.join(root_path, f'{species}_batch_effect.meta') # './../data/ara_batch_effect.meta'
    
    # gene_vocab_path = os.path.join(root_path, f'cross_gene_vocab.json') #'./../data/gene_vocab.json'
    logger.info(f"use species vocab: {use_species_vocab}")
    if use_species_vocab:
        logger.info(f"Using species vocab")
        celltype_vocab_path = os.path.join(root_path, f'{species}_celltype_record_clean_vocab.meta.json')#'./../data/celltype_vocab_ara.meta.json'
        batch_vocab_path = os.path.join(root_path, f'{species}_batch_effect_vocab.meta.json') #'./../data/ara_batch_effect_vocab.meta.json'
        gene_vocab_path = os.path.join(root_path, f'cross_gene_vocab.json') #'./../data/gene_vocab.json'
    else:
        logger.info(f"Using cross species vocab")
        celltype_vocab_path = os.path.join(root_path, f'cross_celltype_record_clean_vocab.meta.json')#'./../data/celltype_vocab_ara.meta.json' /cross_celltype_record_clean_vocab.meta.json
        batch_vocab_path = os.path.join(root_path, f'cross_batch_effect_vocab.meta.json') #'./../data/ara_batch_effect_vocab.meta.json'
        gene_vocab_path = os.path.join(root_path, f'cross_gene_vocab.json') #'./../data/gene_vocab.json'
    
    #binned = True
    
    logger.info(f"config: {config}")
    n_bins=-1
    if binned:
        n_bins = config.n_bins - 1 ## nn.Embedding输入的是0-indexed的数值，所以bins的数量要减1
   
    ensure_directory_exists(output_dir, logger)
    
    process_gene_expression(gene_counts_file_path, cell_type_file, batch_effect_file, gene_vocab_file=gene_vocab_path, celltype_vocab_file=celltype_vocab_path, batch_effect_vocab_file=batch_vocab_path, pad_token=config.pad_token, pad_token_id=config.pad_token_id, seq_length=config.seq_length, include_zero_gene=config.include_zero_gene, binned=binned, n_bins=n_bins,pad_value=config.pad_value, out_path=output_dir, logger=logger)



