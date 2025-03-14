import argparse
import json
import os 
import pandas as pd
import json
from collections import Counter


def generate_json(input_file, output_file):
    gene_dict = {}
    with open(input_file, 'r') as f:
        genes = f.read().splitlines()
        for idx, gene in enumerate(genes):
            gene_dict[gene] = idx # + 1  # Assign unique ID starting from 0 or +1
    
    with open(output_file, 'w') as f:
        json.dump(gene_dict, f, indent=4)
    

def get_metadata(folder_path, col_name, save_col_name=None):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith("meta.csv")]
    print(f"processing {csv_files} in {folder_path}")

    result_df = pd.DataFrame()

    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)

        df = pd.read_csv(file_path)

        if col_name not in df.columns:
            print(f"Column {col_name} does not exist in file {file_name}, file will be skipped.")
            continue

        first_column = df.iloc[:, 0]
        last_column = df[col_name]
        result_df = pd.concat([result_df, pd.concat([first_column, last_column], axis=1)], axis=0)

    result_df.columns = ["cell", col_name]
    if save_col_name is not None:
        result_df.columns = ["cell", save_col_name]
    
    return result_df


def save_metadata(result_df, output_path, file_prefix):
    meta_path = os.path.join(output_path, file_prefix + ".meta")
    result_df.to_csv(meta_path, index=False)

    names_path = os.path.join(output_path, file_prefix + "_names.meta")
    with open(names_path, 'w') as f:
        for name in result_df.iloc[:, 1].unique():
            f.write(str(name) + '\n')
    
    vocab_path = os.path.join(output_path, file_prefix + "_vocab.meta.json")
    generate_json(names_path, vocab_path)
    os.remove(names_path)
    
    print(vocab_path, "Done!")
    print(meta_path, "Done!")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--file_prefix", type=str, required=True)
    parser.add_argument("--do_cell_type", action='store_true', default=False)
    parser.add_argument("--do_batch", action='store_true', default=False)
    parser.add_argument("--col_name", default='orig.ident', type=str, required=True)

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    file_prefix = args.file_prefix
    col_name = args.col_name
    do_cell_type = args.do_cell_type
    do_batch = args.do_batch
    if do_cell_type:
        save_col_name = 'celltype'
    if do_batch:
        save_col_name = 'orig.ident'
    result_df = get_metadata(input_path, col_name, save_col_name)
    
    save_metadata(result_df, output_path, file_prefix)

if __name__ == "__main__":
    main()
        
        
        

        