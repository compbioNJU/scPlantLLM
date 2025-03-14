   
# scPlantLLM

### scPlantLLM(Single-cell Plant Large Language Model), a transformer-based model specifically designed for the exploration of single-cell expression atlases in plants

This project aims to pretrain a large model using 1 million plant single-cell data and further pretrain a cell annotation model and a batch integration model based on this pretrained model. Ultimately, we can use these pretrained models for cell annotation, providing an efficient method for handling large-scale plant single-cell data.

## Stage 1: Data Extraction

### Requirements
- R ≥4.0
- Seurat package
- Python 3.11+
- Refer to `requirements.txt` for the list of dependencies.

### Extract Data from RDS Files

Run the following command to extract data from RDS files and generate `.h5` and `meta.csv` files:

```bash
Rscript extract_rds_data.R ./data/raw ./data/processed
```

- `./data/raw`: Path to the input RDS data.
- `./data/processed`: Path to the output `.h5` and `meta.csv` files.

If fine-tuning the model is required, `meta.csv` can be used to construct the `celltype.meta` file.

## Stage 2: Generate Metadata Information

### Handling Batch Effect

#### Run the script:
```bash
python prepare_meta.py \
    --input_path ./data/processed \
    --output_path ./data/processed/has_celltype \
    --file_prefix batch_effect \
    --col_name orig.ident \
    --do_batch
```

#### Output:
```shell
├── batch_effect.meta
├── batch_effect_vocab.meta.json
```

### Handling Cell Type Information

#### Run the script:
```bash
python prepare_meta.py \
    --input_path ./data/processed \
    --output_path ./data/processed/has_celltype \
    --file_prefix cell_type \
    --col_name seurat_clusters \
    --do_cell_type
```

#### Output:
```shell
├── cell_type.meta
├── cell_type_vocab.meta.json
```

## Stage 3: Build Model Input Data

### Without Cell Type Information

```bash
python preprocess_data.py \
    --input_path ./data/processed \
    --output_path ./data/processed/dont_have_celltype \
    --gene_vocab_file ./config/gene_vocab.json \
    --batch_effect_file ./data/processed/has_celltype/batch_effect.meta \
    --batch_effect_vocab_file ./data/processed/has_celltype/batch_effect_vocab.meta.json
```

### With Cell Type Information

```bash
python preprocess_data.py \
    --input_path ./data/processed \
    --output_path ./data/processed/has_celltype \
    --gene_vocab_file ./config/gene_vocab.json \
    --has_celltype \
    --cell_type_file ./config/celltype_meta.csv \
    --cell_type_vocab_file ./config/celltype_vocab.json \
    --batch_effect_file ./data/processed/has_celltype/batch_effect.meta \
    --batch_effect_vocab_file ./data/processed/has_celltype/batch_effect_vocab.meta.json \
    --test_size 0.1
```

## Stage 4: Use the Model

The model usage is detailed in the tutorial notebook:
```bash
Tutorial.ipynb
```

## Citation

```shell
@article{scPlantLLM,
  title={Harnessing the foundation model for exploration of single-cell expression atlases in plants},
  author={},
  year={2025},
  }
```