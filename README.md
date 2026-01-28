# scPlantLLM: A Foundation Model for Plant Single-Cell Expression Atlases

**scPlantLLM** (Single-cell Plant Large Language Model) is a transformer-based foundation model designed to explore the complexity of plant single-cell RNA sequencing (scRNA-seq) data. Trained on millions of plant single-cell data points, specifically *Arabidopsis thaliana*, scPlantLLM treats single cells as "sentences" and genes as "words" to uncover intricate biological patterns.

By employing a sequential pretraining strategy with masked language modeling, scPlantLLM overcomes common challenges in scRNA-seq analysis, including batch integration, cell type annotation, and gene regulatory network (GRN) inference.

### Key Features

* **Foundation Model Architecture:** Built on a Transformer architecture tailored for plant genomics.
* **High-Resolution Analysis:** Excels in clustering and identifying subtle cellular subtypes.
* **Zero-Shot Learning:** zero-shot cell type annotation.
* **Robust Integration:** effectively handles batch effects across diverse datasets.
* **Interpretability:** Identifies biologically meaningful Gene Regulatory Networks (GRNs).

---

## 🛠️ Environment Setup

To ensure successful reproduction of the project, please set up both Python and R environments as detailed below.

### 1. Python Environment

**Step 1: Create the Conda environment**
We recommend Python 3.10 for compatibility with Flash Attention.

```bash
conda create -n scPlantLLM_Py_Env python=3.10
conda activate scPlantLLM_Py_Env

```

**Step 2: Install dependencies**
*Note: Please comment out `flash_attn` in `requirements.txt` before running the following command, as we will install it manually.*

```bash
pip install -r scPlantLLM_python_environment.txt

```

**Step 3: Manually install Flash Attention**
Install the specific wheel file for `flash-attention` (Compatible with: CUDA 12.2, Torch 2.3, Python 3.10, Linux x86_64).

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

```

### 2. R Environment

**Step 1: Create the environment from YAML**

```bash
conda env create -f scPlantLLM_R_environment.yml
conda activate scPlantLLM_R_Env

```

**Step 2: Verify the installation**
Run the following R commands to ensure `Seurat` and `rhdf5` are correctly loaded:

```r
library(Seurat)
library(rhdf5)
sessionInfo()

```

---

## 🚀 Usage Workflow

### Stage 1: Data Extraction

Convert raw RDS files into the `.h5` and `.csv` formats required by the model.

**Prerequisites:**

* R ≥ 4.0 (with Seurat package)
* Python 3.10 (recommended)

**Command:**

```bash
conda activate scPlantLLM_R_env
Rscript extract_rds_data.R ./data/raw ./data/processed

```

* `./data/raw`: Directory containing input RDS data.
* `./data/processed`: Directory for output `.h5` and `meta.csv` files.
* **Note:** Demo data can be downloaded from  
[SRP169576_RAW.tar.gz (1.1GB)](https://box.nju.edu.cn/seafhttp/f/66ed8930449d41e98b60/?op=view).

This dataset is an **independent validation dataset** that was **not used during model training**.  
After downloading, please extract the files using `tar`:

```bash
tar -xzvf SRP169576_RAW.tar.gz
```

### Stage 2: Generate Metadata Information

In this stage, you will generate vocabularies for batch effects and (optionally) cell types. Choose the option that matches your data availability.

#### Option A: With Cell Type Information

Use this if you have ground-truth cell type labels and want to perform supervised tasks or fine-tuning.

**1. Handle Batch Effects:**

```bash
conda activate scPlantLLM_Py_Env
python prepare_meta.py \
    --input_path ./data/processed \
    --output_path ./data/processed/has_celltype \
    --file_prefix batch_effect \
    --col_name orig.ident \
    --do_batch

```

**2. Handle Cell Types:**

```bash
python prepare_meta.py \
    --input_path ./data/processed \
    --output_path ./data/processed/has_celltype \
    --file_prefix cell_type \
    --col_name celltype \
    --do_cell_type

```

#### Option B: Without Cell Type Information

Use this for zero-shot scenarios or unsupervised embedding generation.

```bash
python prepare_meta.py \
    --input_path ./data/processed \
    --output_path ./data/processed/dont_have_celltype \
    --file_prefix batch_effect \
    --col_name orig.ident \
    --do_batch

```

### Stage 3: Build Model Input Data

Preprocess the metadata and gene expression data into model-ready HDF5 chunks.

#### For Data Without Cell Type Labels

```bash
python preprocess_data.py \
    --input_path ./data/processed \
    --output_path ./data/processed/dont_have_celltype \
    --gene_vocab_file gene_vocab.json \
    --batch_effect_file ./data/processed/dont_have_celltype/batch_effect.meta \
    --batch_effect_vocab_file ./data/processed/dont_have_celltype/batch_effect_vocab.meta.json

```

**Output:** `test_chunk_1.h5`

#### For Data With Cell Type Labels (Split: Train/Valid/Test)

```bash
python preprocess_data.py \
    --input_path ./data/processed \
    --output_path ./data/processed/has_celltype \
    --gene_vocab_file gene_vocab.json \
    --has_celltype \
    --cell_type_file ./data/processed/has_celltype/cell_type.meta \
    --cell_type_vocab_file ./data/processed/has_celltype/cell_type_vocab.meta.json \
    --batch_effect_file ./data/processed/has_celltype/batch_effect.meta \
    --batch_effect_vocab_file ./data/processed/has_celltype/batch_effect_vocab.meta.json \
    --test_size 0.1

```

**Outputs:** `train_chunk_1.h5`, `valid_chunk_1.h5`, `test_chunk_1.h5`

### Stage 4: Inference & Downstream Analysis

📄 **Tutorial.ipynb**  
– Standard inference workflow.

📄 **ZeroShot_Tutorial.ipynb**  
– Zero-shot inference on **unseen datasets**, demonstrating the model’s generalization ability without additional training.


---

## 📚 Citation

If you use scPlantLLM in your research, please cite our work:

```bibtex
@article{10.1093/gpbjnl/qzaf024,
    author = {Cao, Guangshuo and Chao, Haoyu and Zheng, Wenqi and Lan, Yangming and Lu, Kaiyan and Wang, Yueyi and Chen, Ming and Zhang, He and Chen, Dijun},
    title = {Harnessing the Foundation Model for Exploration of Single-cell Expression Atlases in Plants},
    journal = {Genomics, Proteomics & Bioinformatics},
    pages = {qzaf024},
    year = {2025},
    month = {03},
    issn = {1672-0229},
    doi = {10.1093/gpbjnl/qzaf024},
    url = {https://doi.org/10.1093/gpbjnl/qzaf024},
}

```
