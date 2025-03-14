library('Seurat')
library(rhdf5)

# 保存为 HDF5 格式
save_h5 <- function(data, cell_names, gene_names, file_path, data_type) {
  # 获取数据的行列数
  data = t(data)#如果数据在 HDF5 文件中是按列优先存储的，那么用 `h5py` 读取时可能会导致转置。R 的默认行为是按列存储，因此在这里需要转置，才能保持python读取时不需要转置
  n_rows <- nrow(data)
  n_cols <- ncol(data)
    
  # 设置每个chunk的大小，按行分块
  # 例如，设定每个chunk包含 10000 行
  chunk_rows <- min(1000, n_rows)
  chunk_size <- c(chunk_rows, n_cols)  # 动态调整chunk行数
  
  if (file.exists(file_path)) {
    file.remove(file_path)
    cat(sprintf("File %s already exists. It has been removed and will be recreated.\n", file_path))
  }
  
  h5createFile(file_path)
  # 检查并创建父级组
  parent_group <- paste0(data_type, '/')
  h5createGroup(file_path, parent_group)
    
  # 创建数据集并设置chunk大小
  h5createDataset(file_path, paste0(data_type, "/data"), dims = c(n_rows, n_cols),
                  chunk = chunk_size, storage.mode = 'double')
  
  # 写入数据矩阵
  h5write(data, file_path, paste0(data_type, "/data")) 
  
  # 保存细胞名称和基因名称作为额外的属性
  h5write(cell_names, file_path, paste0(data_type, "/cell_names"))
  h5write(gene_names, file_path, paste0(data_type, "/gene_names"))
  
  cat(sprintf("Saved %s data to %s\n", data_type, file_path))
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Please provide two parameters: input path and result path")
}

input_path <- args[1]
result_path <- args[2]
print(input_path)
print(result_path)


root_path <- input_path
matrix_result_path <- result_path
meta_result_path <- result_path

file_list <- list.files(path = root_path, pattern = "\\.rds$", full.names = TRUE)

print(file_list)
kv <- 3000
all_hvg_genes <- c()

for (file in file_list) {
# 从文件名中提取前缀（去掉路径和后缀）
  prefix <- sub("\\.rds$", "", basename(file))
  print("========================================================================================================")
  cat("Processing", file, "\n" )
  print("========================================================================================================")
  seurat_obj <- readRDS(file)
    
  # count data:
  gene_counts <- GetAssayData(object = seurat_obj, assay = "RNA", slot = "data")
  top_variable_genes <- head(VariableFeatures(seurat_obj), kv)
  cell_gene_counts <- gene_counts[top_variable_genes, ]
  cell_gene_counts <- t(as.matrix(cell_gene_counts))
      
        # 提取细胞名称和基因名称
  save_cell_names <- rownames(cell_gene_counts)
  save_gene_names <- colnames(cell_gene_counts)

  # 保存为 HDF5 格式
  h5_file_path <- paste(matrix_result_path, paste0(prefix, ".h5"), sep = '/')
  save_h5(cell_gene_counts, save_cell_names, save_gene_names, h5_file_path, "count")
  rm(cell_gene_counts)
  gc()
  
  ## metadata:
  metadata <- seurat_obj@meta.data
  metadata <- as.matrix(metadata)
  
  metadata_csv_file <- paste(meta_result_path,  paste0(prefix, ".meta.csv"), sep = '/')
  write.csv(metadata, file = metadata_csv_file, row.names = TRUE)

}

hvg_union_csv_file <- paste(meta_result_path, "hvg_union.csv", sep = '/')
write.csv(data.frame(HVG = all_hvg_genes), file = hvg_union_csv_file, row.names = FALSE)

cat("All HVG gene names saved to:", hvg_union_csv_file, "\n")




