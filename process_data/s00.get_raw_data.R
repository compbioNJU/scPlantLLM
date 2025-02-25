library('Seurat')
root_path <- "/mnt/public6/caoguangshuo/scPlantGPT/s01.plans_raw_data/rds"
matrix_result_path <- "/mnt/public6/caoguangshuo/scPlantGPT/s02.data_process/s01.matrix/cross-species_hgv1w"
normalize_matrix_result_path <- "/mnt/public6/caoguangshuo/scPlantGPT/s02.data_process/s01.matrix_normalize/cross-species_hgv1w"
meta_result_path <- "/mnt/public6/caoguangshuo/scPlantGPT/s02.data_process/s04.meta_data"

file_list <- list.files(path = root_path, pattern = "\\.rds$", full.names = TRUE)

print(file_list)
kv <-3000
all_gene_names <- c()

for (file in file_list) {

  prefix <- sub("\\.rds$", "", basename(file))
  print("========================================================================================================")
  cat("Processing", file, "\n" )
  print("========================================================================================================")
  seurat_obj <- readRDS(file)
    
  # count data:
  gene_counts <- GetAssayData(object = seurat_obj, assay = "RNA", slot = "data")

  gene_names <- rownames(gene_counts)
  all_gene_names <- union(all_gene_names, gene_names)

  top_variable_genes <- head(VariableFeatures(seurat_obj), kv)
  print(length(VariableFeatures(seurat_obj)))
  print(length(top_variable_genes))
  cell_gene_counts <- gene_counts[top_variable_genes, ]
  cell_gene_counts_matrix <- t(as.matrix(cell_gene_counts))

  matrix_csv_file <- paste(matrix_result_path, paste0(prefix, "_matrix.csv"), sep = '/')
  write.csv(cell_gene_counts_matrix, file = matrix_csv_file, row.names = TRUE)    
  cat("Saved", matrix_csv_file, "\n")
  
  if (!"SCT" %in% Assays(seurat_obj)) {
  seurat_obj <- SCTransform(seurat_obj, verbose = FALSE)
}

  # normalize data:
  gene_counts <- GetAssayData(object = seurat_obj, assay = "SCT", slot = "data")
  top_variable_genes <- head(VariableFeatures(seurat_obj), kv)
  cell_gene_counts <- gene_counts[top_variable_genes, ]
  cell_gene_counts_matrix <- t(as.matrix(cell_gene_counts))
    
  normal_matrix_csv_file <- paste(normalize_matrix_result_path, paste0(prefix, "_normalize_matrix.csv"), sep = '/')
  write.csv(cell_gene_counts_matrix, file = normal_matrix_csv_file, row.names = TRUE)  
  cat("Saved", normal_matrix_csv_file, "\n")
  
  # metadata:
  metadata <- seurat_obj@meta.data
  metadata <- as.matrix(metadata)
  
  metadata_csv_file <- paste(meta_result_path,  paste0(prefix, ".meta.csv"), sep = '/')
  write.csv(metadata, file = metadata_csv_file, row.names = TRUE)
  cat("Saved", metadata_csv_file, "\n")
}

gene_names_file <- paste("/mnt/public6/caoguangshuo/scPlantGPT/s02.data_process", "cross_gene_names_hgv1w.txt", sep = '/')
writeLines(all_gene_names, gene_names_file)
cat("Saved", gene_names_file, "\n")
cat("All done!")
