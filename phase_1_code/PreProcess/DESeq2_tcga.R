# Load necessary libraries
library(DESeq2)

# Read in the data
data <- read.csv("path_to_your_file/TCGACRC_expression_ALL_labelled.csv", header = TRUE, row.names = 1)

# Split data into metadata and count data
metadata <- data[, c(1, 2)]
rownames(metadata) <- rownames(data)
counts <- data[, -c(1, 2)]

# Revert the log2 transformation
counts <- 2^counts

# Check dimensions
print(dim(counts))
print(dim(metadata))

# Ensure the rownames of metadata and counts are in the same order
metadata <- metadata[rownames(counts), ]
# Now, attempt to create the DESeq2 dataset object again
dds <- DESeqDataSetFromMatrix(
    countData = counts,
    colData = metadata,
    design = ~condition
)

# Estimate size factors and dispersion
dds <- estimateSizeFactors(dds)
dds <- estimateDispersions(dds)

# Apply Variance Stabilizing Transformation
vst_data <- assay(varianceStabilizingTransformation(dds))

# Save the VST data to a CSV file
write.csv(vst_data, "path_to_save/VST_transformed_data.csv")
