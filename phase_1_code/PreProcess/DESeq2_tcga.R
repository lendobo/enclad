# Load necessary libraries
library(DESeq2)
library(tidyverse)

# Read the data
data <- read.csv("../data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_ALL_labelled.csv")

# Filter data for 'CMS2' and 'CMS4' subtypes only
filtered_data <- data %>% filter(CMS_Label %in% c('CMS2', 'CMS4'))

# Separate the metadata (Sample_ID and CMS_Label) and the gene expression data
coldata <- filtered_data[,c('Sample_ID', 'CMS_Label')]
counts <- as.matrix(filtered_data %>% select(-Sample_ID, -CMS_Label))

# Identify samples in coldata that don't exist in counts matrix and remove them
missing_samples <- setdiff(coldata$Sample_ID, colnames(counts))
coldata <- coldata[!coldata$Sample_ID %in% missing_samples,]

# Reorder the columns of counts matrix to match order of samples in coldata
counts <- counts[, coldata$Sample_ID]

# Ensure rownames of coldata match column names of counts for DESeq2 processing
rownames(coldata) <- coldata$Sample_ID

# Construct DESeqDataSet
dds <- DESeqDataSetFromMatrix(countData = counts, colData = coldata, design = ~ CMS_Label)

# Perform differential expression analysis
dds <- DESeq(dds)

# Retrieve and order results by p-value
res <- results(dds)
res_ordered <- res[order(res$pvalue),]

# View top differentially expressed genes
head(res_ordered)
