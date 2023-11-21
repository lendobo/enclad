# if (!require("BiocManager", quietly = TRUE)) {
#     install.packages("BiocManager")
# }

# BiocManager::install("org.Hs.eg.db")


library(devtools)
library(CMSclassifier)
library(org.Hs.eg.db)

filepath <- "data/Synapse/TCGA/TCGACRC_expression.tsv"

sampleData <- read.table(filepath, header = T) # replace with wherever your file is
rownames(sampleData) <- sampleData$feature


entrez_ids <- mapIds(org.Hs.eg.db,
    keys = rownames(sampleData),
    column = "ENTREZID",
    keytype = "SYMBOL",
    multiVals = "first"
)

I <- !is.na(entrez_ids) # HUGO and Entrez are not a perfect match

sampleData <- sampleData[I, ]
rownames(sampleData) <- entrez_ids[I]
exp <- as.matrix(sampleData[, 2:ncol(sampleData)]) # first column is gene name
rownames(exp) <- rownames(sampleData)
colnames(exp) <- names(sampleData)[2:ncol(sampleData)]
classifyCMS.SSP(exp)
