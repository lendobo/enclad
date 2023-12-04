library(devtools)
library(CMSclassifier)


sampleData <- read.table("data/TCGA-COAD-L4/tmp/RNA_samples_for_RPPA_scaled.tsv", header = T, row.names = 1) # replace with wherever your file is
# rownames(sampleData) <- sampleData$gene_name

# print head of rownames(sampleData)
head(rownames(sampleData))

library(org.Hs.eg.db)
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

# Write to file
write.table(classifyCMS.SSP(exp), file = "data/TCGA-COAD-L4/tmp/TCGACRC_CMS_CLASSIFIER_LABELS.tsv", sep = "\t", quote = FALSE, col.names = NA)
