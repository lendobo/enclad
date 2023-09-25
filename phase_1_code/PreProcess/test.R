counts <- read.csv("/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/ENCLAD/phase_1_code/data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_ALL_labelled.csv")
head(counts)

# drop first row

counts <- counts[-2, ]

# set first column as index
rownames(counts) <- counts[, 1]

# drop first column
counts <- counts[, -1]

# show matrix
head(counts)
