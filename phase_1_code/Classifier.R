# install devtools
library(devtools)
library(CMSclassifier)
# library(synapser)


# load data
myData <- read.table("data/LinkedOmics/RNA_expr_Entrez.tsv", sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)

# # Classify using Random Forest
Rfcms <- CMSclassifier::classifyCMS(t(myData), method = "RF")[[3]]

# # Classify using Single Sample Predictor (SSP)
# SScms <- CMSclassifier::classifyCMS(t(myData), method = "SSP")[[3]]
