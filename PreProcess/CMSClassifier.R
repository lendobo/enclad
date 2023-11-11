# install_github("Sage-Bionetworks/CMSclassifier")

# source("http://depot.sagebase.org/CRAN.R")
# pkgInstall("synapseClient")

# library(synapseClient)

library(devtools)
library(CMSclassifier)

# synapseLogin()

# sampleData <- read.table(synGet("syn4983432")@filePath, sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)

filepath <- "/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/TCGACRC_expression.tsv"

sampleData <- read.table(filepath, sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)

Rfcms <- CMSclassifier::classifyCMS(t(sampleData), method = "RF")[[3]]
SScms <- CMSclassifier::classifyCMS(t(sampleData), method = "SSP")[[3]]
