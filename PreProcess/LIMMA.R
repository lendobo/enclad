library(limma)
library(ggplot2)

# Read in the data
protein_data <- read.csv("/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/Proteomics_CMS_groups/protes_only.csv", row.names = 1)
cms_labels <- read.csv("/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/Proteomics_CMS_groups/labels_protes.csv", header = FALSE)

# Convert to factor and remove the 'CMS_Label' level if it exists
cms_factor <- factor(cms_labels$V2)
cms_factor <- cms_factor[cms_factor != "CMS_Label"]
cms_factor <- factor(cms_factor) # Re-level the factor

# Create a design matrix
design <- model.matrix(~ 0 + cms_factor)
colnames(design) <- levels(cms_factor)

# Fit the linear model
fit <- lmFit(protein_data, design)

# Create contrast matrix
contrast.matrix <- makeContrasts(
  CMS1 - CMS2,
  CMS1 - CMS3,
  CMS1 - CMS4,
  CMS2 - CMS3,
  CMS2 - CMS4,
  CMS3 - CMS4,
  levels = design
)

# Fit contrasts and compute statistics
fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)

# Get results
results <- topTable(fit2, adjust = "fdr", sort.by = "B", number = Inf)

head(results)

# write results to file
write.csv(results, file = "/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/Proteomics_CMS_groups/limma_results.csv")

# Create a data frame from the results
results_df <- as.data.frame(results)

# Comparison groups
comp_col <- "CMS2...CMS4"

# Define a new column that specifies the color based on adj.P.Val and log2 fold change
results_df$color <- ifelse(results_df$adj.P.Val < 0.05 & results_df[, comp_col] > 0, "red",
  ifelse(results_df$adj.P.Val < 0.05 & results_df[, comp_col] < 0, "blue", "grey")
)

# Make a Volcano plot
ggplot(results_df, aes(x = results_df[, comp_col], y = -log10(P.Value), color = color)) +
  geom_point() +
  ggtitle("Volcano Plot, CMS2 vs CMS4") +
  xlab("Log2(Fold Change)") +
  ylab("-Log10 P-Value") +
  scale_color_manual(
    values = c("red" = "red", "blue" = "blue", "grey" = "grey"),
    name = "Adj.P.Val < 0.05",
    labels = c("Down", "N.S.", "Up")
  )

# Get results for CMS2 vs CMS4 contrast
results_CMS2_CMS4 <- topTable(fit2, adjust="fdr", coef="CMS2 - CMS4", number=Inf)

# Now we define top100_CMS2 as the top 100 entries from this result set
top500_CMS2 <- head(results_CMS2_CMS4, 500)


# Write the top 100 genes overexpressed in CMS2 to a text file
write.table(row.names(top500_CMS2), file = "../data/Synapse/TCGA/Proteomics_CMS_groups/top500_CMS2_PROT.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)


# Define top100_CMS4 as the top entries from the CMS2 vs CMS4 results with negative logFC
# This indicates higher expression in CMS4 compared to CMS2
top_CMS4_overexpressed <- results_CMS2_CMS4[results_CMS2_CMS4$logFC < 0, ]

# Since we are interested in the top 100, we need to sort them by logFC in ascending order to get the most underexpressed in CMS2 which means overexpressed in CMS4
top_CMS4_overexpressed <- head(top_CMS4_overexpressed[order(top_CMS4_overexpressed$logFC), ], 500)

# Output the gene names of the top 100 overexpressed in CMS4 to a text file
write.table(row.names(top_CMS4_overexpressed), file = "../data/Synapse/TCGA/Proteomics_CMS_groups/top500_CMS4_PROT.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)

