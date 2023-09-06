# %%
import pandas as pd

# Reading the TMT_prot_tumor.csv file
tmt_prot_tumor_df = pd.read_csv("data/LinkedOmics/RNASeq_tumor_normal.tsv", sep="\t")

# Reading the Node_Table_Entrez.csv file, setting all float values to int
node_table_entrez_df = pd.read_csv(
    "data/LinkedOmics/Node_Table_RNA_Entrez.csv", dtype={"Entrez Gene": "Int64"}
)

# find duplicate row names
print(node_table_entrez_df[node_table_entrez_df.index.duplicated(keep=False)])

# # Checking the first few rows of each dataframe to understand their structures
# tmt_prot_tumor_df.head()

# node_table_entrez_df.head()

# %%

# Create a dictionary for mapping 'attrib_name' to 'Entrez Gene' from the Node_Table_Entrez.csv file
mapping_dict = dict(
    zip(node_table_entrez_df["attrib_name"], node_table_entrez_df["Entrez Gene"])
)


# Replace the 'attrib_name' in TMT_prot_tumor.csv with the corresponding 'Entrez Gene'
# Remove rows that do not have a mapping
tmt_prot_tumor_df["attrib_name"] = tmt_prot_tumor_df["attrib_name"].map(mapping_dict)
tmt_prot_tumor_df.dropna(subset=["attrib_name"], inplace=True)

# Save the updated DataFrame to a new CSV file
updated_csv_path = "data/LinkedOmics/RNA_expr_Entrez.tsv"
tmt_prot_tumor_df.to_csv(updated_csv_path, index=False, sep="\t")

# %%
# Reading the RNA_expr_Entrez.tsv file to check for duplicate row names
rna_expr_entrez_df = pd.read_csv('data/LinkedOmics/RNA_expr_Entrez.tsv', sep='\t', index_col=0)

# Checking for duplicate row names
duplicate_row_names = rna_expr_entrez_df.index.duplicated(keep=False)
duplicated_row_names = rna_expr_entrez_df.index[duplicate_row_names].unique()

# Displaying the number and list of duplicated row names, if any
num_duplicated_row_names = len(duplicated_row_names)
duplicated_row_names, num_duplicated_row_names


# %%
