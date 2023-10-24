# %%

# Importing necessary libraries
import pandas as pd

# Load the list of nodes (proteins) that are part of the PPI network
ppi_nodes_path = 'data/results/PPI_Nodes_Synapse.csv'
ppi_nodes = pd.read_csv(ppi_nodes_path)

# Load the proteomics data
proteomics_data_path = 'data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_CMS2.csv'
proteomics_data = pd.read_csv(proteomics_data_path)

# Display the first few rows of each data frame to understand their structure
ppi_nodes.head()

proteomics_data.head()

# %%

# Only keep those rows in the proteomics data that are also in the PPI network
proteomics_data_GC = proteomics_data[proteomics_data['commonName'].isin(ppi_nodes['commonName'])]

proteomics_data_GC.shape

# %%
