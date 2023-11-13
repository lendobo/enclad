# %%
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pymnet import *

# %%
TRRUST_df = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/data/TRRUSTv2/trrust_rawdata.human.tsv', sep='\t')
data_proteins = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/data/Synapse/TCGA/Proteomics_CMS_groups/Prot_Names.csv')
data_RNA = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/data/Synapse/TCGA/RNA_CMS_groups/RNA_names.csv')

#tf_proteins is the first column of tf_bs_proteins
tf_proteins = TRRUST_df.iloc[:,0].unique()

# check overlap
prot_tfs = np.intersect1d(data_proteins, tf_proteins)
print(len(prot_tfs))

rna_tfs = np.intersect1d(data_RNA, prot_tfs)
print(len(rna_tfs))

# incex tf_bs_proteins at prot_tfs
TRRUST_df = TRRUST_df[TRRUST_df.iloc[:,0].isin(prot_tfs)]
tf_targets_db = TRRUST_df.iloc[:,1].unique()

targeted_RNA = np.intersect1d(tf_targets_db, data_RNA)
print(len(targeted_RNA))

targeted_PROTS = np.intersect1d(targeted_RNA, data_proteins)
print(len(targeted_PROTS))

# concatenate prot_tfs and  targeted_PROTS
all_PROTS = np.concatenate((prot_tfs, targeted_PROTS))
print(len(all_PROTS))

# write to csv
pd.DataFrame(all_PROTS).to_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/Diffusion/data/all_TRRUST_prots.csv', index=False)


# %%
# Filter the TRRUST entries
# Check if entries in the first column of tfs_and_targets are in the first column of prot_tfs
condition1 = TRRUST_df['TF_PROT'].isin(prot_tfs)
# Check if entries in the second column of tfs_and_targets are in the second column of targeted_RNA
condition2 = TRRUST_df['BS_RNA'].isin(targeted_RNA)

# Filter tfs_and_targets where both conditions are true
TRRUST_df = TRRUST_df[condition1 & condition2]

def bs_to_bs_prot(bs_value):
    return bs_value if bs_value in targeted_PROTS else np.nan

def tf_prot_to_tf_rna(tf_prot_value):
    return tf_prot_value if tf_prot_value in rna_tfs else np.nan

TRRUST_df['BS_PROT'] = TRRUST_df['BS_RNA'].apply(bs_to_bs_prot)

TRRUST_df['TF_RNA'] = TRRUST_df['TF_PROT'].apply(tf_prot_to_tf_rna)

# place the TF_RNA column before the TF column
cols = TRRUST_df.columns.tolist()
cols = cols[-1:] + cols[:-1]

TRRUST_df = TRRUST_df[cols]

# write to tsv
TRRUST_df.to_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/Diffusion/TRRUST_expanded.tsv', sep='\t', index=False)




# %%
# Create an empty directed graph
TRRUST_sample = TRRUST_df.sample(n=10, random_state=1)
# include all rows that have 'ABL1
TRRUST_sample = TRRUST_df[TRRUST_df['TF_PROT'] == 'ABL1']

G = nx.DiGraph()

# Add edges for each relationship
for _, row in TRRUST_sample.iterrows():
    if pd.notna(row['TF_RNA']) and pd.notna(row['TF_PROT']):
        G.add_edge(row['TF_RNA'] + ".r>", row['TF_PROT'] + ".p>")
    if pd.notna(row['TF_PROT']) and pd.notna(row['BS_RNA']):
        G.add_edge(row['TF_PROT'] + ".p>", row['BS_RNA'] + '.r]')
    if pd.notna(row['BS_RNA']) and pd.notna(row['BS_PROT']):
        G.add_edge(row['BS_RNA'] + '.r]', row['BS_PROT'] + ".p]")


# Define node colors based on unique node types
node_colors = []
for node in G:
    if ".p>" in node:
        node_colors.append('crimson')  # Color for 'TF_PROT' nodes
    elif ".p]" in node:
        node_colors.append('salmon')     # Color for 'BS_PROT' nodes
    elif ".r>" in node:
        node_colors.append('darkolivegreen')    # Color for 'TF_RNA' nodes
    else:  # ".r]" in node
        node_colors.append('lightgreen') # Color for 'BS_RNA' nodes

# Draw the graph with specified node colors
plt.figure(figsize=(10,10), dpi=300)
nx.draw_networkx(G, pos=nx.spring_layout(G), 
                 node_size=250, 
                 with_labels=True,
                 edge_color='lightgrey',
                 node_color=node_colors,  # Use the node colors defined above
                 arrowsize=25, 
                 alpha=0.75,
                 cmap='plasma')

plt.show()

TRRUST_sample.head()

# %%
tf_nodes = set(TRRUST_sample['TF_PROT'])
pos = nx.bipartite_layout(G, tf_nodes, align='vertical')

plt.figure(figsize=(10, 10))
nx.draw_networkx(G, pos=pos, node_size=10, edge_color='lightgrey', with_labels=False)
plt.title('Bipartite Network Visualization')
plt.show()

# %%
