# %%
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from diffupy.diffuse import run_diffusion_algorithm
from diffupy.matrix import Matrix
from diffupy.diffuse_raw import diffuse_raw
from diffupy.kernels import regularised_laplacian_kernel, diffusion_kernel


# %%
TRRUST_df = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/data/TRRUSTv2/trrust_rawdata.human.tsv', sep='\t')
data_proteins = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/data/Synapse/TCGA/Proteomics_CMS_groups/Prot_Names.csv')
data_RNA = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/data/Synapse/TCGA/RNA_CMS_groups/RNA_names.csv')

#tf_proteins is the first column of tf_bs_proteins
tf_proteins = TRRUST_df.iloc[:,0].unique()

# check overlap
prot_tfs = np.intersect1d(data_proteins, tf_proteins)
print(f'Number of TFs in sample: {len(prot_tfs)}')

rna_tfs = np.intersect1d(data_RNA, prot_tfs)
print(f'Number of TF mRNAs: {len(rna_tfs)}')

# incex tf_bs_proteins at prot_tfs
TRRUST_df = TRRUST_df[TRRUST_df.iloc[:,0].isin(prot_tfs)]
tf_targets_db = TRRUST_df.iloc[:,1].unique()

targeted_RNA = np.intersect1d(tf_targets_db, data_RNA)
print(f'Number of targeted RNA: {len(targeted_RNA)}')

targeted_PROTS = np.intersect1d(targeted_RNA, data_proteins)
print(f'Proteins encoded by targeted RNA:  {len(targeted_PROTS)}')

DEA_PROTS_CMS4 = pd.read_csv('../data/Synapse/TCGA/Proteomics_CMS_groups/top_100_CMS4_PROT.csv')
DEA_RNA = pd.read_csv('../data/Synapse/TCGA/RNA_CMS_groups/top100_RNA_CMS4.csv')

# check overlap between DEA_PROTS_CMS4 and targeted_PROTS
DEA_PROTS_CMS4 = DEA_PROTS_CMS4.to_numpy()
DEA_targeted_PROTS = np.intersect1d(DEA_PROTS_CMS4, targeted_PROTS)
print(f'Top {len(DEA_PROTS_CMS4)} DEA proteins in targeted_PROTS: {len(DEA_targeted_PROTS)}')

# check overlap between DEA_PROTS_CMS4 and prot_tfs
DEA_TF_PROTS = np.intersect1d(DEA_PROTS_CMS4, prot_tfs)
print(f'Top {len(DEA_PROTS_CMS4)} DEA proteins in prot_tfs: {len(DEA_TF_PROTS)}')
print(f'RNA coding for Top DEA TF: {np.intersect1d(DEA_TF_PROTS, rna_tfs)}\n')


# concatenate prot_tfs and  targeted_PROTS
all_PROTS = list(set(np.concatenate((prot_tfs, targeted_PROTS))))
print(f'Total number of scaffold proteins (Before STRING expansion): {len(all_PROTS)}')



# # write to csv
# pd.DataFrame(all_PROTS).to_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/Diffusion/data/all_TRRUST_prots.csv', index=False)


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

# sum over columns 'TF_RNA, TF_PROT, BS_RNA, BS_PROT'
num_nodes = TRRUST_df[['TF_RNA', 'TF_PROT', 'BS_RNA', 'BS_PROT']].nunique().sum()
print(num_nodes)


# write to tsv
TRRUST_df.to_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/Diffusion/TRRUST_expanded.tsv', sep='\t', index=False)

# DataFrame for RNA (TF_RNA + BS_RNA)
df_rna = pd.DataFrame({
    'RNA_TF_BS': list(set(TRRUST_df['TF_RNA']) | set(TRRUST_df['BS_RNA']))
})

# DataFrame for Proteins (TF_PROT + BS_PROT)
df_prot = pd.DataFrame({
    'PROT_TF_BS': list(set(TRRUST_df['TF_PROT']) | set(TRRUST_df['BS_PROT']))
})

df_prot.shape


# %%
TRRUST_sample = TRRUST_df.sample(n=10, random_state=1)
# include all rows that have 'ABL1
TRRUST_sample = TRRUST_df[TRRUST_df['TF_PROT'] == 'ABL1']

# CREATING THE BASIC DOGMA GRAPH
G = nx.DiGraph()

# Add edges for each relationship
for _, row in TRRUST_df.iterrows():
    if pd.notna(row['TF_RNA']) and pd.notna(row['TF_PROT']):
        G.add_edge(row['TF_RNA'] + ".r>", row['TF_PROT'] + ".p>")
    if pd.notna(row['TF_PROT']) and pd.notna(row['BS_RNA']):
        G.add_edge(row['TF_PROT'] + ".p>", row['BS_RNA'] + '.r]')
    if pd.notna(row['BS_RNA']) and pd.notna(row['BS_PROT']):
        G.add_edge(row['BS_RNA'] + '.r]', row['BS_PROT'] + ".p]")


# # write TRRUST_df to csv
# TRRUST_df.to_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/Diffusion/TRRUST_adjacency_mat.csv', index=False)
# TRRUST_sample.head()

# Make a 2 column dataframe, first column = set(TF_RNA + BS_RNA), second column = set(TF_PROT + BS_PROT)



# %%
def construct_directed_adj(edges_df, nodes):
    # Initialize a 100x100 matrix with zeros
    adjacency_matrix = pd.DataFrame(0, index=nodes, columns=nodes)

    # Define sets for undirected and directed interactions
    undirected_annotations = {'complex', 'predicted', 'PPrel: binding/association; complex', 'reaction', 'PPrel: dissociation', 'PPrel: binding/association; complex; input'}
    directed_annotations_gene1_to_gene2 = {'catalyze; input', 'catalyze; reaction', 'PPrel: activation', 'PPrel: activation, indirect effect'}
    directed_annotations_gene2_to_gene1 = {'catalyzed by; complex; input', 'catalyzed by', 'activated by; catalyzed by', 'activated by', 'PPrel: activated by; complex; input'}

    # Iterate over the rows in the edges dataframe
    for _, row in edges_df.iterrows():
        gene1, gene2 = row['name'].split(' (FI) ')
        annotation = row['FI Annotation']

        # Check if both genes are in the nodes list
        if gene1 in nodes and gene2 in nodes:
            # Set undirected edge
            if annotation in undirected_annotations:
                adjacency_matrix.loc[gene1, gene2] = 1
                adjacency_matrix.loc[gene2, gene1] = 1
            
            # Set directed edge from gene1 to gene2
            if annotation in directed_annotations_gene1_to_gene2:
                adjacency_matrix.loc[gene1, gene2] = 1
            
            # Set directed edge from gene2 to gene1
            elif annotation in directed_annotations_gene2_to_gene1:
                adjacency_matrix.loc[gene2, gene1] = 1
    
    return adjacency_matrix


# Loading Edges
edges_all_PROTS = pd.read_csv('data/FI_TRRUST_Edges.csv')

# construct RNA adjacency matrix
PPI_interactions = construct_directed_adj(edges_all_PROTS, all_PROTS)

# count all 1s in the adjacency matrix
PPI_interactions.sum().sum()

# PPI_interactions.head()


# %%
def get_protein_role(protein):
    """
    Determine the role of the protein. (Transcription Factor or NOT)
    Args:
    protein (str): The protein name.
    Returns:
    str: The suffix indicating the protein's role.
    """
    # Logic to determine the role
    # For example, using a mapping dictionary or checking specific conditions
    # Return ".p>" for transcription factors, and ".p]" for regulated proteins
    if protein in tf_proteins:
        return ".p>"
    elif protein in targeted_PROTS:
        return ".p]"
    else:
        print(protein)


# INTEGRATING both networks
for gene1 in PPI_interactions.columns:
    for gene2 in PPI_interactions.index:
        # Check if there is an edge from gene1 to gene2
        if PPI_interactions.loc[gene1, gene2] == 1:
            gene1_role = get_protein_role(gene1)
            gene2_role = get_protein_role(gene2)

            # Add directed edge with appropriate suffixes based on roles
            G.add_edge(gene1 + gene1_role, gene2 + gene2_role)


# %% Investigating the Subgraph
def get_connected_subgraph(G, start_node, num_nodes, mode='bfs'):
    if mode == 'bfs':
        # Perform BFS and get the first 'num_nodes' from the list
        nodes = list(nx.bfs_tree(G, start_node))[:num_nodes]
    else:
        # Perform DFS and get the first 'num_nodes' from the list
        nodes = list(nx.dfs_tree(G, start_node))[:num_nodes]

    # Create a subgraph using these nodes
    subgraph = G.subgraph(nodes)
    return subgraph

########################################## SUBGRAPHS and SYNTH GRAPHS #####################################################################
top_node = DEA_TF_PROTS[0] + ".r>"
p = 200  # Define how many nodes you want in the subgraph

# TEsting both directed and undirected subgraphs
subG_dir = get_connected_subgraph(G, top_node, p, mode='dfs')
subG_undir = subG_dir.to_undirected()

# RANDOM GRAPH
random_g = nx.erdos_renyi_graph(p, 0.1, seed=6, directed=False)

# SCALE FREE GRAPH
if p <= 100:
    m = random.choice([1,2,2])
elif 100 < p <= 300:
    m = random.choice([3,5,6])
elif 300 < p <= 500:
    m = random.choice([5,8,10])
elif 500 < p <= 1000:
    m = random.choice([10,15,20])
else:
    m = 20

scale_free_g = nx.barabasi_albert_graph(p, m, seed=42)

def graph_analysis(network):
    # get degree distribution
    degrees = [val for (node, val) in network.degree()]
    plt.hist(degrees, bins=20)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title(fr'Degree Distribution for {len(degrees)} nodes')
    plt.show()

    # plot on a log-log scale
    plt.scatter(np.log10(range(1, len(degrees) + 1)), np.log10(sorted(degrees, reverse=True)))
    plt.xlabel('log10(Rank)')
    plt.ylabel('log10(Degree)')
    plt.title(fr'Log-Log Degree Distribution for {len(degrees)} nodes')
    plt.show()

graph_analysis(scale_free_g)

# %%

# Define node colors based on unique node types
node_colors = []
for node in subG_dir:
    if ".p>" in node:
        node_colors.append('forestgreen')  # Color for 'TF_PROT' nodes
    elif ".p]" in node:
        node_colors.append('palegreen')     # Color for 'BS_PROT' nodes
    elif ".r>" in node:
        node_colors.append('royalblue')    # Color for 'TF_RNA' nodes
    else:  # ".r]" in node
        node_colors.append('skyblue') # Color for 'BS_RNA' nodes

# Draw the undirected subgraph
plt.figure(figsize=(10, 10), dpi=300)
nx.draw_networkx(subG_undir, pos=nx.spring_layout(subG_undir, weight=1),
                 node_size=250,
                 with_labels=True,
                 font_size=5,
                 font_color='grey',
                 edge_color='lightgrey',
                 node_color=node_colors,
                 alpha=0.75)

# Show the plot
plt.show()

# # Now draw the subgraph
# plt.figure(figsize=(10, 10), dpi=300)
# nx.draw_networkx(subG_dir, pos=nx.spring_layout(subG_dir, weight=1, scale=1), 
#                  node_size=250, 
#                  with_labels=False,
#                  edge_color='lightgrey',
#                  node_color=node_colors,  # Adjust if node_colors needs to be recalculated for subG
#                  arrows=True,
#                  arrowsize=25, 
#                  alpha=0.75)

# plt.show()


# # plot random_g
# plt.figure(figsize=(10, 10), dpi=300)
# nx.draw_networkx(random_g, pos=nx.spring_layout(random_g, weight=1, scale=1), 
#                  node_size=250, 
#                  with_labels=True,
#                  edge_color='lightgrey',
#                  arrowsize=25, 
#                  alpha=0.75)

# plt.show()

################################## DIFFUSION ##############################################################################

# %%
import random

gchoice = subG_undir # random_g
node_choice = list(subG_undir.nodes())
print(node_choice)

# A very small sigma2 value
sigma2 = 80

# A small add_diag value
add_diag = 1

lap_kernel = regularised_laplacian_kernel(gchoice, sigma2=sigma2, add_diag=add_diag, normalized=True)
diff_kernel = diffusion_kernel(gchoice, sigma2=sigma2, normalized=True)

# Preparing the nodes of subG
network_nodes = list(gchoice.nodes())
label_values = np.array([1 if node in node_choice else 0 for node in network_nodes])

input_matrix = Matrix(mat=label_values, rows_labels=network_nodes, cols_labels=['score'])


# Perform the diffusion
diffusion_results = diffuse_raw(graph=None, scores=input_matrix, k=diff_kernel)

# Convert the Matrix object to a dictionary of node scores
diffusion_scores_dict = {node: score for node, score in zip(diffusion_results.rows_labels, diffusion_results.mat.flatten())}

# Sum over the diffusion scores to check the total 'heat'
total_heat = sum(diffusion_scores_dict.values())
# plot distribution of scores
plt.hist(diffusion_scores_dict.values(), bins=20)
plt.xlabel('Diffusion Score')
plt.ylabel('Frequency')
plt.title(fr'Diffusion Scores for {len(diffusion_scores_dict)} nodes, $\sigma^2$ = {sigma2}, add_diag = {add_diag}')
plt.show()

# print(diffusion_scores_dict)
# print(f"Total diffusion score: {total_heat}")

# Normalize the diffusion scores for coloring
max_score = max(diffusion_scores_dict.values())
min_score = min(diffusion_scores_dict.values())
norm = plt.Normalize(vmin=min_score, vmax=max_score)
cmap = plt.cm.coolwarm

node_colors = [cmap(norm(diffusion_scores_dict[str(node)])) for node in gchoice.nodes()]

# pos = nx.spring_layout(gchoice)  # or any other layout algorithm
# nx.draw_networkx(gchoice, 
#                  pos, 
#                  node_color=node_colors, 
#                  with_labels=True,
#                  font_size=5,
#                  arrows=True, 
#                  arrowsize=1,
#                  alpha=0.8)
# plt.show()

# Draw the undirected subgraph
plt.figure(figsize=(10, 10), dpi=300)
nx.draw_networkx(gchoice, pos=nx.spring_layout(gchoice, weight=1),
                 node_size=250,
                 with_labels=True,
                 font_size=5,
                 font_color='grey',
                 edge_color='lightgrey',
                 node_color=node_colors,
                 alpha=0.75)

# Show the plot
plt.show()

# %%
