# %%
# Importing the required packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Increase fontsize
plt.rcParams.update({"font.size": 16})

# Generating a smaller mock data matrix with 10 genes and 10 samples
genes_small = [f"Gene_{i}" for i in range(1, 11)]
samples_small = [f"Sample_{j}" for j in range(1, 11)]
data_matrix_small = np.random.rand(10, 10)

# Creating a minimalistic heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(
    data_matrix_small,
    cmap="viridis",
    yticklabels=genes_small,
    xticklabels=samples_small,
    cbar=False,
)
plt.title("Patient-Specific Heatmap")
plt.ylabel("Genes")
plt.xlabel("Patients")
plt.tight_layout()
plt.show()

# %%
# Generate correlation data for Gene 1 and Gene 2
gene1 = np.random.rand(1)
gene2 = np.random.rand()

# Plot the data using seaborn
plt.figure(figsize=(8, 8))
sns.scatterplot(x=gene1, y=gene2, color="blue")
plt.xlabel("Gene 4")
plt.ylabel("Gene 5")
plt.title("Correlation between two genes")
plt.show()

# %%
### GRAPH ###
# Generate random network with 10 nodes, using networkx
import networkx as nx

# Create a random network with 10 nodes
G = nx.gnm_random_graph(10, 20)

# Visualize the network
plt.figure(figsize=(8, 8))
nx.draw(G, with_labels=True, node_color="skyblue", node_size=800, alpha=0.8)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generating mock data matrix with 10 genes and 2 columns (1 for genes and 1 for proteins)
gene_labels = [f"Gene_{i}" for i in range(1, 11)]
protein_labels = [f"Protein_{i}" for i in range(1, 11)]
data_matrix_two_columns = np.random.rand(10, 2)

# Creating the heatmap with yticklabels on both the left and right
fig, ax = plt.subplots(figsize=(5, 8))

sns.heatmap(
    data_matrix_two_columns, cmap="Greys", yticklabels=gene_labels, cbar=False, ax=ax
)
ax.set_yticklabels(gene_labels, va="center")
ax.set_xticklabels(["Genes", "Proteins"], rotation=0)

# Adding protein labels on the right
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(ax.get_yticks())
ax2.set_yticklabels(protein_labels, va="center")
ax2.yaxis.tick_right()

plt.tight_layout()
plt.title("Patient-Specific Gene-Protein Heatmap")
plt.show()


# %%
# Define gene and protein labels
gene_labels = [f"Gene_{i}" for i in range(1, 11)]
protein_labels = [f"Protein_{i}" for i in range(1, 11)]

# Create a new bipartite graph
B_simple = nx.Graph()

# Add nodes with the bipartite attribute
B_simple.add_nodes_from(gene_labels, bipartite=0)
B_simple.add_nodes_from(protein_labels, bipartite=1)

# Connect each gene to its corresponding protein
for i in range(10):
    B_simple.add_edge(gene_labels[i], protein_labels[i])
# Directly defining positions for nodes without relying on nx.bipartite.sets
pos_simple = {}
for index, gene in enumerate(gene_labels):
    pos_simple[gene] = (1, index)
for index, protein in enumerate(protein_labels):
    pos_simple[protein] = (2, index)

# Visualizing the graph
plt.figure(figsize=(10, 12))
nx.draw(B_simple, pos=pos_simple, with_labels=True, 
        node_color=["skyblue" if node in gene_labels else "lightcoral" for node in B_simple.nodes()],
        node_size=1500, font_size=12, font_weight='bold')
plt.title("Bipartite Graph of Genes and Proteins")
plt.axis("off")
plt.show()

