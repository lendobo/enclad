# %%
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Use dark plots
plt.style.use("dark_background")


reactome_file_path = "/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/Codebase2/data/KnowledgeGraphDBs/Giant_component_FIVIZ.csv"

# # Attempt to read the Reactome network data
# # read xlsx file
# try:
#     reactome_data = pd.read_excel(reactome_file_path)
#     reactome_data_head = reactome_data.head()
# except Exception as e:
#     reactome_data_head = str(e)

try:
    reactome_data = pd.read_csv(reactome_file_path)
    reactome_data_head = reactome_data.head()
except Exception as e:
    reactome_data_head = str(e)

reactome_data_head

# %%

# Split the 'name' column into 'source' and 'target' nodes
reactome_data[["source", "target"]] = reactome_data["name"].str.split(
    " \(FI\) ", expand=True
)

# Create a NetworkX graph from the DataFrame
G = nx.from_pandas_edgelist(
    reactome_data,
    "source",
    "target",
    ["EDGE_TYPE", "FI Annotation", "FI Direction", "FI Score"],
)

# Show some basic information about the graph
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
# info = nx.info(G)

num_nodes, num_edges

# %%

# Calculate the degree for each node in the original graph
degree_dict = dict(G.degree())
print(degree_dict)

# Calculate the degree distribution of the original graph
degree_values = list(degree_dict.values())
degree_counts = np.bincount(degree_values)
degree_prob = degree_counts / np.sum(degree_counts)

# Create a dictionary to store nodes by their degree
nodes_by_degree = {}
for node, degree in degree_dict.items():
    if degree not in nodes_by_degree:
        nodes_by_degree[degree] = []
    nodes_by_degree[degree].append(node)

# print(nodes_by_degree)
# print(degree_prob)

# Subsample nodes based on their degree to preserve the degree distribution
selected_nodes = []
for degree, prob in enumerate(degree_prob):
    if prob > 0 and degree in nodes_by_degree:
        num_to_select = int(np.ceil(len(nodes_by_degree[degree]) * prob))
        selected_nodes.extend(
            np.random.choice(nodes_by_degree[degree], num_to_select, replace=False)
        )

# Create a subgraph with the selected nodes
G_subsampl = G.subgraph(selected_nodes)

# Show some basic information about the subsampled graph
G_subsamp_nodenum = G_subsampl.number_of_nodes()
G_subsampl_edgenum = G_subsampl.number_of_edges()

G_subsamp_nodenum, G_subsampl_edgenum

# %%

# Visualize degree distribtion of subsampled graph
subsampl_degrees = dict(G_subsampl.degree())
subsampl_degree_vals = list(subsampl_degrees.values())
sumbsampl_degree_counts = np.bincount(subsampl_degree_vals)
degree_prob_subsampled_degree_preserve = sumbsampl_degree_counts / np.sum(
    sumbsampl_degree_counts
)


# Visualize both in a subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.plot(degree_prob, "o")
ax1.set_xlabel("Degree")
ax1.set_ylabel("Probability")
ax1.set_title("Degree Distribution of Original Graph")
ax2.plot(degree_prob_subsampled_degree_preserve, "o")
ax2.set_xlabel("Degree")
ax2.set_ylabel("Probability")
ax2.set_title("Degree Distribution of Subsampled Graph")
plt.show()

# export subsampled node names
subsampled_node_names = list(G_subsampl.nodes())
subsampled_node_names_df = pd.DataFrame(subsampled_node_names)
subsampled_node_names_df.to_csv(
    "/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/Codebase2/data/KnowledgeGraphDBs/PPI_subsampled_node_names.csv",
    index=False,
)


# %%
# Visualize original graph
plt.figure(figsize=(16, 12))
nx.draw_networkx(G, with_labels=False, node_size=10, width=0.1)
plt.show()

# %%
