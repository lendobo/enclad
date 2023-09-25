# %%
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import random

# %%
# Function to generate a Barabási–Albert network with node labels
def generate_BA_network(n, m, labels):
    G = nx.barabasi_albert_graph(n, m, seed=29)
    label_mapping = {}
    for i, label in enumerate(labels):
        label_mapping[i] = label
    G = nx.relabel_nodes(G, label_mapping)
    return G

# Function to detect communities using the Girvan-Newman algorithm in NetworkX
def detect_communities_girvan_newman(G):
    comp = community.girvan_newman(G)
    first_level_communities = next(comp)
    partition = {}
    for idx, subgraph in enumerate(first_level_communities):
        for node in subgraph:
            partition[node] = idx
    return partition

# Function to check how well the biological markers (WNT, MYC, Stromal, TGFbeta) are clustered in communities
def check_marker_communities(partition, marker_prefixes):
    marker_communities = {}
    for prefix in marker_prefixes:
        marker_communities[prefix] = [comm for node, comm in partition.items() if prefix in node]
    return marker_communities
    
# Modified Function to label nodes in selected communities and update the partition dictionary
def label_marker_communities(G, partition, selected_communities, marker_labels):
    label_mapping = {}
    new_partition = {}
    for comm, marker in zip(selected_communities, marker_labels):
        nodes_in_community = [node for node, community in partition.items() if community == comm]
        for node in nodes_in_community:
            new_label = f"{marker}_{node}"
            label_mapping[node] = new_label
            new_partition[new_label] = comm
    G = nx.relabel_nodes(G, label_mapping)
    partition.update(new_partition)
    return G, partition

# Function to label edges with rate constraints ('kinase' or 'complex') and associated weights
def label_edges(G):
    # Randomly select 5 nodes from each network
    selected_nodes = random.sample(G.nodes(), 5)
    for u, v, data in G.edges(data=True):
        if u in selected_nodes or v in selected_nodes:
            data['label'] = 'complex'
            data['weight'] = 0.8
        else:
            data['label'] = 'activate'
            data['weight'] = 0.4
    return G

# Function to visualize networks with edge and node colors based on their types and communities
def visualize_network_with_edge_color(G, partition, title):
    edge_colors = ['#ebc634' if data['label'] == 'complex' else '#c887fa' for u, v, data in G.edges(data=True)]
    node_colors = [partition[node] for node in G.nodes()]
    pos = nx.spring_layout(G)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.5)
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.rainbow, node_size=50)
    
    # Add legends
    edge_legend = {'Complex (0.8)': '#ebc634', 'Kinase (0.4)': '#c887fa'}
    node_legend = {f"WNT": 'salmon', 'Other': 'skyblue'}
    
    for label, color in edge_legend.items():
        plt.plot([0], [0], color=color, label=label)
        
    for label, color in node_legend.items():
        plt.scatter([], [], c=[color], s=50, label=label)
    
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Node/Edge types')
    plt.title(title)
    plt.axis('off')


# %%

# model parameters
num_nodes = 50
m = 2

# Generate Barabási–Albert networks for CMS2 and CMS4 with generic labels first
G_CMS2 = generate_BA_network(num_nodes, m, [f"Node_{i}" for i in range(num_nodes)])
G_CMS4 = generate_BA_network(num_nodes, m, [f"Node_{i}" for i in range(num_nodes)])

# Label edges with rate constraints ('kinase' or 'complex') and associated weights
G_CMS2 = label_edges(G_CMS2)
G_CMS4 = label_edges(G_CMS4)

# Detect communities in the networks
partition_CMS2 = detect_communities_girvan_newman(G_CMS2)
partition_CMS4 = detect_communities_girvan_newman(G_CMS4)

# print(partition_CMS2)

# Choose two communities from each graph to represent biological markers
selected_communities_CMS2 = [0, 1]
selected_communities_CMS4 = [0, 1]

# Label nodes in these communities and update partitions
G_CMS2, partition_CMS2 = label_marker_communities(G_CMS2, partition_CMS2, selected_communities_CMS2, ['MYC', 'WNT'])
G_CMS4, partition_CMS4 = label_marker_communities(G_CMS4, partition_CMS4, selected_communities_CMS4, ['TGFbeta', 'Angiogenesis'])

# # print node degrees
# print("Node degrees in CMS2 network:")
# print(G_CMS2.degree())

# %%
# KEPT IN FOR NOW TO SHOW WHICH FUNCTIONS ABOVE CAN BE DELETED
# Visualize the newly labeled networks
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# nx.draw(G_CMS2, with_labels=False, node_size=50, node_color=[partition_CMS2.get(node, -1) for node in G_CMS2.nodes()], cmap=plt.cm.rainbow)
# plt.title('CMS2 Network with Marker Communities')

# plt.subplot(1, 2, 2)
# nx.draw(G_CMS4, with_labels=False, node_size=50, node_color=[partition_CMS4.get(node, -1) for node in G_CMS4.nodes()], cmap=plt.cm.rainbow)
# plt.title('CMS4 Network with Marker Communities')

# plt.show()

# %%
# Visualize the networks with edge colors
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
visualize_network_with_edge_color(G_CMS2, partition_CMS2, "CMS2 Network")

plt.subplot(1, 2, 2)
visualize_network_with_edge_color(G_CMS4, partition_CMS4, "CMS4 Network")

plt.show()


# %%
