from pymnet import *
import networkx as nx
import random
import matplotlib.pyplot as plt

# Creating sample networkx graphs for each layer
G1 = nx.erdos_renyi_graph(n=20, p=0.2)  # First layer graph
G2 = nx.erdos_renyi_graph(n=20, p=0.2)  # Second layer graph


# Create a Multiplex Network with ordinal coupling
net = MultiplexNetwork(couplings='ordinal', noEdge=0, directed=False)

# Add nodes to the network
num_nodes_per_layer = 20
for i in range(num_nodes_per_layer):
    net.add_node(i)

# Add layers to the network
net.add_layer(0)  # First layer
net.add_layer(1)  # Second layer


# Assume net is your already created MultiplexNetwork object
for node in G1.nodes():
    net.add_node(node, layer=0)

for edge in G1.edges():
    net[edge[0], edge[1], 0] = 1  # Adds edge in the first layer

for node in G2.nodes():
    net.add_node(node, layer=1)

for edge in G2.edges():
    net[edge[0], edge[1], 1] = 1  # Adds edge in the second layer


# COloring edges and nodes
# Assign a random value (e.g., between 0 and 1) to each edge in each layer
for u, v in G1.edges():
    net[u, v, 0] = random.random()  # Assigning to layer 0
for u, v in G2.edges():
    net[u, v, 1] = random.random()  # Assigning to layer 1
# Initialize node values
node_values = {i: 0 for i in range(num_nodes_per_layer)}
# Calculate the sum of edge values for each node
for u, v in G1.edges():
    node_values[u] += net[u, v, 0]
    node_values[v] += net[u, v, 0]
for u, v in G2.edges():
    node_values[u] += net[u, v, 1]
    node_values[v] += net[u, v, 1]

# Create a colormap
cmap = plt.cm.viridis  # or any other colormap

# Normalize node values to [0, 1] for the colormap
min_val, max_val = min(node_values.values()), max(node_values.values())
node_colors = {node: cmap((val - min_val) / (max_val - min_val)) for node, val in node_values.items()}

# Similarly, set colors for edges based on their assigned values (using first layer as example)
edge_colors_layer_0 = {(u, v, 0): cmap(net[u, v, 0]) for u, v in G1.edges()}
edge_colors_layer_1 = {(u, v, 1): cmap(net[u, v, 1]) for u, v in G2.edges()}

# Combine edge color dictionaries
combined_edge_colors = {**edge_colors_layer_0, **edge_colors_layer_1}


# Define layer colors for visual distinction
layer_colors = {0: "red", 1: "green"}
# Define layer alpha (transparency) settings
# layer_alpha_values = {0: 0.5, 1: 0.5}  # 50% transparency for both layers


# Draw the network with specified parameters
draw(net,
     layout="spring",  # Layout of the nodes
     layershape="rectangle",  # Shape of the layer
    #  layerAlphaDict=layer_alpha_values,  # Setting specific layer alpha values
     defaultLayerAlpha=0.25,  # Default alpha value for any layers not specified in layerAlphaDict
     azim=-51,  # Azimuthal viewing angle
     elev=22,  # Elevation viewing angle
     layerColorDict=layer_colors,  # Dictionary for layer colors
    #  defaultNodeColor="black",  # Default color for nodes
     defaultEdgeColor="gray",  # Default color for edges
     nodeColorDict=node_colors,
    #  edgeColorDict=combined_edge_colors,
    #  edgeColorRule={"rule": "edgeweight", "colormap": "viridis", "scaleby": 1.0},
     figsize=(10, 10),  # Size of the figure
     autoscale=True,  # Whether to autoscale the figure
     layergap=1.0,  # Gap between layers
     alignedNodes=True,  # Align nodes across layers
     show=True)  # Whether to show the figure)