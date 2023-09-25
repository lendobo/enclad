import networkx as nx
import matplotlib.pyplot as plt
import random

# Function to visualize the state of the network with alpha values
def visualize_continuous_state(G, pos, state, title):
    node_colors = []
    node_alphas = []
    for node in G.nodes():
        if state[node] > 0.5:
            node_colors.append('lightgreen')
        else:
            node_colors.append('salmon')
        node_alphas.append(state[node])
    nx.draw(G, pos, with_labels=True, node_color=node_colors, alpha=0.8, node_size=[state[node]*800 for node in G.nodes()], arrows=True)
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='black')
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha)
    plt.title(title)
    plt.show()

# Generate a Barabási–Albert network with 20 nodes
n = 20  # Number of nodes
m = 2  # Number of edges to attach from a new node to existing nodes

G = nx.barabasi_albert_graph(n, m)

# Label 3 connected nodes as 'WNT Pathway'
wnt_nodes = [0, 1, 2]  # These nodes are guaranteed to be connected in a Barabási–Albert graph when starting from 0
for node in wnt_nodes:
    G.nodes[node]['pathway'] = 'WNT'

# Label 3 different connected nodes as 'TGFβ'
tgfb_nodes = [3, 4, 5]  # These nodes are also guaranteed to be connected when starting from 3
for node in tgfb_nodes:
    G.nodes[node]['pathway'] = 'TGFβ'

# Convert the network to a directed graph
G_directed = G.to_directed()

# Randomly remove one direction from each pair of reciprocal edges to introduce random directionality
for edge in list(G.edges()):
    if (edge[1], edge[0]) in G_directed.edges():
        if random.choice([True, False]):
            G_directed.remove_edge(*edge)
        else:
            G_directed.remove_edge(edge[1], edge[0])

# Initialize continuous states
state_A_continuous = {}
state_B_continuous = {}

# Assign states considering the WNT and TGFβ pathways
for node in G_directed.nodes():
    if node in wnt_nodes:
        state_A_continuous[node] = 1.0
        state_B_continuous[node] = 0.0
    elif node in tgfb_nodes:
        state_A_continuous[node] = 0.0
        state_B_continuous[node] = 1.0
    else:
        state_A_continuous[node] = random.uniform(0, 1)
        state_B_continuous[node] = random.uniform(0, 1)

# Visualize the network
# alpha = 0.5
edge_alpha =0.1
pos = nx.spring_layout(G)
labels = {node: G.nodes[node].get('pathway', '') for node in G.nodes()}

# Visualize State A with continuous values
visualize_continuous_state(G_directed, pos, state_A_continuous, 'Continuous State A')

# Visualize State B with continuous values
visualize_continuous_state(G_directed, pos, state_B_continuous, 'Continuous State B')
