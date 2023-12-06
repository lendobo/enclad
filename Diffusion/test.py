# Consolidated code block to create the requested network

import matplotlib.pyplot as plt
import networkx as nx

def create_and_draw_network():
    # Create a new graph
    G = nx.Graph()
    # Define nodes
    nodes_A = [f"{i}.p" for i in range(1, 6)]
    nodes_B = [f"{i}.t" for i in range(1, 6)]
    # Add nodes
    G.add_nodes_from(nodes_A)
    G.add_nodes_from(nodes_B)
    # Add edges to form a complete circle in each layer
    for i in range(5):
        G.add_edge(nodes_A[i], nodes_A[(i-1)%5])  # A layer
        G.add_edge(nodes_B[i], nodes_B[(i-1)%5])  # B layer
    # Add edges between corresponding nodes of different layers
    for i in range(5):
        G.add_edge(nodes_A[i], nodes_B[i])
    # Add manual edges
    G.add_edge("1.t", "3.t")
    G.add_edge("1.p", "4.p")



    # Initial positions for the nodes
    pos = {node:(i, 1) for i, node in enumerate(nodes_A)}
    pos.update({node:(i, 0) for i, node in enumerate(nodes_B)})
    # Shift odd nodes up
    def shift_odd_nodes_up(positions, shift_amount=0.2):
        new_positions = positions.copy()
        for node in positions:
            if int(node[0]) % 2 != 0:  # Check if the node number is odd
                x, y = positions[node]
                new_positions[node] = (x, y + shift_amount)
            if int(node[0]) == 3:
                x, y = positions[node]
                new_positions[node] = (x, y + shift_amount-0.35)
        return new_positions

    shifted_pos = shift_odd_nodes_up(pos)

    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(G, shifted_pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=12)
    plt.title("Custom Network with Shifted Odd Nodes")
    plt.show()

    return G

# Call the function to create and draw the network
# create_and_draw_network()

# get degrees
G = create_and_draw_network()
degrees = [val for (node, val) in G.degree()]
print(degrees)
