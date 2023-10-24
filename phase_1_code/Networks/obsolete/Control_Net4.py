import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.linalg import expm, pinv
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage


def update(num):
    ax.clear()
    nx.draw(G, pos, with_labels=True, node_color='skyblue', font_weight='bold', node_size=700, font_size=18, ax=ax)
    
    # Annotate nodes with their states
    for i, (x, y) in pos.items():
        ax.text(x, y, f"{states[num, i]:.2f}", fontsize=12, ha='right', va='bottom')
    
    ax.set_title(f"Time = {times[num]:.1f}")


# Define the adjacency matrix A
A = np.array([[0, 1, 0, 0],
              [1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0]])

# Define the control matrix B
B = np.array([[1, 0],
              [0, 0],
              [0, 0],
              [0, 1]])

# Define initial and final states
x0 = np.array([0, 0, 0, 0])
xf = np.array([1, 0, 0, 0])

# Define the time T for transition
T = 1

# Calculate the matrix exponential of AT
exp_AT = expm(A * T)

# Calculate the pseudo-inverse of B
B_pseudo_inv = pinv(B)

# Calculate the control signal u(t)
u_t = B_pseudo_inv @ (xf - exp_AT @ x0)
u_t


##################### 8 nodes #############################
# Define the network using NetworkX
G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)

# Define parameters for the simulation
delta_t = 0.1  # time step size
time_steps = int(T / delta_t)  # number of time steps

# Initialize lists to store time evolution
states = [x0]
times = [0]

# Discrete-time simulation
for t in range(1, time_steps + 1):
    x_prev = states[-1]
    x_next = np.dot(expm(A * delta_t), x_prev) + np.dot(B, u_t) * delta_t
    states.append(x_next)
    times.append(round(t * delta_t, 1))


# Convert states to numpy array for easier indexing
states = np.array(states)

# # Create the animation
# fig, ax = plt.subplots(figsize=(6, 6))
# pos = nx.spring_layout(G, seed=42)  # positions for all nodes

# ani = FuncAnimation(fig, update, frames=range(time_steps+1), repeat=False)
# plt.show()

# Compute the pairwise Euclidean distance between states
distance_matrix = squareform(pdist(states, metric='euclidean'))

# Display a portion of the distance matrix
distance_matrix[:5, :5]

# Perform hierarchical clustering
Z = linkage(squareform(distance_matrix), method='complete')

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=times, leaf_rotation=45)
plt.xlabel('Time Steps', fontsize=16)
plt.ylabel('Euclidean Distance', fontsize=16)
plt.title('Hierarchical Clustering of Network States', fontsize=20)
plt.show()

############################### 8 NODES #####################################
# Generate a random adjacency matrix for 8-node network
np.random.seed(42)
A = np.random.rand(8, 8)

# Normalize the matrix to keep values between 0 and 1
A = A / np.max(A)

# Number of nodes
n = A.shape[0]

# Initialize minimum control nodes and best combination
min_control_nodes = n
best_combination = []

# Loop over all possible combinations of nodes
for num_nodes in range(1, n + 1):
    for nodes in combinations(range(n), num_nodes):
        B_temp = np.zeros((n, len(nodes)))
        
        # Populate the control matrix for the current combination of nodes
        for i, node in enumerate(nodes):
            B_temp[node, i] = 1
        
        # Calculate the controllability matrix
        C = B_temp
        for i in range(1, n):
            C = np.hstack((C, np.linalg.matrix_power(A, i) @ B_temp))
        
        # Check if the system is controllable
        if np.linalg.matrix_rank(C) == n:
            if num_nodes < min_control_nodes:
                min_control_nodes = num_nodes
                best_combination = nodes
            break

# Define initial and final states
x0 = np.zeros(8)
xf = np.ones(8)

# Time for transition
T = 1.0

# Control matrix B for best combination of control nodes
B_best = np.zeros((n, len(best_combination)))
for i, node in enumerate(best_combination):
    B_best[node, i] = 1

# Calculate control signal u(t)
exp_AT = expm(A * T)
B_pseudo_inv = pinv(B_best)
u_t = B_pseudo_inv @ (xf - exp_AT @ x0)

# Time steps for simulation
delta_t = 0.1
time_steps = int(T / delta_t)

# Initialize states
states = [x0]

# Discrete-time simulation to validate
for t in range(1, time_steps + 1):
    x_prev = states[-1]
    x_next = np.dot(expm(A * delta_t), x_prev) + np.dot(B_best, u_t) * delta_t
    states.append(x_next)

# Convert states to numpy array for easier indexing
states = np.array(states)

# Create the animation
fig, ax = plt.subplots(figsize=(6, 6))
G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
pos = nx.spring_layout(G, seed=42)  # positions for all nodes

def update(num):
    ax.clear()
    nx.draw(G, pos, with_labels=True, node_color='skyblue', font_weight='bold', node_size=700, font_size=18, ax=ax)
    
    # Annotate nodes with their states
    for i, (x, y) in pos.items():
        ax.text(x, y, f"{states[num, i]:.2f}", fontsize=12, ha='right', va='bottom')
    
    ax.set_title(f"Time = {num * delta_t:.1f}")

ani = FuncAnimation(fig, update, frames=range(time_steps+1), repeat=False)

# Create a writer object for saving GIF
writer = PillowWriter(fps=1)
ani.save('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/ENCLAD/results/network_control_transition8.gif', writer=writer)

# Show the animation inline (for demonstration)
plt.close(fig)  # Close the plot to avoid double display
ani = FuncAnimation(fig, update, frames=range(time_steps+1), repeat=False)
plt.show()

