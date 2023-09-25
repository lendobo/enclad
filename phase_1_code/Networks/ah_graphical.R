# install.packages("fastnet", repos = "http://cran.r-project.org")
library(fastnet)
# install.packages("huge", repos = "http://cran.r-project.org")
library(huge)
library(igraph)

# Set the number of nodes
n_nodes <- 300

# Simulate a scale-free network using the net.barabasi.albert function
sf_network <- net.barabasi.albert(n = n_nodes, m = 15)

# print first 10 entries of sf_network
print(sf_network[1:10])


# Initialize a variable to keep track of the total number of connections
total_connections <- 0
# Iterate through each element of the sf_network list
for (node in sf_network) {
    # Add the number of connections for the current node to the total
    total_connections <- total_connections + length(node)
}
# Each edge is counted twice, so divide by 2 to get the actual number of edges
n_edges <- total_connections / 2
# Now calculate the density
density <- (2 * n_edges) / (n_nodes * (n_nodes - 1))

# Print the density
print(density)


############# Using huge to calculate the true precision matrix #############
# Convert the simulated network to an adjacency matrix
# Here, we'll create a zero matrix and fill in the connections based on the list structure of sf_network
adj_matrix <- matrix(0, nrow = n_nodes, ncol = n_nodes)
for (i in 1:length(sf_network)) {
    for (j in sf_network[[i]]) {
        adj_matrix[i, j] <- 1
        adj_matrix[j, i] <- 1 # since the network is undirected
    }
}

# # export to csv
# write.csv(adj_matrix, file = "adj_matrix.csv")

# Use the huge function to estimate the precision matrix and other parameters
huge_result <- huge(adj_matrix, method = "glasso")

# The estimated precision matrix is stored in huge_result$opt.icov
precision_matrix <- huge_result$icov

str(huge_result$icov)

# To obtain the partial correlation matrix, invert the precision matrix
partial_correlation_matrix <- solve(precision_matrix)

# Print the partial correlation matrix
print(partial_correlation_matrix)
