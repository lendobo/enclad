# Libraries
library(ggraph)
library(igraph)
library(tidyverse)
library(RColorBrewer)

# create a data frame giving the hierarchical structure of your individuals
set.seed(1234)
d1 <- data.frame(from = "origin", to = paste("group", seq(1, 6), sep = ""))

# specify the subgroups for each group
subgroups <- list(
  c("Mutual Information", "Correlation-based"),
  c("SVD", "Spectral", "Other"),
  c("Random Walk", "Heat Diffusion", "Graph Kernels"),
  c("RecGNN", "GCN", "GAE", "STGNN"),
  c("Metapath1", "Metapath2"),
  c("BNfinder", "BANJO", "Nemhauser")
)

# create an empty dataframe
d2 <- data.frame(from = character(), to = character(), stringsAsFactors = FALSE)

# populate the dataframe
group_names <- paste("group", seq(1, length(subgroups)), sep = "")
for (i in seq_along(subgroups)) {
  group_subgroups <- subgroups[[i]]
  d2 <- rbind(d2, data.frame(from = group_names[i], to = group_subgroups, stringsAsFactors = FALSE))
}

edges <- rbind(d1, d2)

# create a dataframe with connection between leaves (individuals)
all_leaves <- d2$to
connect <- rbind(
  data.frame(from = sample(all_leaves, length(all_leaves), replace = T), to = sample(all_leaves, length(all_leaves), replace = T)),
  data.frame(from = sample(head(all_leaves), 15, replace = T), to = sample(tail(all_leaves), 15, replace = T)),
  data.frame(from = sample(all_leaves[5:7], 15, replace = T), to = sample(all_leaves[15:17], 15, replace = T)),
  data.frame(from = sample(all_leaves[18:20], 15, replace = T), to = sample(all_leaves[10:12], 15, replace = T))
)
connect$value <- runif(nrow(connect))


# create a vertices data.frame. One line per object of our hierarchy
vertices <- data.frame(
  name = unique(c(as.character(edges$from), as.character(edges$to))),
  value = runif(length(unique(c(as.character(edges$from), as.character(edges$to)))))
)

# Let's add a column with the group of each name. It will be useful later to color points
vertices$group <- edges$from[match(vertices$name, edges$to)]

################### LABEL PARAMETERS 1 ##############################################################
# Let's add information concerning the label we are going to add: angle, horizontal adjustement and potential flip
# calculate the ANGLE of the labels
vertices$id <- NA
myleaves <- which(is.na(match(vertices$name, edges$from)))
nleaves <- length(myleaves)

# Assign id based on the order of appearance in the layout
layout <- as_data_frame(layout_as_tree(graph_from_data_frame(edges, vertices = vertices)))
vertices <- vertices %>% arrange(match(name, layout$name))

vertices$id[myleaves] <- seq(1:nleaves)
vertices$angle <- 90 - 360 * vertices$id / nleaves
vertices$hjust <- ifelse(vertices$angle < -90, 1, 0)
vertices$angle <- ifelse(vertices$angle < -90, vertices$angle + 180, vertices$angle)


#################### PLOT ########################################################
# Create a graph object
mygraph <- igraph::graph_from_data_frame(edges, vertices = vertices)

# The connection object must refer to the ids of the leaves:
from <- match(connect$from, vertices$name)
to <- match(connect$to, vertices$name)

ggraph(mygraph, layout = "dendrogram", circular = TRUE) +
  geom_conn_bundle(data = get_con(from = from, to = to), alpha = 0.2, width = 0.9, aes(colour = ..index..)) +
  scale_edge_colour_distiller(palette = "RdPu") +
  geom_node_text(aes(x = x * 1.15, y = y * 1.15, filter = leaf, label = name, angle = angle, hjust = hjust, colour = group), size = 2, alpha = 1) +
  geom_node_point(aes(filter = leaf, x = x * 1.07, y = y * 1.07, colour = group, size = value, alpha = 0.2)) +
  scale_colour_manual(values = rep(brewer.pal(9, "Dark2"), 30)) +
  scale_size_continuous(range = c(0.1, 10)) +
  theme_void() +
  theme(
    legend.position = "none",
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  expand_limits(x = c(-1.3, 1.3), y = c(-1.3, 1.3))
