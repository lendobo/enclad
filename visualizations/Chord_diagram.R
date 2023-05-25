# install circlize library
# install.packages("circlize")

# load the library
library(circlize)

# define matrix
mat = matrix(c(0,1,1,0,0,0,
               1,0,1,1,0,0,
               1,1,0,0,0,0,
               0,1,0,0,1,1,
               0,0,1,1,0,0,
               0,0,0,1,0,0), byrow = TRUE, nrow = 6)

# label your items
rownames(mat) = colnames(mat) = c("Graph ML", "Graph Convolution",
    "Graph Kernels", "Network Propagation", "Random Walk", "Heat Diffusion")

# set colors
grid.col = c("Graph ML" = "#328038",
     "Graph Convolution" = "#3fa5ff", "Graph Kernels" = "#cacaca",
     "Network Propagation" = "#C90000", "Random Walk" = "#d7d300",
     "Heat Diffusion" = "skyblue")

# create chord diagram
chordDiagram(mat, grid.col = grid.col, transparency = 0.5)

# # add legend
# legend(x = "bottom", legend = c("Graph ML", "Graph Convolution",
#      "Graph Kernels", "Network Propagation", "Random Walk", "Heat Diffusion"),
#        fill = c("darkorange", "darkorange", "darkorange", "darkorange",
#         "darkorange", "skyblue"), cex = 0.8, ncol = 3, 
#        box.col = "white", box.lwd = 0.5, box.lty = 1, title = "Methods", title.col = "black", title.cex = 0.8, 
#        title.adj = 0, horiz = TRUE, inset = c(0.1, 0))

