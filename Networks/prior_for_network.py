# %%
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (MONIKA)
project_dir = os.path.dirname(script_dir)

# Add the project directory to the Python path
sys.path.append(project_dir)

# Change the working directory to the project directory
os.chdir(project_dir)

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from diffupy.diffuse import run_diffusion_algorithm
from diffupy.matrix import Matrix
from diffupy.diffuse_raw import diffuse_raw
from diffupy.kernels import regularised_laplacian_kernel, diffusion_kernel
import random

# %%
# switch matplotlib background to dark mode
plt.style.use('dark_background')
