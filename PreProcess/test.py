import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = 'data/LinkedOmics/linked_rna.cct'

# Load the data as pandas dataframe
data = pd.read_csv(filename, sep='\t')

print(data.head())