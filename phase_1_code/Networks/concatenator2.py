import numpy as np
import matplotlib.pyplot as plt
import pickle

from itertools import combinations

n = 5
b = int(0.75 * n)

all_combinations = list(combinations(range(n), b))

print(len(all_combinations))