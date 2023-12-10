import random
import numpy as np
import networkx as nx

density = 0.1
p = 626

density_params = {
        0.02: [(100, 1), (300, 3), (500, 5), (750, 8), (1000, 10)],
        0.03: [(100, 2), (300, 5), (500, 8), (750, 11), (1000, 15)],
        0.04: [(100, 2), (300, 6), (500, 10), (750, 15), (1000, 20)],
        0.1: [(100, 5), (300, 15), (500, 25), (750, 38), (1000, 50)],
        0.2: [(100, 10), (300, 30), (500, 50), (750, 75), (1000, 100)]
    }

# Determine m based on p and the desired density
m = 20  # Default value if p > 1000
closest_distance = float('inf')
for size_limit, m_value in density_params[density]:
    distance = abs(p - size_limit)
    if distance < closest_distance:
        closest_distance = distance
        m = m_value

print(f'm: {m}')