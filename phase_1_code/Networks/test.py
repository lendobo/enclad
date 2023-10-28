import numpy as np

duration = 10

l_lo = 0 # 0.04050632911392405
l_hi = 0.4 # 0.1569620253164557

p_range = 300
n = [500, 250, 100, 50]
b_values = [int(0.7 * sampsize) for sampsize in n]   # [int(0.7 * n), int(0.75 * n), int(0.8 * n)]
Q_values = 1000 # 1000
lambda_range = np.linspace(l_lo, l_hi, 20)


with open(f'out/duration_{p_range,n,b_values,Q_values,len(lambda_range)}.txt', 'w') as f:
    f.write(f"{duration} seconds")