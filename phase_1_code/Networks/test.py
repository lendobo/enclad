import numpy as np

mat1 = np.ones((3, 3))
mat2 = np.ones((3, 3))

summy = np.sum(mat1.flatten(), mat2.flatten())
print(summy)