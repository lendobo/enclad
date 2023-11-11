import numpy as np
import matplotlib.pyplot as plt

# generate x values from 1 to 100
x = np.arange(1, 10)

# plot 2^x aainst 10^x
plt.plot(x, 2 ** x, label='2^x')
plt.plot(x, 10 ** x, label='10^x')

# add title and labels
plt.title('2^x vs 10^x')
plt.xlabel('x')
plt.ylabel('y')

# add legend
plt.legend()


plt.show()

plt.plot(np.log10(2**x), np.log10(10**x))
plt.show()