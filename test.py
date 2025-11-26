import numpy as np
import matplotlib.pyplot as plt
def sig(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-20, 20, 300)
plt.plot(x, list(map(sig, x)))
plt.grid(True)
plt.xlim((-5, 5))
plt.ylim((0, 1))
plt.show()