import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

ax.bar(np.arange(5), range(1, 6), 0.3, color = 'r', label = 'M'),
ax.bar(np.arange(5) + 0.3, range(5), 0.3, color = 'b', label = 'N')

ax.legend(loc = 1)

ax.set_xticks(np.arange(10) * 0.5)
ax.set_xticklabels(['ab'] * 10, rotation = 45, fontsize = 11)

plt.show()
