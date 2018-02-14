import matplotlib.pyplot as plt
import numpy

colors = [
    '#FF69B4', '#BDB76B', '#FF7F50', '#DDA0DD',
    '#4B0082', '#4B0082', '#006400', '#008080',
    '#00CED1', '#00008B', '#A9A9A9', '#DC143C',
    '#DAA520', '#4169E1'
]

labels = [
    'Cholesky',
    'QR',
    'GeMM',
    'SVD',
    'FFT'
]

val = [
    [(2, 0.78), (8.0, 5.31), (7, 1.99), (39.0, 82.89), (2.0 / 50, 0.45)],
    [(8, 6.09), (34.0, 33.52), (27, 23.28), (252, 496.63), (121.0 / 50, 8.27)]
]

fig, ax = plt.subplots(1, 2)
n = len(labels)

arch = ['MKL', 'SB/REVEL']

for i in xrange(2):
    for j in xrange(2):
        ax[i].bar(numpy.arange(n) + j * 0.3, [(a[j] / a[0]) if j == 0 else a[0] / a[j] for a in val[i]], width = 0.28, label = arch[j])
    ax[i].set_xticks(numpy.arange(n))
    ax[i].set_xticklabels(labels, rotation = 75)
    ax[i].set_ylim(0, 4)

ax[0].set_title('Small')
ax[1].set_title('Large')
ax[1].legend()

plt.show()
