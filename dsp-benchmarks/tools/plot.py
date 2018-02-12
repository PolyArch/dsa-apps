import matplotlib.pyplot as plt
import numpy as np

colors = [
    '#FF69B4', '#BDB76B', '#FF7F50', '#DDA0DD',
    '#4B0082', '#4B0082', '#006400', '#008080',
    '#00CED1', '#00008B', '#A9A9A9', '#DC143C',
    '#DAA520', '#8B0000', '#FFFF00', '#4169E1'
]

legends = [
    '',
    'CONFIG', 'ISSUED', 'ISSUED_MULTI', 'CONST_FILL',
    'SCR_FILL', 'DMA_FILL', 'REC_WAIT', 'CORE_WAIT', 
    'SCR_BAR_WAIT', 'CMD_QUEUE', 'CGRA_BACK', 'DRAIN', 
    'NOT_IN_USE'
]

raw = open('breakdowns.csv', 'r').readlines()
size = raw[0].strip().split('|')
arch = raw[1].split()

raw = raw[2:]
raw = map(lambda x: map(float, x.strip().split()), raw)

n = len(raw)

fig, ax = plt.subplots(1, n)

fig.tight_layout(pad = 0.1)

bar_width = 0.2

fields = 14

for no in xrange(n):
    line = np.array(raw[no])
    m = 1 + (len(line) - 1) / fields
    ax[no].set_title(size[no])
    ax[no].bar([0], line[0], bar_width, color = 'r', label = 'MKL')
    ind = (bar_width + 0.01) * np.arange(1, m)
    bars = np.array(line[1::fields])
    for i in xrange(fields - 1, 0, -1):
        ax[no].bar(ind, bars, bar_width, color = colors[i], label = legends[i])
        bars = bars - line[1::fields] * line[i+1::fields]
    if no == n - 1:
        ax[no].legend(loc = 1, ncol = 2)
    ax[no].set_xticks(bar_width * np.arange(m))
    ax[no].set_xticklabels(arch, rotation = 75)
plt.subplots_adjust(bottom = 0.15, top = 0.9)
plt.show()
