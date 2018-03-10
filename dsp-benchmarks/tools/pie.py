import matplotlib.pyplot as plt
import numpy as np

colors = [
    '#FF69B4', '#BDB76B', '#FF7F50', '#DDA0DD',
    '#4B0082', '#4B0082', '#006400', '#008080',
    '#00CED1', '#00008B', '#A9A9A9', '#DC143C',
    '#DAA520', '#4169E1'
]

legends = [
    'Scratch Pad (%.2f mm$^2$)',
    'INT-ADD/MUL (%.2f mm$^2$)',
    #'FMul/Div/Sqrt (%.2f mm$^2$)',
    #'F-ADD/MUL (%.2f mm$^2$)',
    'Network (%.2f mm$^2$)',
    'RISCV (%.2f mm$^2$)',
    'Stream Engine (%.2f mm$^2$)',
    'Vector Port (%.2f mm$^2$)'
]

explode = [
    0.0,
    0.1,
    #0.1,
    #0.1,
    0.0,
    0.0,
    0.0,
    0.0
]


val = [
    0.11,
    0.06 + 0.15,
    #0.34,
    #0.25 + 0.22,
    0.34,
    0.16,
    0.02,
    0.03
]

labels = [legends[i] % val[i] for i in xrange(len(val))]

total = sum(val)

val = map(lambda x: x / total * 100, val)

fig, ax = plt.subplots(1, 1)

ax.pie(val, labels = labels, shadow = False, explode = explode)
ax.axis('equal')

plt.show()
