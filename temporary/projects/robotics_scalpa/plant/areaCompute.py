from matplotlib import pyplot as pl

import numpy as np

from scipy import interpolate

x = np.linspace(-0.0301, 0.036, 500)

y = 71309.0396153*(x**4) - 840.1696*(x**3) - 69.2694*(x**2) + 0.471299*x - 0.003878872

pl.figure(figsize = (8, 4))

pl.plot(x, y, color="blue", linewidth = 1.5)

pl.show()

pl.figure

pl.plot

pl.show