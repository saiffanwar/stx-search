import matplotlib.pyplot as plt
import numpy as np



fig, ax = plt.subplots(1, 1, figsize=(10, 5))
xs = np.arange(-10,10,1)
for temp in [10,20,30,40,50,60,70,80,90,100]:
    ys = np.exp(-xs/temp)
#    ys = [1-y for y in ys]

    ax.plot(xs, ys, label=f'Temperature: {temp}')
ax.set_xlabel('x')
ax.set_ylabel('exp(-x/temp)')
ax.legend()
plt.show()


