import pickle as pck
from matplotlib import pyplot as plt
import numpy as np

with open('results/probabilities.pck', 'rb') as f:
    action_probs = pck.load(f)

#action_probs = [p*100 for p in action_probs]
print(np.unique( action_probs, return_counts=True))
print(np.sum(action_probs))

plt.bar(range(len(action_probs)), action_probs)
plt.show()
