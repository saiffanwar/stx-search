import pickle as pck
from matplotlib import pyplot as plt
import numpy as np
import os

with open(os.getcwd()+'/results/METR_LA/model_output.pck', 'rb') as f:
    policy_pred, policy_true, value_pred, value_true = pck.load(f)

    policy_pred = policy_pred.detach().cpu().numpy()
    policy_true = policy_true.detach().cpu().numpy()
    value_pred = value_pred.detach().cpu().numpy()
    value_true = value_true.detach().cpu().numpy()

#action_probs = [p*100 for p in action_probs]
print(np.unique(policy_pred[0]))
print(np.unique(policy_true[0]))

#plt.bar(range(len(action_probs)), action_probs)
#plt.show()
