import pickle as pck
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm

#os.system('rm -rf figures/*')
#with open(os.getcwd()+'/results/METR_LA/model_output.pck', 'rb') as f:
#    policy_pred, policy_true, value_pred, value_true = pck.load(f)
#
#    policy_pred = policy_pred.detach().cpu().numpy()
#    policy_true = policy_true.detach().cpu().numpy()
#    value_pred = value_pred.detach().cpu().numpy()
#    value_true = value_true.detach().cpu().numpy()
#
#print(policy_pred)
#
#for i in range(len(policy_pred)):
#    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
#    ax[0].plot(list(range(len(policy_pred[i]))), policy_pred[i], label='pred')
#    ax[1].plot(list(range(len(policy_true[i]))), policy_true[i], label='true')
##    ax[0].bar(range(len(policy_pred[i])), policy_pred[i])
##    print(np.max(policy_pred[i]), np.min(policy_pred[i]))
##    ax[1].bar(range(len(policy_true[i])), policy_true[i])
#    fig.savefig(f'figures/policy_pred_true_{i}.pdf')
#    plt.close()
#
#
with open('results/METR_LA/simulated_annealing/all_results.pck', 'rb') as f:
    best_exps, best_scores = pck.load(f)

print(best_scores)

