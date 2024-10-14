import pickle as pck
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm

os.system('rm -rf figures/*')
with open(os.getcwd()+'/results/METR_LA/model_output.pck', 'rb') as f:
    policy_pred, policy_true, value_pred, value_true = pck.load(f)

    policy_pred = policy_pred.detach().cpu().numpy()
    policy_true = policy_true.detach().cpu().numpy()
    value_pred = value_pred.detach().cpu().numpy()
    value_true = value_true.detach().cpu().numpy()

print(policy_pred)

for i in range(len(policy_pred)):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax[0].plot(list(range(len(policy_pred[i]))), policy_pred[i], label='pred')
    ax[1].plot(list(range(len(policy_true[i]))), policy_true[i], label='true')
#    ax[0].bar(range(len(policy_pred[i])), policy_pred[i])
#    print(np.max(policy_pred[i]), np.min(policy_pred[i]))
#    ax[1].bar(range(len(policy_true[i])), policy_true[i])
    fig.savefig(f'figures/policy_pred_true_{i}.pdf')
    plt.close()


#
#
#def visualise_exp_evolution(all_paths):
#
#    fig, ax = plt.subplots()
#    node_locations = {}
#    for n, path in tqdm(enumerate( all_paths)):
#        xs, ys = [], []
#        for node in path:
#            if node in node_locations.keys():
#                pass
#            else:
#                node_locations[node] = [path.index(node), n]
#                xs.append(node_locations[node][0])
#                ys.append(node_locations[node][1])
#
#            if node != path[0]:
#                prev_node = path[path.index(node)-1]
#                start_x, start_y = node_locations[prev_node]
#                end_x, end_y = node_locations[node]
#
#                ax.plot([start_x, end_x], [start_y, end_y], 'k-', zorder=0, alpha=0.5)
#
#            ax.scatter(xs, ys, s=100, zorder=1, )
#
#    fig.savefig(f'figures/tree_search.pdf')
#
#
#with open(os.getcwd()+'/results/METR_LA/5000/1training_data_worker_0_0.pck', 'rb') as f:
#    x_train, y_probs, y_vals, all_paths = pck.load(f)
#
#print(len(all_paths))
#all_paths = all_paths[-1000:]
#visualise_exp_evolution(all_paths)
