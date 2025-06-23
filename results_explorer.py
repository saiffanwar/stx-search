import pickle as pck
import matplotlib.pyplot as plt
import numpy as np


def load_results():
    file_path = 'results/training_results/pge_losses.pkl'
    with open(file_path, 'rb') as f:
        results = pck.load(f)
    plt.plot(list(range(len(results))), [float(r)
             for r in results], label='PGE Losses')
    plt.show()


with open('scaler.pck', 'rb') as f:
    scaler = pck.load(f)

def plot_fidelities(dataset='METR_LA', model='TGCN'):
    event_ids = [5363900, 933912, 209805, 6220576, 2307113, 2054301, 1872427, 1170528, 6177968, 859791]
    exp_sizes = [20, 50, 75, 100]
    file_path = f'results/{dataset}/'

    # all_results = {method: {exp_size: [] for exp_size in exp_sizes} for method in ['stx_search', 'tgnnexplainer']}
    # for method in ['stx_search', 'tgnnexplainer']:
    #     for exp_size in exp_sizes:
    #         for event_id in event_ids:
    #             try:
    #                 with open(f'{file_path}{method}/{method}_{model}_{dataset}_{event_id}_{exp_size}.pck', 'rb') as f:
    #                     results = pck.load(f)
    #                     if method == 'stx_search':
    #                         all_results[method][exp_size].append(abs(results['target_exp_y'].item() - results['target_model_y'].item()))
    #                     elif method == 'tgnnexplainer':
    #                         all_results[method][exp_size].append(abs(scaler.inverse_transform(results['target_model_y']) - scaler.inverse_transform(results['exp_pred'])))
    #             except:
    #                 pass
    #
    #
    # with open(f'results/{dataset}/{model}_fidelity_results.pkl', 'wb') as f:
    #     pck.dump(all_results, f)

    with open(f'results/{dataset}/{model}_fidelity_results.pkl', 'rb') as f:
        all_results = pck.load(f)

    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    for method in all_results.keys():
        method_results = []
        for exp_size in all_results[method].keys():
            method_results.append(np.mean(all_results[method][exp_size]))
        axes.plot(exp_sizes, method_results, label=f'{method}')
    axes.set_title(f'Fidelity Comparison for {dataset} - {model}')
    axes.set_xlabel('Exp Size')
    axes.set_ylabel('Explanation Instance MAE')
    fig.legend(['STX Search', 'TGNNExplainer'], loc='upper right')
    plt.show()

    print(all_results)


plot_fidelities('METR_LA', 'TGCN')
