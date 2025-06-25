import pickle as pck
import matplotlib.pyplot as plt
import numpy as np


def load_results(dataset='PEMS_BAY', model='TGCN'):
    file_path = f'results/training_results/{model}_{dataset}_PGE_losses.pkl'
    with open(file_path, 'rb') as f:
        results = pck.load(f)
    plt.plot(list(range(len(results))), [float(r)
             for r in results], label='PGE Losses')
    plt.show()


def base_model_losses(dataset='PEMS_BAY', model='TGCN'):
    file_path = f'results/training_results/{dataset}_{model}_losses.pck'
    with open(file_path, 'rb') as f:
        results = pck.load(f)

    training_losses = results['train'][1:]
    validation_losses = results['eval'][1:]

    plt.plot(list(range(len(training_losses))), [float(r) for r in training_losses], label='Training Losses')
    plt.plot(list(range(len(validation_losses))), [float(r) for r in validation_losses], label='Validation Losses')
    plt.title(f'{dataset} - {model} Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



with open('scaler.pck', 'rb') as f:
    scaler = pck.load(f)

def plot_fidelities(dataset='METR_LA', model='TGCN'):
    event_ids = [5363900, 933912, 209805, 6220576, 2307113, 2054301, 1872427, 1170528, 6177968, 859791]
    exp_sizes = [20, 50, 75, 100]
    file_path = f'results/{dataset}/'
    methods = ['stx_search', 'tgnnexplainer', 'pg_explainer']

    all_results = {method: {exp_size: [] for exp_size in exp_sizes} for method in methods}
    for method in methods:
        for exp_size in exp_sizes:
            for event_id in event_ids:
                print(f'Processing {method} for event {event_id} with exp size {exp_size}')
                with open(f'{file_path}{method}/{method}_{model}_{dataset}_{event_id}_{exp_size}.pck', 'rb') as f:
                    results = pck.load(f)
                    if method == 'stx_search':
                        all_results[method][exp_size].append(abs(results['target_exp_y'].item() - results['target_model_y'].item()))
                    elif method == 'tgnnexplainer':
                        all_results[method][exp_size].append(abs(scaler.inverse_transform(results['target_model_y']) - scaler.inverse_transform(results['exp_pred'])))
                    elif method == 'pg_explainer':
                        all_results[method][exp_size].append(abs(scaler.inverse_transform(results['target_model_y'].item()) - scaler.inverse_transform(results['exp_pred'].item())))


    with open(f'results/{dataset}/{model}_fidelity_results.pkl', 'wb') as f:
        pck.dump(all_results, f)
    #
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


# plot_fidelities('METR_LA', 'TGCN')
load_results()
# base_model_losses('PEMS_BAY', 'TGCN')
