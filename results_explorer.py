import pickle as pck
import matplotlib.pyplot as plt


def load_results():
    file_path = 'results/training_results/pge_losses.pkl'
    with open(file_path, 'rb') as f:
        results = pck.load(f)
    return results


results = load_results()

plt.plot(list(range(len(results))), [float(r)
         for r in results], label='PGE Losses')
plt.show()
