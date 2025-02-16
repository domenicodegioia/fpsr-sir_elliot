import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Name of the dataset')
parser.add_argument('--top_k', type=int, help='Top-k')
args = parser.parse_args()
dataset = args.dataset if args.dataset else 'amazon-cds'
k = args.top_k if args.top_k else 100


print(f"Loading matrices...")
sim_matrix_knn = np.load(f'heatmap/{dataset}/itemknn/similarity_matrix.npy')
sim_matrix_fpsr = np.load(f'heatmap/{dataset}/fpsr/similarity_matrix.npy')

n_items = sim_matrix_knn.shape[0]
spearman_coeffs = []

print(f"Shape from ItemKNN:\t{sim_matrix_knn.shape}")
print(f"Shape from FPSR:\t{sim_matrix_fpsr.shape}")
print(f"# items:\t{n_items}")

for i in tqdm(range(n_items)):
    i_knn = sim_matrix_knn[i, :]
    i_fpsr = sim_matrix_fpsr[i, :]

    sorted_indices_knn = np.argsort(-i_knn)[:k]  # top-k
    sorted_indices_fpsr = np.argsort(-i_fpsr)[:k]

    coefficient, _ = spearmanr(sorted_indices_knn, sorted_indices_fpsr)
    spearman_coeffs.append((i, coefficient))

avg_spearman = np.mean([c for i,c in spearman_coeffs])

# print(f"Coefficiente di Spearman per ogni item: {spearman_coeffs}")
print(f"Media dei coefficienti di Spearman: {avg_spearman}")
