import torch
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Name of the dataset')
parser.add_argument('--model', type=str, help='Name of the model')
args = parser.parse_args()
dataset = args.dataset
model = args.model

print(f"Loading matrix...")
# similarity_matrix = torch.from_numpy(np.load('heatmap/gowalla/fpsr/self_S.npy')).to(device)
similarity_matrix = torch.from_numpy(np.load(f'heatmap/{dataset}/{model}/similarity_matrix.npy')).to(device)
print(f"Loaded")

print(f"Computing stats...")
data = []
for i in tqdm(range(similarity_matrix.shape[0]), desc="Processing items"):
    row = similarity_matrix[i]

    min_value = torch.min(row).item()
    max_value = torch.max(row).item()
    mean_value = torch.mean(row).item()
    std_dev = torch.std(row).item()

    row_cpu = row.cpu().numpy()
    spearman_corr, _ = spearmanr(row_cpu, np.arange(len(row_cpu)))

    data.append({
        'Item ID': i + 1,
        'Min': min_value,
        'Max': max_value,
        'Mean': mean_value,
        'Standard Deviation': std_dev,
        'Spearman Coefficient': spearman_corr
    })

print(f"Creating dataframe...")
df = pd.DataFrame(data)
print(df.head(20))

print(f"Writing file...")
# df.to_csv('heatmap/gowalla/fpsr/item_stat.tsv', sep='\t', index=False)
# print(f"Created heatmap/gowalla/fpsr/item_stat.txt")
df.to_csv(f'heatmap/{dataset}/{model}/item_stat.tsv', sep='\t', index=False)
print(f"Created heatmap/{dataset}/{model}/item_stat.txt")
