import numpy as np
from scipy.stats import spearmanr


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Name of the dataset')
parser.add_argument('--model', type=str, help='Name of the model')
args = parser.parse_args()
dataset = args.dataset if args.dataset else 'amazon-cds'
model = args.model if args.dataset else 'fpsr'


print(f"Loading matrix...")
similarity_matrix = np.load(f'heatmap/{dataset}/{model}/similarity_matrix.npy')
print(f"Collecting values...")
values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
#del similarity_matrix
print(f"Computing stats...")


min_value = np.min(values)
max_value = np.max(values)
print(f"Min: {min_value}")
print(f"Max: {max_value}")


n = len(values)
values = values[values != 0]
print(f"Elementi non nulli: {len(values)} / {n}")


mean_value = np.mean(values)
std_dev = np.std(values)
print(f"Mean: {mean_value}")
print(f"Standard deviation: {std_dev}")


# spearman_corr, _ = spearmanr(values, np.arange(len(values)))
# print(f"Spearman Coefficient: {spearman_corr}")


# map = {
#     'amazon-cds': {'id_most': 288, 'votes_most': 601, 'id_least': 12524, 'votes_least': 1},
#     'douban': {'id_most': 89, 'votes_most': 1934, 'id_least': 88, 'votes_least': 1},
#     'yelp2018': {'id_most': 783, 'votes_most': 1258, 'id_least': 10140, 'votes_least': 1},
#     'gowalla': {'id_most': 2525, 'votes_most': 1415, 'id_least': 40914, 'votes_least': 1},
# }
# id_most  = map[dataset]['id_most']
# id_least = map[dataset]['id_least']
# votes_most  = map[dataset]['votes_most']
# votes_least = map[dataset]['votes_least']
# most_list  = similarity_matrix[id_most]
# least_list = similarity_matrix[id_least]
# # sp, _ = spearmanr(most_list, least_list)
# print("Most voted item:", id_most, ", votes:", votes_most)
# print("Least voted item:", id_least, ", votes:", votes_least)
# # print(f"Spearman Coefficient between them: {sp}")


import os
print(f"Writing files...")
with open(f'{os.getcwd()}/heatmap/{dataset}/{model}/global_stat.txt', 'w') as f:
    f.write(f"Min: {min_value}\n")
    f.write(f"Max: {max_value}\n")
    f.write(f"Mean: {mean_value}\n")
    f.write(f"Standard deviation: {std_dev}\n")
    f.write(f"Elementi non nulli: {len(values)} / {n}\n")
    # f.write(f"Spearman Coefficient: {spearman_corr}\n\n")
    # f.write(f"Most voted item: {id_most}, votes: {votes_most}\n")
    # f.write(f"Least voted item: {id_least}, votes: {votes_least}\n")
    # f.write(f"Spearman Coefficient between them: {sp}\n")
    print(f"Created {os.getcwd()}/heatmap/{dataset}/{model}/global_stat.txt")
