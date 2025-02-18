import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Name of the dataset')
parser.add_argument('--model', type=str, help='Name of the model')
args = parser.parse_args()
dataset = args.dataset if args.dataset else 'gowalla'
print(f"Dataset:\t{dataset}")
model = args.model if args.model else 'fpsr'
print(f"Model:\t\t{model}")


heatmap = np.load(f"heatmap/{dataset}/{model}/similarity_matrix.npy")


first = 0
last = 51
heatmap_sampled = heatmap[first:last, first:last]
print(f"We are considering items from {first} to {last}")
del heatmap


cmap = matplotlib.colormaps["Blues"]
# norm = Normalize(vmin=np.min(heatmap_sampled), vmax=np.max(heatmap_sampled))
norm = Normalize(vmin=0, vmax=1)
plt.figure(figsize=(50,50))
img = plt.imshow(heatmap_sampled, cmap=cmap, norm=norm)
model_name = ""
if model == "itemknn":
    model_name = "Item-kNN"
else:
    model_name = "FPSR"
tick_values = np.arange(first, last)
tick_positions = tick_values - first
plt.xticks(ticks=tick_positions, labels=tick_values, fontsize=50, rotation=90)
plt.yticks(ticks=tick_positions, labels=tick_values, fontsize=50)
plt.xlabel("item", fontsize=120)
plt.ylabel("item", fontsize=120)

cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=50)

import time
print(f"Storing figure...")
start = time.time()
plt.savefig(f"heatmap/{dataset}/{model}/plot_{model_name}_{dataset}_{first}to{last}.svg", format="svg", dpi=500, bbox_inches='tight')
end = time.time()
print(f"Storing has taken: {end - start}")

# plt.show(dpi=500)
print("Finished")
