file_path = "test.txt"

with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

data_dict = {}

for line in lines:
    numbers = list(map(int, line.split()))  # Convertire la riga in una lista di numeri
    if numbers:  # Assicurarsi che la riga non sia vuota
        key = numbers[0]  # Primo numero come chiave
        values = numbers[1:]  # Tutti gli altri numeri come valori
        data_dict[key] = values

import pandas as pd
df = pd.DataFrame(
    [(key, value) for key, values in data_dict.items() for value in values],
).sort_values(by=[0,1])

df.to_csv("test.tsv", sep='\t', header=False, index=False)
