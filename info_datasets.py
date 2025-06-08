import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform


def analyze_recommendation_dataset(file_path, min_interactions=5):
    # Caricamento del dataset
    df = pd.read_csv(file_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Numero totale di interazioni
    total_interactions = len(df)
    print(f"Numero totale di interazioni: {total_interactions}")

    # Numero totale di utenti e oggetti
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    print(f"Numero totale di utenti: {num_users}")
    print(f"Numero totale di oggetti: {num_items}")

    # Sparsità del dataset
    density = total_interactions / (num_users * num_items)
    sparsity = 1 - (total_interactions / (num_users * num_items))
    print(f"Sparsità: {sparsity:.6f}")
    print(f"Densità: {density:.6f}")

    # Numero medio di interazioni per utente e per oggetto
    interactions_per_user = df.groupby('user_id').size()
    interactions_per_item = df.groupby('item_id').size()
    avg_interactions_per_user = interactions_per_user.mean()
    avg_interactions_per_item = interactions_per_item.mean()
    print(f"Numero medio di interazioni per utente: {avg_interactions_per_user:.2f}")
    print(f"Numero medio di interazioni per oggetto: {avg_interactions_per_item:.2f}")

    # Percentuale di utenti con almeno 'min_interactions' interazioni
    users_above_min = (interactions_per_user >= 5).mean() * 100
    print(f"Percentuale di utenti con almeno 5 interazioni: {users_above_min:.2f}%")

    users_above_min = (interactions_per_user >= 2).mean() * 100
    print(f"Percentuale di utenti con almeno 2 interazioni: {users_above_min:.2f}%")

    users_above_min = (interactions_per_user >= 1).mean() * 100
    print(f"Percentuale di utenti con almeno 1 interazione: {users_above_min:.2f}%")

    # Copertura dell’interazione: percentuale di oggetti con almeno una interazione
    coverage = (df['item_id'].nunique() / num_items) * 100
    print(f"Copertura dell’interazione: {coverage:.2f}%")

# Esempio di utilizzo
analyze_recommendation_dataset('data/woman-1m/woman-1m.tsv')
