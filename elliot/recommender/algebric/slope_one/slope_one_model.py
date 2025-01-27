
"""

Lemire, Daniel, and Anna Maclachlan. "Slope one predictors for online rating-based collaborative filtering."
Proceedings of the 2005 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics
"""
import pickle

import numpy as np


class SlopeOneModel:
    def __init__(self, data):
        self._data = data
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._i_train = self._data.i_train_dict

    # def initialize(self):
    #     freq = np.empty((self._num_items, self._num_items))
    #     dev = np.empty((self._num_items, self._num_items))
    #
    #     # Computation of freq and dev arrays.
    #     for u, u_ratings in self._i_train.items():
    #         for i, r_ui in u_ratings.items():
    #             for j, r_uj in u_ratings.items():
    #                 freq[i, j] += 1
    #                 dev[i, j] += r_ui - r_uj
    #
    #     for i in range(self._num_items):
    #         dev[i, i] = 0
    #         for j in range(i + 1, self._num_items):
    #             dev[i, j] = dev[i, j]/freq[i, j] if freq[i, j] != 0 else 0
    #             dev[j, i] = -dev[i, j]
    #
    #     self.freq = freq
    #     self.dev = dev
    #
    #     # mean ratings of all users: mu_u
    #     self.user_mean = [np.mean([r for (_, r) in self._i_train[u].items()]) for u in range(self._num_users)]

    def initialize(self):
        # Inizializzare freq e dev a zeri
        freq = np.zeros((self._num_items, self._num_items))
        dev = np.zeros((self._num_items, self._num_items))

        # Computazione ottimizzata di freq e dev
        for u_ratings in self._i_train.values():
            items = list(u_ratings.keys())
            ratings = list(u_ratings.values())
            num_ratings = len(items)

            for x in range(num_ratings):
                i = items[x]
                r_ui = ratings[x]
                for y in range(x + 1, num_ratings):  # Solo sopra la diagonale
                    j = items[y]
                    r_uj = ratings[y]
                    freq[i, j] += 1
                    freq[j, i] += 1
                    dev[i, j] += r_ui - r_uj
                    dev[j, i] += r_uj - r_ui

        # Normalizzazione e simmetria
        for i in range(self._num_items):
            for j in range(i + 1, self._num_items):
                if freq[i, j] != 0:
                    dev[i, j] /= freq[i, j]
                    dev[j, i] = -dev[i, j]

        self.freq = freq
        self.dev = dev

        # Calcolo delle medie degli utenti
        self.user_mean = [
            np.mean(list(u_ratings.values())) for u_ratings in self._i_train.values()
        ]

    # def predict(self, user, item):
    #     Ri = [j for (j, _) in self._i_train[user].items() if self.freq[item, j] > 0]
    #     pred = self.user_mean[user]
    #     if Ri:
    #         pred += sum(self.dev[item, j] for j in Ri) / len(Ri)
    #     return pred

    def predict(self, user, item):
        pred = self.user_mean[user]  # Punto di partenza: media dell'utente
        dev_sum = 0  # Somma delle deviazioni
        count = 0  # Numero di elementi con freq > 0

        for j, _ in self._i_train[user].items():
            if self.freq[item, j] > 0:
                dev_sum += self.dev[item, j]
                count += 1

        if count > 0:
            pred += dev_sum / count  # Aggiungi la media delle deviazioni

        return pred

    # def get_user_recs(self, u, mask, k=100):
    #     uidx = self._data.public_users[u]
    #     user_mask = mask[uidx]
    #     # user_items = self._data.train_dict[u].keys()
    #     # indexed_user_items = [self._data.public_items[i] for i in user_items]
    #     predictions = {self._data.private_items[iidx]: self.predict(uidx, iidx) for iidx in range(self._num_items) if user_mask[iidx]}
    #
    #     indices, values = zip(*predictions.items())
    #     indices = np.array(indices)
    #     values = np.array(values)
    #     local_k = min(k, len(values))
    #     partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
    #     real_values = values[partially_ordered_preds_indices]
    #     real_indices = indices[partially_ordered_preds_indices]
    #     local_top_k = real_values.argsort()[::-1]
    #     return [(real_indices[item], real_values[item]) for item in local_top_k]

    def get_user_recs(self, u, mask, k=100):
        user_id = self._data.public_users[u]
        user_mask = mask[user_id]
        predictions = np.full(self._num_items, -np.inf)
        for iidx in range(self._num_items):
            if user_mask[iidx]:
                predictions[iidx] = self.predict(user_id, iidx)
        valid_items = np.sum(user_mask)
        local_k = min(k, valid_items)
        top_k_indices = np.argpartition(predictions, -local_k)[-local_k:]
        top_k_values = predictions[top_k_indices]
        sorted_top_k_indices = top_k_indices[np.argsort(-top_k_values)]
        return [(self._data.private_items[idx], predictions[idx]) for idx in sorted_top_k_indices]

    def get_model_state(self):
        saving_dict = {}
        saving_dict['freq'] = self.freq
        saving_dict['dev'] = self.dev
        saving_dict['user_mean'] = self.user_mean
        return saving_dict

    def set_model_state(self, saving_dict):
        self.freq = saving_dict['freq']
        self.dev = saving_dict['dev']
        self.user_mean = saving_dict['user_mean']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
