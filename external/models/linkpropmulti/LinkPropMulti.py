import torch
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import time

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class LinkPropMulti(RecMixin, BaseRecommenderModel):
    r"""
    LinkProp-Multi: Iterative Degree-Aware Link Propagation

    Based on:
    "Revisiting Neighborhood-based Link Prediction for Collaborative Filtering"
    https://arxiv.org/abs/2203.15789
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 1024, int, None),
            ("_alpha", "alpha", "alpha", 0.34, float, None),
            ("_beta", "beta", "beta", 0.5, float, None),
            ("_gamma", "gamma", "gamma", 0.5, float, None),
            ("_delta", "delta", "delta", 0.5, float, None),
            ("_r", "r", "r", 2, int, None),
            ("_t", "t", "t", 0.1, float, None)
        ]
        self.autoset_params()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._inter = self._data.sp_i_train.astype(np.float64).tocsr()
        self._preds = None

    def train(self):
        start = time.time()

        M = self._inter.copy()  # Matrice iniziale delle interazioni (CSR)
        M_T = M.transpose().tocsr()  # Transposta in CSR per efficienza

        user_deg = np.array(M.sum(axis=1)).flatten() + 1e-9
        item_deg = np.array(M.sum(axis=0)).flatten() + 1e-9

        for step in range(self._r):
            iter_start = time.time()
            self.logger.info(f"LinkProp iteration {step + 1}/{self._r}")

            # Calcolo dei gradi con gli esponenti
            user_deg_alpha = np.power(user_deg, -self._alpha if step > 0 else 0.0)
            item_deg_beta = np.power(item_deg, -self._beta)
            user_deg_gamma = np.power(user_deg, -self._gamma)
            item_deg_delta = np.power(item_deg, -self._delta)

            D_u_alpha = sp.diags(user_deg_alpha)
            D_i_beta = sp.diags(item_deg_beta)
            D_u_gamma = sp.diags(user_deg_gamma)
            D_i_delta = sp.diags(item_deg_delta)

            # Step di propagazione separato in due fasi
            A = (D_u_alpha @ M) @ D_i_beta
            B = (D_u_gamma @ M) @ D_i_delta

            Temp = A @ M_T  # (n_users x n_users)
            L = Temp @ B  # (n_users x n_items)
            L = L.tocsr()

            if step == self._r - 1:
                self._preds = L
                break

            # Rimozione link già presenti
            M_bool = M.copy()
            M_bool.data = np.ones_like(M_bool.data)
            L = L - L.multiply(M_bool)
            L.eliminate_zeros()

            # Applica filtro top-t%: seleziona solo i punteggi più alti
            if L.nnz > 0:
                cutoff = int(L.nnz * self._t)
                if cutoff > 0:
                    top_indices = np.argpartition(-L.data, cutoff)[:cutoff]
                    mask = np.zeros_like(L.data, dtype=bool)
                    mask[top_indices] = True
                    L.data = L.data * mask
                    L.eliminate_zeros()

            self.logger.info(f"Iteration {step + 1}: {L.nnz} links propagated in {time.time() - iter_start:.2f} sec")

            # Propagazione dei nuovi link nella matrice M
            M = M + L
            M.eliminate_zeros()

            # Aggiorna i gradi
            user_deg = np.array(M.sum(axis=1)).flatten() + 1e-9
            item_deg = np.array(M.sum(axis=0)).flatten() + 1e-9

        total_time = time.time() - start
        self.logger.info(f"Training completed in {total_time:.2f} seconds")

        self.evaluate()

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(tqdm(range(0, self._num_users, self._batch_eval))):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            predictions = self.get_users_rating(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_users_rating(self, batch_start, batch_stop):
        return self._preds[batch_start:batch_stop].toarray()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), torch.tensor(preds).to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    @property
    def name(self):
        return "LinkPropMulti" + f"_{self.get_params_shortcut()}"