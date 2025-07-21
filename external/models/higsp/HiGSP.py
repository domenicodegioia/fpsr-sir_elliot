import torch
import numpy as np
from scipy.sparse import csr_matrix
import scipy
from tqdm import tqdm
import scipy.sparse as sp
import time
from sklearn.mixture import GaussianMixture
import sys
from scipy.sparse.linalg import svds

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class HiGSP(RecMixin, BaseRecommenderModel):
    r"""
    Hierarchical Graph Signal Processing for Collaborative Filtering

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3589334.3645368>`_

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 1024, int, None),
            ("_pri_factor", "pri_factor", "pri_factor", 80, int, None),
            ("_alpha_1", "alpha_1", "alpha_1", 0.08, float, None),
            ("_alpha_2", "alpha_2", "alpha_2", 0.73, float, None),
            ("_order1", "order1", "order1", 2, int, None),
            ("_order2", "order2", "order2", 12, int, None),
            ("_n_clusters", "n_clusters", "n_clusters", 25, int, None),
        ]

        self.autoset_params()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        row, col = data.sp_i_train.nonzero()
        self.adj_mat = np.zeros(data.sp_i_train.shape, dtype=np.float64)
        self.adj_mat[row, col] = 1.0
        # self.d_mat_i, self.d_mat_i_inv, self.vt, self.norm_adj = None, None, None, None



    def bmatpow(self, mat, order):
        with torch.no_grad():
            R = torch.FloatTensor(mat)
            mat = torch.FloatTensor(mat)
            I = torch.FloatTensor(np.expand_dims(np.identity(mat.shape[1]), axis=0).repeat(self._n_clusters, axis=0))
            if order == 1:
                return R.detach().numpy()
            for ord in range(2, order + 1):
                R = torch.bmm(R.transpose(1, 2), mat)
            R = R.detach().numpy()
        return R

    def matpow(self, mat, order):
        with torch.no_grad():
            R = torch.FloatTensor(mat)
            mat = torch.FloatTensor(mat)
            if order == 1:
                return R.detach().numpy()
            for ord in range(2, order + 1):
                R = torch.matmul(R.transpose(0, 1), mat)
            R = R.detach().numpy()
        return R

    def normalize_adj_mat(self, adj_mat):
        adj_mat = sp.csr_matrix(adj_mat)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_inv = 1.0 / d_inv
        d_inv[np.isinf(d_inv)] = 0.
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.A
        return norm_adj

    def normalize_adj_mat_sp(self, adj_mat):
        adj_mat = sp.csr_matrix(adj_mat)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_i = d_mat
        d_inv = 1.0 / d_inv
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_i_inv = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        return norm_adj, d_mat_i, d_mat_i_inv

    def construct_cluster_wise_filter(self, adj_mat):
        # Cluster users based on their interactions
        clustering = GaussianMixture(n_components=self._n_clusters, verbose=2)
        print("Fitting clustering...")
        start = time.time()
        cluster_labels = clustering.fit_predict(adj_mat)
        end = time.time()
        self.logger.info(f"Time for clustering: {end - start}")
        n_clusters = len(set(cluster_labels))
        self.logger.info(f"Number of clusters: {n_clusters}")

        C = np.zeros((n_clusters, adj_mat.shape[0], adj_mat.shape[1]))
        C[cluster_labels, [i for i in tqdm(range(adj_mat.shape[0]))], :] = adj_mat

        # Construct filters for each cluster
        A_tilde_list = []
        for i in tqdm(range(n_clusters)):
            adj_mat = C[i, :, :]
            C_tilde = adj_mat
            A_tilde = C_tilde.T @ C_tilde
            A_tilde = self.normalize_adj_mat(A_tilde)
            A_tilde = np.expand_dims(A_tilde, axis=0)
            A_tilde_list.append(A_tilde)
        A_tilde = np.concatenate(A_tilde_list, axis=0)
        L_tilde = np.expand_dims(np.identity(A_tilde.shape[1]), axis=0).repeat(self._n_clusters, axis=0) - A_tilde
        L_tilde_k = self.bmatpow(L_tilde, self._order1)
        local_filter = np.expand_dims(np.identity(A_tilde.shape[1]), axis=0).repeat(self._n_clusters, axis=0) - L_tilde_k
        return local_filter, cluster_labels

    def construct_global_aware_filter(self, adj_mat):
        # Construct ideal low-pass filter
        norm_adj, d_mat_i, d_mat_i_inv = self.normalize_adj_mat_sp(adj_mat)
        norm_adj = norm_adj.tocsc()
        # ut, s, vt = np.linalg.svd(norm_adj, self._pri_factor)
        k = min(self._pri_factor, min(norm_adj.shape) - 1)
        _, _, vt = svds(norm_adj, k=k)
        # vt = np.flip(vt, axis=0)
        global_filter1 = d_mat_i @ vt.T @ vt @ d_mat_i_inv

        # Construct high-order low-pass filter
        R_tilde = self.normalize_adj_mat(adj_mat)
        P_tilde = R_tilde.T @ R_tilde
        L_tilde = np.identity(P_tilde.shape[1]) - P_tilde
        L_tilde_k = self.matpow(L_tilde, self._order2)
        global_filter2 = np.identity(P_tilde.shape[1]) - L_tilde_k
        return global_filter1, global_filter2

    def train(self):
        start = time.time()

        self.logger.info(f"Construct item-wise filter")
        self.item_cluster_filter, self.item_cluster_labels = self.construct_cluster_wise_filter(self.adj_mat)

        self.logger.info(f"Construct globally-aware filter")
        self.global_filter1, self.global_filter2 = self.construct_global_aware_filter(self.adj_mat)

        self.logger.info(f"Predict user future interactions")
        self.preds = self.compute_preds()

        end = time.time()
        self.logger.info(f"Training has taken: {end - start}")

        self.evaluate()

    def compute_preds(self):
        ratings = 0.0

        # Predictions from cluster-wise filter
        n_clusters = len(set(self.item_cluster_labels))
        C = np.zeros((n_clusters, self.adj_mat.shape[0], self.adj_mat.shape[1]))
        C[self.item_cluster_labels, [i for i in range(self.adj_mat.shape[0])], :] = self.adj_mat
        with torch.no_grad():
            C = torch.FloatTensor(C)
            filter = torch.FloatTensor(self.item_cluster_filter)
            ratings += torch.bmm(C, filter)
            ratings = torch.sum(ratings, dim=0, keepdim=False)
            ratings = ratings.detach().numpy()
        # Predictions from ideal low-pass filter in globally-aware filter
        ratings += self._alpha_1 * self.adj_mat @ self.global_filter1
        # Predictions from high-order low-pass filter in globally-aware filter
        ratings += self._alpha_2 * self.adj_mat @ self.global_filter2
        return ratings

    def get_users_rating(self, batch_start, batch_stop):
        return self.preds[batch_start:batch_stop]

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
        return "FaGSP" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"