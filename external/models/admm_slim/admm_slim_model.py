import pickle
import time
import numpy as np
import sys
import scipy.sparse as sp
from tqdm import tqdm
import torch
import random

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")

def soft_threshold(x, threshold):
    return (np.abs(x) > threshold) * (np.abs(x) - threshold) * np.sign(x)

class ADMMSlimModel:
    def __init__(self,
                 data,
                 num_users,
                 num_items,
                 l1,
                 l2,
                 alpha,
                 iterations,
                 rho,
                 random_seed=42):
        # set seed
        # self.random_seed = random_seed
        # random.seed(self.random_seed)
        # np.random.seed(self.random_seed)
        # torch.manual_seed(self.random_seed)
        # torch.cuda.manual_seed(self.random_seed)
        # torch.cuda.manual_seed_all(self.random_seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        #
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._data = data
        self.num_users = num_users
        self.num_items = num_items
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.iterations = iterations
        self.rho = rho

        self.X = self._data.sp_i_train
        self.item_means = self.X.mean(axis=0).getA1()
        self.X = self.X.toarray() - self.item_means

        self._w_sparse = None
        self.pred_mat = None

    def train(self):
        # pre-compute
        self.X = self._data.sp_i_train
        G = (self.X.T @ self.X).toarray()
        diag = self.l2 * np.diag(np.power(self.item_means, self.alpha)) + self.rho * np.identity(self.num_items)
        logger.info("Computing P...")
        P = np.linalg.inv(G + diag).astype(np.float32)
        logger.info("Computing B_aux...")
        B_aux = (P @ G).astype(np.float32)

        # initialize
        Gamma = np.zeros(G.shape, dtype=float)
        C = np.zeros(G.shape, dtype=float)
        del diag, G

        # iterate until convergence
        for i in tqdm(range(self.iterations), disable=True):
            start = time.time()
            B_tilde = B_aux + P @ (self.rho * C - Gamma)
            gamma = np.diag(B_tilde) / (np.diag(P) + 1e-7)
            B = B_tilde - P * gamma
            C = soft_threshold(B + Gamma / self.rho, self.l1 / self.rho)
            Gamma += self.rho * (B - C)
            logger.info(f"Iteration {i} has taken:\t{time.time() - start}")

        self._w_sparse = C

    def prepare_predictions(self):
        self.pred_mat = self.X.dot(self._w_sparse) #.toarray()

    # def predict(self, u, i):
    #     return self.pred_mat[u, i]

    def get_user_recs(self, user, mask, k=100):
        ui = self._data.public_users[user]
        user_mask = mask[ui]
        predictions = self.pred_mat[ui].copy()
        predictions[~user_mask] = -np.inf
        valid_items = user_mask.sum()
        local_k = min(k, valid_items)
        top_k_indices = np.argpartition(predictions, -local_k)[-local_k:]
        top_k_values = predictions[top_k_indices]
        sorted_top_k_indices = top_k_indices[np.argsort(-top_k_values)]
        return [(self._data.private_items[idx], predictions[idx]) for idx in sorted_top_k_indices]
