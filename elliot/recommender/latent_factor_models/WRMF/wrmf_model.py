"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")

class WRMFModel(object):
    """
    Simple Matrix Factorization class
    """

    def __init__(self, factors, data, random, alpha, reg):

        self._data = data
        self.random = random
        self.C = alpha * self._data.sp_i_train
        self.train_dict = self._data.train_dict
        self.user_num, self.item_num = self._data.num_users, self._data.num_items

        self.X = sp.csr_matrix(self.random.normal(scale=0.01,
                                                  size=(self.user_num, factors)))
        self.Y = sp.csr_matrix(self.random.normal(scale=0.01,
                                                  size=(self.item_num, factors)))
        self.X_eye = sp.eye(self.user_num)
        self.Y_eye = sp.eye(self.item_num)
        self.lambda_eye = reg * sp.eye(factors)

        self.user_vec, self.item_vec, self.pred_mat = None, None, None

    def train_step(self):
        yTy = self.Y.T.dot(self.Y)
        xTx = self.X.T.dot(self.X)
        for u in range(self.user_num):
            Cu = self.C[u, :].toarray()
            Pu = Cu.copy()
            Pu[Pu != 0] = 1
            CuI = sp.diags(Cu, [0])
            yTCuIY = self.Y.T.dot(CuI).dot(self.Y)
            yTCuPu = self.Y.T.dot(CuI + self.Y_eye).dot(Pu.T)
            self.X[u] = spsolve(yTy + yTCuIY + self.lambda_eye, yTCuPu)
        for i in range(self.item_num):
            Ci = self.C[:, i].T.toarray()
            Pi = Ci.copy()
            Pi[Pi != 0] = 1
            CiI = sp.diags(Ci, [0])
            xTCiIX = self.X.T.dot(CiI).dot(self.X)
            xTCiPi = self.X.T.dot(CiI + self.X_eye).dot(Pi.T)
            self.Y[i] = spsolve(xTx + xTCiIX + self.lambda_eye, xTCiPi)

        self.pred_mat = self.X.dot(self.Y.T).A

    def predict(self, user, item):
        return self.pred_mat[self._data.public_users[user], self._data.public_items[item]]

    def get_user_recs(self, user, mask, k=100):
        user_id = self._data.public_users[user]
        user_recs = self.pred_mat[user_id]
        masked_recs = np.where(mask[user_id], user_recs, -np.inf)
        valid_items = np.sum(mask[user_id])
        local_k = min(k, valid_items)
        if local_k == 0:
            return []
        top_k_indices = np.argpartition(masked_recs, -local_k)[-local_k:]
        top_k_values = masked_recs[top_k_indices]
        sorted_top_k_indices = top_k_indices[np.argsort(-top_k_values)]
        return [(self._data.private_items[idx], masked_recs[idx]) for idx in sorted_top_k_indices]

    def get_model_state(self):
        saving_dict = {}
        saving_dict['pred_mat'] = self.pred_mat
        saving_dict['X'] = self.X
        saving_dict['Y'] = self.Y
        saving_dict['C'] = self.C
        return saving_dict

    def set_model_state(self, saving_dict):
        self.pred_mat = saving_dict['pred_mat']
        self.X = saving_dict['X']
        self.Y = saving_dict['Y']
        self.C = saving_dict['C']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
