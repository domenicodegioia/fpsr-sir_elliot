import math

import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm

from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .FPSRModel import FPSRModel


class FPSR(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._sampler = cs.Sampler(self._data.i_train_dict)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        self._params_list = [
            # (variable_name, public_name, shortcut, default, reading_function, printing_function)
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_eigen_dim", "eigen_dim", "eigen_dim", 256, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_rho", "rho", "rho", 5000, int, None),
            ("_w_1", "w_1", "w_1", 0.5, float, None),
            ("_w_2", "w_2", "w_2", 5.0, float, None),
            ("_eta", "eta", "eta", 1.0, float, None),
            ("_tau", "tau", "tau", 0.3, float, None),
            ("_eps", "eps", "eps", 4e-3, float, None)
        ]
        self.autoset_params()

        row, col = data.sp_i_train.nonzero()
        # col = np.array([c + self._num_users for c in col])
        # self._inter = np.array([row, col])
        # self._inter = coo_matrix(np.ones_like(row), (row, col), dtype=np.float64)  # coo_matrix
        self._inter = coo_matrix((np.ones_like(row, dtype=np.float64), (row, col)),
                                 shape=(self._num_users, self._num_items))

        self._model = FPSRModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            eigen_dim=self._eigen_dim,
            l_w=self._l_w,
            tau=self._tau,
            eta=self._eta,
            eps=self._eps,
            w_1=self._w_1,
            w_2=self._w_2,
            rho=self._rho,
            inter=self._inter,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "FPSR" \
            + f"_{self.get_base_params_shortcut()}" \
            + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)

                    if math.isnan(loss) or math.isinf(loss) or (not loss):
                        break

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        raise NotImplementedError

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        raise NotImplementedError

    def evaluate(self, it=None, loss=0):
        raise NotImplementedError

    def restore_weights(self):
        raise NotImplementedError
