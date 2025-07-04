
from tqdm import tqdm
import os
import time

from elliot.utils.write import store_recommendation
from .custom_sampler import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .UltraGCNModel import UltraGCNModel

import math
import torch
import numpy as np


class UltraGCN(RecMixin, BaseRecommenderModel):
    r"""
    UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3459637.3482291>`_

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 1e-4, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            # L = -(w1 + w2*\beta)) * log(sigmoid(e_u e_i)) - \sum_{N-} (w3 + w4*\beta) * log(sigmoid(e_u e_i'))
            ("_w1", "w1", "w1", 1e-7, float, None),
            ("_w2", "w2", "w2", 1, float, None),
            ("_w3", "w3", "w3", 1, float, None),
            ("_w4", "w4", "w4", 1, float, None),
            ("_ii_n_n", "ii_n_n", "ii_n_n", 10, int, None),  # ii_neighbor_num
            ("_i_w", "i_w", "i_w", 1e-3, float, None),
            ("_n_n", "n_n", "n_n", 200, int, None),  # negative_num
            ("_n_w", "n_w", "n_w", 200, int, None),  # negative_weight
            ("_g", "g", "g", 1e-4, float, None),  # weight of l2 normalization
            ("_l", "l", "l", 2.75, float, None),
            ("_s_s_p", "s_s_p", "s_s_p", False, bool, None)
        ]
        self.autoset_params()

        row, col = data.sp_i_train.nonzero()
        edge_index = list(zip(row, col))

        interacted_items = [[] for _ in range(self._num_users)]
        for (u, i) in edge_index:
            interacted_items[u].append(i)

        self._sampler = Sampler(edge_index=edge_index,
                                num_items=self._num_items,
                                batch_size=self._batch_size,
                                negative_num=self._n_n,
                                sampling_sift_pos=self._s_s_p,
                                interacted_items=interacted_items,
                                seed=self._seed)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        ii_neighbor_mat, ii_constraint_mat = self.get_ii_constraint_mat(data.sp_i_train, self._ii_n_n)

        items_D = np.sum(data.sp_i_train, axis=0).reshape(-1)
        users_D = np.sum(data.sp_i_train, axis=1).reshape(-1)

        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

        constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                          "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}

        self._model = UltraGCNModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            w1=self._w1,
            w2=self._w2,
            w3=self._w3,
            w4=self._w4,
            initial_weight=self._i_w,
            negative_num=self._n_n,
            negative_weight=self._n_w,
            ii_neighbor_mat=ii_neighbor_mat,
            ii_constraint_mat=ii_constraint_mat,
            constraint_mat=constraint_mat,
            gamma=self._g,
            lm=self._l,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "UltraGCN" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    @staticmethod
    def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero=False):

        A = train_mat.T.dot(train_mat)  # I * I
        n_items = A.shape[0]
        res_mat = torch.zeros((n_items, num_neighbors))
        res_sim_mat = torch.zeros((n_items, num_neighbors))
        if ii_diagonal_zero:
            A[range(n_items), range(n_items)] = 0
        items_D = np.sum(A, axis=0).reshape(-1)
        users_D = np.sum(A, axis=1).reshape(-1)

        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
        all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
        for i in range(n_items):
            row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
            row_sims, row_idxs = torch.topk(row, num_neighbors)
            res_mat[i] = row_idxs
            res_sim_mat[i] = row_sims

        return res_mat.long(), res_sim_mat.float()

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            start = time.time()
            n_batch = int(self._data.transactions / self._batch_size) if self._data.transactions % self._batch_size == 0 else int(self._data.transactions / self._batch_size) + 1
            with tqdm(total=n_batch, disable=not self._verbose) as t:
                for x in self._sampler.train_loader:
                    steps += 1
                    users, pos_items, neg_items = self._sampler.step(x)
                    loss += self._model.train_step((users,
                                                    pos_items,
                                                    neg_items))

                    if math.isnan(loss) or math.isinf(loss) or (not loss):
                        break

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            end = time.time()
            #self.logger.info(f"Training has taken: {end - start}")

            need_test = True
            if it < 50 and it % 5 != 0:
                need_test = False
            if need_test:
                self.evaluate(it, loss / (it + 1))

        if it + 1 == self._epochs and self._write_best_iterations:  # never met an early stopping condition
            with open(self._config.path_output_rec_performance + '/best_iterations.tsv', 'a') as f:
                f.write(self.name + '\t' + str(self._params.best_iteration) + '\n')
            self.logger.info(f"Best iteration: {self._params.best_iteration}")
            self.logger.info(f"Current configuration: {self.name}")

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(torch.arange(offset, offset_stop))
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False