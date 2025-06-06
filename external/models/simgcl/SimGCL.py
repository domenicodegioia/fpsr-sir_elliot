from tqdm import tqdm
import numpy as np
import torch
import os

from elliot.utils.write import store_recommendation
from .custom_sampler import *
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .SimGCLModel import SimGCLModel

from torch_sparse import SparseTensor

import math


class SimGCL(RecMixin, BaseRecommenderModel):
    r"""
    Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3477495.3531937>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        n_layers: Number of stacked propagation layers
        eps: Perturbation rate
        reg_cl: Regularization for cl loss

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        SimGCL:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          batch_size: 512
          factors: 64
          batch_size: 256
          l_reg: 0.1
          n_layers: 2
          eps: 0.2
          reg_cl: 0.01
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_w", "l_reg", "l_reg", 0.01, float, None),
            ("_reg_cl", "reg_cl", "reg_cl", 0.01, float, None),
            ("_n_layers", "n_layers", "n_layers", 1, int, None),
            ("_normalize", "normalize", "normalize", True, bool, None),
            ("_eps", "eps", "eps", 0.2, float, None)
        ]
        self.autoset_params()

        self.ui_dict = {u: list(set(self._data.i_train_dict[u])) for u in self._data.i_train_dict}
        self.edge_index = self._data.edge_index.to_numpy().tolist()

        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        edge_index = np.array([row, col])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.adj = SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                                col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                                sparse_sizes=(self._num_users + self._num_items,
                                              self._num_users + self._num_items))

        self._model = SimGCLModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            l_w=self._l_w,
            n_layers=self._n_layers,
            eps=self._eps,
            reg_cl=self._reg_cl,
            adj=self.adj,
            normalize=self._normalize,
            random_seed=self._seed,
        )

    @property
    def name(self):
        return "SimGCL" \
            + f"_{self.get_base_params_shortcut()}" \
            + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            n_batch = int(
                self._data.transactions / self._batch_size) if self._data.transactions % self._batch_size == 0 else int(
                self._data.transactions / self._batch_size) + 1
            with tqdm(total=n_batch, disable=not self._verbose) as t:
                for batch in next_batch_pairwise(data=self.edge_index,
                                                 num_items=self._num_items,
                                                 batch_size=self._batch_size,
                                                 ui_dict=self.ui_dict,
                                                 seed=self._seed):
                    steps += 1
                    current_loss = self._model.train_step(batch)
                    loss += current_loss

                    if math.isnan(loss) or math.isinf(loss) or (not loss):
                        break

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        self._model.eval()
        gu, gi = self._model.propagate_embeddings()
        self._model.train()
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(gu[offset: offset_stop], gi)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / (it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][
                    self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

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