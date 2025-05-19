import pickle
from tqdm import tqdm
import time

import torch

from .custom_sampler_rendle import Sampler
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from .MF_model import MFModel


class MF2020(RecMixin, BaseRecommenderModel):
    r"""
    Matrix Factorization (implementation from "Neural Collaborative Filtering vs. Matrix Factorization Revisited")

    For further details, please refer to the `paper <https://dl.acm.org/doi/pdf/10.1145/3383313.3412488>`_

    Args:


    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        external.MF2020:
          meta:
            save_recs: True
          epochs: 10
          factors: 10
          lr: 0.001
          reg: 0.0025
          m: 3
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_factors", "factors", "f", 10, int, None),
            ("_learning_rate", "lr", "lr", 0.05, None, None),
            ("_regularization", "reg", "reg", 0, None, None),
            ("_m", "m", "m", 0, int, None),
        ]
        self.autoset_params()

        self._sampler = Sampler(self._data.i_train_dict, self._m, self._data.sp_i_train, self._seed)

        # This is not a real batch size. Its only purpose is the live visualization of the training
        self._batch_size = 40000

        self._model = MFModel(self._factors,
                              self._data,
                              self._learning_rate,
                              self._regularization,
                              self._seed)

    def get_recommendations(self, k: int = 100):
        self._model.prepare_predictions()
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        with torch.no_grad():
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(offset, offset_stop)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    @property
    def name(self):
        return "MF2020" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions * (self._m + 1) // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)/len(batch)
                    t.set_postfix({'loss': f'{loss/steps:.5f}'})
                    t.update()

            self.evaluate(it, loss/(it + 1))
