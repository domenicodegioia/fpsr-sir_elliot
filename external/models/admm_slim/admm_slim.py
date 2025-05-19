import pickle
import time

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from .admm_slim_model import ADMMSlimModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation


class ADMMSlim(RecMixin, BaseRecommenderModel):
    r"""
    ADMM SLIM: Sparse Recommendations for Many Users

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3336191.3371774>`_

    Args:
        eigen_dim: Number of eigenvectors extracted

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        external.ADMMSlim:
          meta:
            verbose: True
          eigen_dim: 256

    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_l1", "l1", "l1", 0.001, float, None),
            ("_l2", "l2", "l2", 0.001, float, None),
            ("_alpha", "alpha", "alpha", 0.001, float, None),
            ("_rho", "rho", "rho", 100, int, None),
            ("_iterations", "iterations", "iterations", 50, int, None)
        ]

        self.autoset_params()

        # self._ratings = self._data.train_dict
        # self._sp_i_train = self._data.sp_i_train
        # self._i_items_set = list(range(self._num_items))

        self._model = ADMMSlimModel(self._data,
                                    self._num_users,
                                    self._num_items,
                                    self._l1,
                                    self._l2,
                                    self._alpha,
                                    self._iterations,
                                    self._rho)


    @property
    def name(self):
        return "ADMMSlim" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in self._data.train_dict.keys()}

    def get_recommendations(self, k: int = 10):
        self._model.prepare_predictions()

        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test


    # def predict(self, u: int, i: int):
    #     return self._model.predict(u, i)

    def train(self):
        if self._restore:
            return self.restore_weights()

        self.logger.info(f"Start training...")
        start = time.time()
        self._model.train()
        self.logger.info(f"The similarity computation has taken:\t{time.time() - start}")

        start = time.time()
        self.evaluate()
        self.logger.info(f"Evaluation has taken:\t{time.time() - start}")
