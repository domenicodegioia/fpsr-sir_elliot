import numpy as np
import random
import time


class Sampler:
    def __init__(self, indexed_ratings, m, sparse_matrix, seed):
        np.random.seed(seed)
        random.seed(seed)

        ratings = sparse_matrix.nonzero()
        self.rating_users = ratings[0]
        self.rating_items = ratings[1]

        self._users, self.idx_start, self.count = np.unique(self.rating_users, return_counts=True, return_index=True)

        #self._sparse = sparse_matrix
        # self._indexed_ratings = indexed_ratings
        # self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = np.unique(self.rating_items)
        self._nitems = len(self._items)
        # self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        # self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self._m = m
        # self._nonzero = self._sparse.nonzero()
        # self._num_pos_examples = len(self._nonzero[0])
        # self._positive_pairs = list(zip(*self._nonzero, np.ones(len(self._nonzero[0]), dtype=np.int32)))

    # def step(self, batch_size):
    #     """Converts a list of positive pairs into a two class dataset.
    #     Args:
    #       positive_pairs: an array of shape [n, 2], each row representing a positive
    #         user-item pair.
    #       num_negatives: the number of negative items to sample for each positive.
    #     Returns:
    #       An array of shape [n*(1 + num_negatives), 3], where each row is a tuple
    #       (user, item, label). The examples are obtained as follows:
    #       To each (user, item) pair in positive_pairs correspond:
    #       * one positive example (user, item, 1)
    #       * num_negatives negative examples (user, item', 0) where item' is sampled
    #         uniformly at random.
    #     """
    #     time_start = time.time()
    #
    #     def user_training_matrix(u):
    #         pos_u = self.rating_items[self.idx_start[u]:self.idx_start[u] + self.count[u]]
    #         neg_u = np.setdiff1d(np.array(self._items), pos_u, assume_unique=True)
    #         sampled_neg_u = np.random.choice(neg_u, self._m * len(pos_u), replace=True)
    #         return np.c_[np.repeat(u, len(pos_u) + len(sampled_neg_u)), np.r_[
    #             np.c_[pos_u, np.ones(len(pos_u), dtype=int)], np.c_[
    #                 sampled_neg_u, np.zeros(len(sampled_neg_u), dtype=int)]]]
    #
    #     training_matrix = np.concatenate([user_training_matrix(u) for u in self._users])
    #
    #     samples_indices = random.sample(range(training_matrix.shape[0]), training_matrix.shape[0])
    #     training_matrix = training_matrix[samples_indices]
    #     # print(f"Sampling has taken {round(time.time() - time_start, 2)} seconds")
    #     for start in range(0, training_matrix.shape[0], batch_size):
    #         yield training_matrix[start:min(start + batch_size, training_matrix.shape[0])]

    def step(self, batch_size):
        training_data = []

        for idx, u in enumerate(self._users):
            start_idx = self.idx_start[idx]
            end_idx = start_idx + self.count[idx]
            pos_items = self.rating_items[start_idx:end_idx]
            num_pos = len(pos_items)

            # Sample negative items
            neg_items = np.setdiff1d(self._items, pos_items, assume_unique=True)
            sampled_neg_items = np.random.choice(neg_items, self._m * num_pos, replace=True)

            # Construct training examples: [user, item, label]
            user_pos = np.column_stack((np.full(num_pos, u), pos_items, np.ones(num_pos, dtype=np.int32)))
            user_neg = np.column_stack((np.full(len(sampled_neg_items), u), sampled_neg_items,
                                        np.zeros(len(sampled_neg_items), dtype=np.int32)))

            training_data.append(np.vstack((user_pos, user_neg)))

        training_matrix = np.vstack(training_data)

        # Shuffle in-place for better memory efficiency
        np.random.shuffle(training_matrix)

        # Yield batches
        for start in range(0, training_matrix.shape[0], batch_size):
            yield training_matrix[start:start + batch_size]
