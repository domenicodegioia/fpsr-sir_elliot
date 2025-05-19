import pickle
import numpy as np
import torch
import random

class MFModel(object):
    def __init__(self,
                 factors,
                 data,
                 lr,
                 reg,
                 random_seed,
                 *args):
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._factors = factors
        self._users = data.users
        self._items = data.items
        self._num_users = data.num_users
        self._num_items = data.num_items
        self._lr = lr
        self._reg = reg

        self._preds = None

        self._global_bias = torch.tensor(0.0).to(self.device)
        self._user_bias = torch.zeros(self._num_users).to(self.device)
        self._item_bias = torch.zeros(self._num_items).to(self.device)
        self._user_factors = torch.normal(mean=0.0, std=0.1, size=(self._num_users, self._factors)).to(self.device)
        self._item_factors = torch.normal(mean=0.0, std=0.1, size=(self._num_items, self._factors)).to(self.device)

    @property
    def name(self):
        return "MF2020"

    def train_step(self, batch, **kwargs):
        total_loss = 0
        users, items, ratings = batch[:, 0], batch[:, 1], batch[:, 2]
        ratings = torch.tensor(ratings, device=self.device)

        gb_ = self._global_bias
        uf_ = self._user_factors[users]
        if_ = self._item_factors[items]
        ub_ = self._user_bias[users]
        ib_ = self._item_bias[items]

        preds = gb_ + ub_ + ib_ + torch.sum(uf_ * if_, dim=1)

        sigmoid = torch.sigmoid(preds)
        loss = torch.sum((torch.log(1 + torch.exp(preds)) + (1.0 - ratings) * preds) * (ratings > 0))

        grad = ratings - sigmoid

        self._user_factors[users] += self._lr * (grad.view(-1, 1) * if_ - self._reg * uf_)
        self._item_factors[items] += self._lr * (grad.view(-1, 1) * uf_ - self._reg * if_)
        self._user_bias[users] += self._lr * (grad - self._reg * ub_)
        self._item_bias[items] += self._lr * (grad - self._reg * ib_)
        self._global_bias += self._lr * (grad.mean() - self._reg * gb_)

        total_loss += loss

        return total_loss

    def prepare_predictions(self):
        # self._preds = self._user_bias.unsqueeze(1) + (
        #             self._global_bias + self._item_bias + torch.matmul(self._user_factors, self._item_factors.T))
        self._preds = self._user_bias.unsqueeze(1).to(torch.float16) + (
                self._global_bias.to(torch.float16) +
                self._item_bias.to(torch.float16) +
                torch.matmul(self._user_factors.to(torch.float16), self._item_factors.T.to(torch.float16))
        )
        # ####### check sparsity of self._preds #######
        # preds_cpu = self._preds.cpu()
        # non_zero_count = torch.count_nonzero(preds_cpu)
        # total_elements = self._preds.numel()
        # percentage_non_zero = (non_zero_count.item() / total_elements) * 100
        # print(f"Percentuale di valori non zero: {percentage_non_zero:.2f}%")

    def predict(self, offset, offset_stop):
        users = torch.arange(offset, offset_stop).to(self.device)
        user_factors = self._user_factors[users]
        item_factors = self._item_factors.T
        predictions = self._user_bias[users].unsqueeze(1) + self._global_bias + self._item_bias + torch.matmul(
            user_factors, item_factors)
        return predictions

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), torch.tensor(preds).to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
