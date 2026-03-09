from TALENT.model.classical_methods.base import classical_methods
import os.path as ops
import pickle
import time
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class NCMMethod(classical_methods):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(not is_regression)
        assert(args.cat_policy != 'indices')
        assert(not args.tune)

    def construct_model(self, model_config = None):
        from sklearn.neighbors import NearestCentroid
        self.model = NearestCentroid()

    def fit(self, data, info, train=True, config=None):
        super().fit(data, info, train, config)
        if not train:
            return
        tic = time.time()
        self.model.fit(self.N['train'], self.y['train'])
        self.trlog['best_res'] = self.model.score(self.N['val'], self.y['val'])
        time_cost = time.time() - tic
        with open(ops.join(self.args.save_path, 'best-val-{}.pkl'.format(self.args.seed)), 'wb') as f:
            pickle.dump(self.model, f)
        return time_cost

    def _predict_proba(self, X):
        distances = euclidean_distances(X, self.model.centroids_)
        neg_distances = -distances
        exp_neg_dist = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
        probabilities = exp_neg_dist / np.sum(exp_neg_dist, axis=1, keepdims=True)
        return probabilities

    def predict(self, data, info, model_name):
        N, C, y = data
        with open(ops.join(self.args.save_path, 'best-val-{}.pkl'.format(self.args.seed)), 'rb') as f:
            self.model = pickle.load(f)
        self.data_format(False, N, C, y)
        test_label = self.y_test
        test_logit = self._predict_proba(self.N_test)
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        return vres, metric_name, test_logit
