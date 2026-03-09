from TALENT.model.classical_methods.ncm import NCMMethod
import os.path as ops
import pickle

class NaiveBayesMethod(NCMMethod):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)

    def construct_model(self, model_config = None):
        from sklearn.naive_bayes import GaussianNB
        self.model = GaussianNB()

    def predict(self, data, info, model_name):
        N, C, y = data
        with open(ops.join(self.args.save_path, 'best-val-{}.pkl'.format(self.args.seed)), 'rb') as f:
            self.model = pickle.load(f)
        self.data_format(False, N, C, y)
        test_label = self.y_test
        test_logit = self.model.predict_proba(self.N_test)
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        return vres, metric_name, test_logit
