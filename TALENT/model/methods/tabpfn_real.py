from TALENT.model.methods.tabpfn_v2 import TabPFNMethod


class TabPFNRealMethod(TabPFNMethod):
    def construct_model(self, model_config = None,cat_indices=[]):
        if self.is_regression:
            raise ValueError("TabPFN-Real only supports classification tasks.")
        else:
            from TALENT.model.models.tabpfn_v2 import TabPFNClassifier
            self.model = TabPFNClassifier(
                model_path = "./TALENT/model/models/models_tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
                device = self.args.device,
                random_state = self.args.seed,
                n_estimators = 4,
                ignore_pretraining_limits = True,
                categorical_features_indices = cat_indices
            )