# from lightgbm import LGBMClassifier
from lightgbm import plot_importance
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import shap

class MapCleanClassifier():
    def __init__(self):
        # self.model = LGBMClassifier( 
        #     unbalanced = True,
        #     max_depth=8,        # limit depth
        #     num_leaves=64,      # avoid too many leaves
        #     min_data_in_leaf=50,
        #     feature_fraction=0.8,
        #     bagging_fraction=0.8,
        #     bagging_freq=5,
        #     lambda_l1=1.0,      # L1 regularization
        #     lambda_l2=1.0,      # L2 regularization
        #     n_jobs=-1, 
        #     verbose=-1
        # )
        self.params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "is_unbalance": True,        # or use scale_pos_weight
                "max_depth": 8,
                "num_leaves": 64,
                "min_data_in_leaf": 50,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l1": 1.0,
                "lambda_l2": 1.0,
                "num_threads": -1,
                "verbose": -1,
                "learning_rate": 0.05
            }
        self.booster = None
        self.eval_results = None
        self.n_rounds = 25
        self.rounds_step = 5
        self.early_stopping_rounds = 5

    # def fit(self, X, y, **kwargs):
    #     return self.model.fit(X, y, **kwargs)

    def fit(self, X_train, y_train, X_val, y_val, X_test= None, y_test=None):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data   = lgb.Dataset(X_val,   label=y_val)
        
        data_list = [train_data, val_data]
        data_name_list = ["train", "val"]
        if X_test is not None: 
            test_data  = lgb.Dataset(X_test,  label=y_test)
            data_list.append(test_data)
            data_name_list.append("test")
            
        # --- to store the loss values per iteration ---
        self.eval_results = {}

        # --- train using native LightGBM API ---
        self.booster = lgb.train(
            params = self.params,
            train_set=train_data,
            valid_sets=data_list,
            valid_names=data_name_list,
            num_boost_round=self.n_rounds,
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(period=self.rounds_step),     # print every 10 iterations
                lgb.record_evaluation(self.eval_results)
            ]
        )

    
    def predict(self, X):
        # return self.model.predict(X)
        return (self.booster.predict(X) >= 0.5).astype(bool)
    
    def plot_loss(self):
        # --- plot the curves ---
        plt.figure(figsize=(10,6))
        plt.plot(self.eval_results["train"]["binary_logloss"], label="Train")
        plt.plot(self.eval_results["val"]["binary_logloss"], label="Validation")
        plt.plot(self.eval_results["test"]["binary_logloss"], label="Test")
        plt.xlabel("Iteration")
        plt.ylabel("Binary Logloss")
        plt.title("Training / Validation / Test Loss Curves")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_importance(self):
        # plot_importance(self.booster, max_num_features=30, importance_type='split')
        # plt.show()
        plot_importance(self.booster, max_num_features=30, importance_type='gain')
        plt.show()
    
    def plot_norm_importance(self):
        gain_importance = self.booster.feature_importance(importance_type='gain')
        feature_names = self.booster.feature_name()

        fi = pd.DataFrame({
            "feature": feature_names,
            "gain": gain_importance
        }).sort_values("gain", ascending=True)

        fi['perc'] = fi.gain/fi.gain.sum()
        plt.barh(fi['feature'], fi['perc'])
        plt.xlabel('Percentage Importance')
        plt.title('Feature Importance based on Gain')
        plt.show()
    
    def plot_shap(self, X):
        explainer = shap.TreeExplainer(self.booster)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, feature_names=X.columns)

        