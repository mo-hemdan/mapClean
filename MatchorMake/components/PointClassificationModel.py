import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, log_loss
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l2
from tqdm.keras import TqdmCallback
import pandas as pd
import pickle

class RandomClassifier():
    def __init__(self):
        self.n = 0
        self.true_p, self.false_p = None, None
    
    def fit(self, X_train, y_train):
        self.n = len(y_train)
        self.true_p = np.sum(y_train == True) / self.n
        self.false_p = np.sum(y_train == False) / self.n

    def predict(self, X_test):
        length = len(X_test)
        return np.random.choice([False, True], length, p=[self.false_p, self.true_p])
    
    def predict_proba(self, X_test):
        length = len(X_test)
        return np.full((length,), self.true_p)


class PointClassificationModels():
    def __init__(self):
        self.scaler = None
        self.models = {}
    
    def initialize_models(self, n_features):
        self.models['Random'] =                 RandomClassifier()
        self.models['Logistic Regression'] =    LogisticRegression(random_state=0)
        self.models['Decision Tree'] =          DecisionTreeClassifier()
        self.models['Random Forest'] =          RandomForestClassifier(n_estimators = 100)  
        self.models['FNN'] =                    Sequential([
                                                    Input(shape=(n_features,)),
                                                    # keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), input_shape=(p,)),
                                                    # keras.layers.Dense(8, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                                                    Dense(4, activation='relu', kernel_regularizer=l2(0.001)),
                                                    Dense(1, activation='sigmoid')
                                                ])
        self.models['FNN'].compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        
    def adjust_scaler(self, X_train):
        self.scaler = StandardScaler().fit(X_train)
    
    def fit(self, X_train, y_train):
        
        X_train_scaled = self.scaler.transform(X_train)
        for classifier_name in self.models.keys():
            print("Running " + classifier_name + "...")
            if classifier_name[:3] != 'FNN': self.models[classifier_name].fit(X_train, y_train)
            else: self.models[classifier_name].fit(X_train_scaled.astype(np.float32), y_train, 
                                                   verbose=0, 
                                                   epochs=3, 
                                                   batch_size=32, 
                                                   validation_split=0.2, 
                                                   callbacks=[TqdmCallback(verbose=1)],
                                                   shuffle=True)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)  
    
    def post_process(self, X_test, predict_df, prefix='', threshold = 9):
        for classifier_name in self.models.keys():
            predict_df.loc[X_test[prefix + 'distance_to_matched_road'] > threshold, classifier_name+'Proba'] = 0.25
            predict_df.loc[X_test[prefix + 'distance_to_matched_road'] > threshold, classifier_name] = False
        return predict_df
    
    def predict(self, X_test, apply_post_process=False):
        X_test_scaled = self.scaler.transform(X_test)
        self.data_frame_created = False
        for classifier_name in self.models.keys():
            print("Running " + classifier_name + "...")
            if classifier_name[:3] != 'FNN': 
                y_predict_proba = self.models[classifier_name].predict_proba(X_test)
                if y_predict_proba.ndim > 1: y_predict_proba = y_predict_proba[:,1] #TODO: There is a difference between models and NN models
                predict_df = pd.DataFrame({
                    classifier_name:      self.models[classifier_name].predict(X_test), 
                    classifier_name+'Proba': y_predict_proba
                    })
            else:
                y_predict_proba = self.models[classifier_name].predict(X_test_scaled.astype(np.float32))
                y_predict = y_predict_proba > 0.5
                predict_df = pd.DataFrame({
                    classifier_name:      y_predict[:, 0], 
                    classifier_name+'Proba': y_predict_proba[:, 0]
                    })
                        
            if not self.data_frame_created: 
                df = predict_df
                self.data_frame_created = True
            else:                           df = pd.concat([df, predict_df], axis=1)
        
        df.index = X_test.index
        
        if apply_post_process: df = self.post_process(X_test, df)

        return df
    
    def evaluate(self, X_test, y_test, apply_post_process=False):
        
        predict_df = self.predict(X_test, apply_post_process)
        
        results = dict()
        for classifier_name in self.models.keys():
            y_predict = predict_df[classifier_name]
            y_predict_proba = predict_df[classifier_name+'Proba']
            
            results[classifier_name] = dict()
            results[classifier_name]['Acc'] = accuracy_score(y_test, y_predict) # accuracy: (tp + tn) / (p + n)
            results[classifier_name]['Percision'] = float(precision_score(y_test, y_predict)) # precision tp / (tp + fp)
            results[classifier_name]['Recall'] = float(recall_score(y_test, y_predict)) # recall: tp / (tp + fn)
            results[classifier_name]['F1 score'] = float(f1_score(y_test, y_predict)) # f1: 2 tp / (2 tp + fp + fn)
            results[classifier_name]['Cohens Kappa'] = float(cohen_kappa_score(y_test, y_predict))
            results[classifier_name]['ROC AUC'] = float(roc_auc_score(y_test, y_predict))
            results[classifier_name]['Loss'] = log_loss(y_test, y_predict_proba)

            tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
            results[classifier_name] |= {  # Update Operator
                    'Matching Acc': float(tp/(tp+fn))*100, 
                    'Making Acc': float(tn/(tn+fp))*100, 
                    'tp': int(tp), # 'TruePositive (match->match)': tp,
                    'fn': int(fn), # 'FalsePositive (match->make)': fn,
                    'tn': int(tn), # 'TrueNegative (make->make)': tn,
                    'fp': int(fp) # 'False Negative (make->match)': fp
                    }
        return results

    def print_results(self, results):
        for classifier_name in results.keys():
            print('Classifier: ', classifier_name)
            for metric in results[classifier_name].keys():
                print(f'       {metric}:{results[classifier_name][metric]}')