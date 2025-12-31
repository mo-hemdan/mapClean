import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)
import matplotlib.pyplot as plt
import json
import geopandas as gpd
# Three metrics:
# 1. Missing roads Detection Accuracy
# 2. Wrong Roads Percentage
# 3. Point Accuracy


class MapEvaluator:
    def __init__(self):
        self.new_wrong_roads = 0
        self.total_missing_roads = 0
        self.total_n_roads = 0
        self.total_n_matched_roads = 0
        self.results_dict = {
            "model_name": [],
            "train": {
                "acc": [],
                "precision": [],
                "recall": [],
                "f1_score": [],
                "balanced_accuracy": [],
                "accuracy_match": [],
                "accuracy_make": [],
            },
            "val": {
                "acc": [],
                "precision": [],
                "recall": [],
                "f1_score": [],
                "balanced_accuracy": [],
                "accuracy_match": [],
                "accuracy_make": [],
            },
            "test": {
                "acc": [],
                "precision": [],
                "recall": [],
                "f1_score": [],
                "balanced_accuracy": [],
                "accuracy_match": [],
                "accuracy_make": [],
                "overall_acc": [],
                # "miss_R_detec_acc": [],
                # "wrong_R_num": [],
                # "wrong_R_perc": [],
                # "road_mistakes": [], 
                # "wholeNetwork_acc": [],
                # "matchedRoadNetwork_acc": []
                'road_detec_acc': [],
                'new_wrong_roads': [],
                # 'n_wrong_R_per': [],
                'road_errors': [],
                'road_network_acc': [],
                'whole_network_acc': [],
                'f1_score_neg': [],
                'f1_score_pos': [],
                'precision_neg': [],
                'precision_pos': [],
                'recall_neg': [],
                'recall_pos': [],
                'internal_acc': [],
                'intern_make_percent': [],
                'uncert_make_percent': []
            },
        }

    def set_total_n_matched_roads(self, n):
        self.total_n_matched_roads = n
    
    # First: Missing road detection accuracy
    def get_missing_roads_detection_acc(self, matchedRoad_test, y_test, y_predict_n1_test, threshold = 0.05):
        # Road Coverage Metric: (How much P_make has been detected out of the total) 80% threshold
        
        # Getting missing roads and predicted ones
        missing_roads = matchedRoad_test[y_test] # y_test = 1 for map making
        missing_roads = np.unique(missing_roads)
        self.total_missing_roads = len(missing_roads)
        missing_roads, covered_point_counts = np.unique(matchedRoad_test[y_test], return_counts=True)
        pred_missing_roads, pred_covered_point_counts = np.unique(matchedRoad_test[y_test & y_predict_n1_test], return_counts=True)
        
        # Getting predicted points and whole points per road
        coverage_prec = dict()
        for r, c in zip(missing_roads, covered_point_counts):
            if r not in coverage_prec:
                coverage_prec[r] = [0, 0]
            coverage_prec[r][0] = c
        for r, c in zip(pred_missing_roads, pred_covered_point_counts):
            if r not in coverage_prec:
                coverage_prec[r] = [0, 0]
                print('Warning a predicted road not in missing ones')
            coverage_prec[r][1] = c
        
        # Getting the making points coverage per road
        coverage_prec_v = dict()
        for k in coverage_prec:
            val = coverage_prec[k]
            if k not in coverage_prec_v:
                coverage_prec_v[k] = 0
            coverage_prec_v[k] = val[1] /val[0]
        coverage_prec_v
        
        # Applying the threshold
        total_detected_roads = 0
        for p in coverage_prec_v.values():
            if p >= threshold:
                total_detected_roads += 1

        detection_acc = total_detected_roads / len(coverage_prec_v)
        
        return detection_acc, total_detected_roads
    

    def get_make_points(self, X_test, y_test, y_predict):
        false_make_points = X_test.loc[(~y_test) & (y_predict), ['lon', 'lat', 'angle', 'matching_score']]
        true_make_points = X_test.loc[(y_test) & (y_predict), ['lon', 'lat', 'angle', 'matching_score']]
        false_make_points = gpd.GeoDataFrame(false_make_points, geometry=gpd.points_from_xy(false_make_points.lon, false_make_points.lat), crs=3857)
        
        true_make_points  = gpd.GeoDataFrame(true_make_points,  geometry=gpd.points_from_xy(true_make_points.lon,  true_make_points.lat),  crs=3857)

        return false_make_points, true_make_points
    
    # Second: Wrong Roads Detection Accuracy
    # def get_new_wrong_roads(self, X_test_all, y_test, y_predict_n1_test, return_all=False):
    #     # Out of originally map matching points, get the map making ones, look up how many are there, cluster them using some cuml (DBSCAN using GPUs) to get number of clusters and then get the number as the new roads
    #     false_make_points, _ = self.get_make_points(X_test_all, y_test, y_predict_n1_test)
    #     false_make_points = false_make_points[['lon', 'lat']].to_numpy()
    #     # If no points, give just zero
    #     if len(false_make_points) == 0: 
    #         wrong_roads_make_perc = 0
    #         self.new_wrong_roads = 0
    #         if return_all:
    #             return wrong_roads_make_perc, [], []
    #         return wrong_roads_make_perc
            
    #     make_points = false_make_points[['lon', 'lat']].to_numpy()

    #     clusterer = HDBSCAN(
    #         min_cluster_size=2,
    #         core_dist_n_jobs=32
    #     )

    #     if len(make_points) >= 2:
    #         labels = clusterer.fit_predict(make_points)
    #     else:
    #         # Not enough points to cluster
    #         labels = [0] * len(make_points)
                    
    #     print("Clusters found:", len(np.unique(labels)))
    #     self.new_wrong_roads = len(np.unique(labels))
        
    #     new_wrong_road_prec = self.new_wrong_roads / self.total_missing_roads
        
    #     if return_all:
    #         return new_wrong_road_prec, make_points, labels
        
    #     return new_wrong_road_prec
    

    def get_road_accuracy(self, road_network, n_wrong_roads, matchedRoad_test, y_true, y_predict, threshold):
        '''
        Here you have three types of roads:
        (1) The originally matched roads (total_n_matched_roads)
        (2) The missing roads that we are trying to detect (total_missing_roads)
        (3) The detected missing roads (total_detected_roads)
        (4) The new wrong roads that we are mistakenly adding (n_wrong_roads)

        The total roads in the network is:
        total_n_roads = (1) + (2) + (4)
        total_error_roads = (2) - (3) + (4)

        per_r_correct = 1 - (total_error_roads / total_n_roads)
        '''


        # Getting Detected roads out of the missing ones
        road_detection_acc, total_detected_roads = self.get_missing_roads_detection_acc(matchedRoad_test, y_true, y_predict, threshold)   

        # we have total_missing_roads, total_detected_roads, n_wrong_roads, n_matched_roads

        total_n_roads = self.total_n_matched_roads + self.total_missing_roads + n_wrong_roads

        total_error_roads = (self.total_missing_roads - total_detected_roads) + n_wrong_roads

        road_network_acc = 1 - (total_error_roads / total_n_roads) 

        total_whole_n_roads = len(road_network.edges()) + self.total_missing_roads + n_wrong_roads
        whole_road_network_acc = 1 - (total_error_roads / (total_whole_n_roads))

        # wrong_roads_prec, points, labels = self.get_new_wrong_roads(X_test_all, y_test, y_predict_n1_test, return_all=True)
        # n_wrong_roads = len(np.unique(labels))
        
        # n_mistake_roads = self.total_missing_roads - total_detected_roads
        # n_mistake_roads += n_wrong_roads
        
        # new_total_roads = self.total_n_roads + n_wrong_roads
        # wholeRoadNetwork_acc = 1 - (n_mistake_roads / new_total_roads)
        
        # new_total_matched_roads = self.total_n_matched_roads + n_wrong_roads
        # matchedRoadNetwork_acc = 1 - (n_mistake_roads / new_total_matched_roads)
        
        self.results_dict['test']['road_detec_acc'].append(road_detection_acc)
        self.results_dict['test']['new_wrong_roads'].append(n_wrong_roads)
        # self.results_dict['test']['n_wrong_R_per'].append(wrong_roads_prec)
        self.results_dict['test']['road_errors'].append(total_error_roads)
        self.results_dict['test']['road_network_acc'].append(road_network_acc)
        self.results_dict['test']['whole_network_acc'].append(whole_road_network_acc)
        
        return road_network_acc, whole_road_network_acc
    
    def get_mokbel_accuracy(self, road_network, n_wrong_roads, matchedRoad_test, y_true, y_predict, threshold):
        '''
        Here you have three types of roads:
        (1) The originally matched roads (total_n_matched_roads)
        (2) The missing roads that we are trying to detect (total_missing_roads)
        (3) The detected missing roads (total_detected_roads)
        (4) The new wrong roads that we are mistakenly adding (n_wrong_roads)

        The total roads in the network is:
        total_n_roads = (1) + (2) + (4)
        total_error_roads = (2) - (3) + (4)

        per_r_correct = 1 - (total_error_roads / total_n_roads)
        '''
        road_detection_acc, total_detected_roads = self.get_missing_roads_detection_acc(matchedRoad_test, y_true, y_predict, threshold)   
        R_correct = total_detected_roads
        R_missing = self.total_missing_roads
        R_wrong = n_wrong_roads
        
        mokbel_acc = (R_correct / R_missing) * (R_correct / (R_correct + R_wrong))
        self.results_dict['test']['mokbel_acc'].append(mokbel_acc)

        total_n_roads = self.total_n_matched_roads + self.total_missing_roads + n_wrong_roads

        total_error_roads = (self.total_missing_roads - total_detected_roads) + n_wrong_roads

        road_network_acc = 1 - (total_error_roads / total_n_roads) 

        total_whole_n_roads = len(road_network.edges()) + self.total_missing_roads + n_wrong_roads
        whole_road_network_acc = 1 - (total_error_roads / (total_whole_n_roads))
        
        self.results_dict['test']['road_detec_acc'].append(road_detection_acc)
        self.results_dict['test']['new_wrong_roads'].append(n_wrong_roads)
        self.results_dict['test']['road_errors'].append(total_error_roads)
        self.results_dict['test']['road_network_acc'].append(road_network_acc)
        self.results_dict['test']['whole_network_acc'].append(whole_road_network_acc)
        
        return road_network_acc, whole_road_network_acc


    
    # def get_wholeRoadNetwork_accuracy_old(self, road_network, X_test_all, matchedRoad_test, y_test, y_predict_n1_test, threshold):
    #     self.total_n_roads = road_network.number_of_edges()
        
    #     road_detection_acc, total_detected_roads = self.get_missing_roads_detection_acc(matchedRoad_test, y_test, y_predict_n1_test, threshold)   
    #     wrong_roads_prec, points, labels = self.get_new_wrong_roads(X_test_all, y_test, y_predict_n1_test, return_all=True)
    #     n_wrong_roads = len(np.unique(labels))
        
    #     n_mistake_roads = self.total_missing_roads - total_detected_roads
    #     n_mistake_roads += n_wrong_roads
        
    #     new_total_roads = self.total_n_roads + n_wrong_roads
    #     wholeRoadNetwork_acc = 1 - (n_mistake_roads / new_total_roads)
        
    #     new_total_matched_roads = self.total_n_matched_roads + n_wrong_roads
    #     matchedRoadNetwork_acc = 1 - (n_mistake_roads / new_total_matched_roads)
        
    #     self.results_dict['test']['miss_R_detec_acc'].append(road_detection_acc)
    #     self.results_dict['test']['wrong_R_num'].append(n_wrong_roads)
    #     self.results_dict['test']['wrong_R_perc'].append(wrong_roads_prec)
    #     self.results_dict['test']['road_mistakes'].append(n_mistake_roads)
    #     self.results_dict['test']['wholeNetwork_acc'].append(wholeRoadNetwork_acc)
    #     self.results_dict['test']['matchedRoadNetwork_acc'].append(matchedRoadNetwork_acc)
        
    #     total_road_acc = ((road_detection_acc) + 1- wrong_roads_prec)/2
    #     return matchedRoadNetwork_acc, wholeRoadNetwork_acc, total_road_acc, road_detection_acc, wrong_roads_prec, points, labels
    
    
    # def get_road_accuracy(self, X_test_all, matchedRoad_test, y_test, y_predict_n1_test, threshold):
    #     # TODO: compute the accuracy according to the whole set of roads in the road network not only the one we have or missing or removed
    #     road_detection_acc, total_detected_roads = self.get_missing_roads_detection_acc(matchedRoad_test, y_test, y_predict_n1_test, threshold)   
    #     wrong_roads_prec, points, labels = self.get_new_wrong_roads(X_test_all, y_test, y_predict_n1_test, return_all=True)

    #     self.results_dict['test']['miss_R_detec_acc'].append(road_detection_acc)
    #     self.results_dict['test']['wrong_R_num'].append(len(np.unique(labels)))
    #     self.results_dict['test']['wrong_R_perc'].append(wrong_roads_prec)
        
    #     total_road_acc = ((road_detection_acc) + 1- wrong_roads_prec)/2
    #     return total_road_acc, road_detection_acc, wrong_roads_prec, points, labels
    
    def compute_overall_acc(self, y_pred_train, y_pred_val, y_pred_test, y_true_train, y_true_val, y_true_test):
        y_pred_all = np.hstack([y_pred_train, y_pred_val, y_pred_test])
        y_true_all = np.hstack([y_true_train, y_true_val, y_true_test])

        y_pred_intern = np.hstack([y_pred_train, y_pred_val])
        y_true_intern = np.hstack([y_true_train, y_true_val])

        acc = (y_pred_all == y_true_all).mean()
        self.results_dict["test"]["overall_acc"].append(acc)


        # F1 (you already have this)
        f1_neg, f1_pos = f1_score(y_true_all, y_pred_all, average=None)

        # Precision per class
        prec_neg, prec_pos = precision_score(y_true_all, y_pred_all, average=None)

        # Recall per class
        rec_neg, rec_pos = recall_score(y_true_all, y_pred_all, average=None)

        # Store results
        self.results_dict["test"]["f1_score_neg"].append(f1_neg)
        self.results_dict["test"]["f1_score_pos"].append(f1_pos)

        self.results_dict["test"]["precision_neg"].append(prec_neg)
        self.results_dict["test"]["precision_pos"].append(prec_pos)

        self.results_dict["test"]["recall_neg"].append(rec_neg)
        self.results_dict["test"]["recall_pos"].append(rec_pos)

        internal_acc = (y_pred_intern == y_true_intern).mean()
        self.results_dict["test"]["internal_acc"].append(internal_acc)

        intern_make_percent = float(y_true_intern.mean())
        uncert_make_percent = float(y_true_test.mean())

        self.results_dict["test"]["intern_make_percent"].append(intern_make_percent)
        self.results_dict["test"]["uncert_make_percent"].append(uncert_make_percent)


        print(f'IMP Metrics: F1(match): {f1_neg:.2f}, F1(make): {f1_pos:.2f}, Pre(match):{prec_neg:.2f}, Pre(make): {prec_pos:.2f}, Recall(match): {rec_neg:.2f}, Recall(make): {rec_pos:.2f}')
        print(f'Second Metrics:  internalACC:{internal_acc:.2f}, Per Internal:{intern_make_percent:.2f}, Uncertain:{uncert_make_percent:.2f}')

        return acc
    
    def get_metrics(self, y_true, y_pred, split_type):
        acc = accuracy_score(y_true, y_pred)
        precision = float(precision_score(y_true, y_pred))  # precision tp / (tp + fp)
        recall = float(recall_score(y_true, y_pred))  # recall: tp / (tp + fn)
        f1 = float(f1_score(y_true, y_pred))
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        # Mask for Matching Points
        mask_0 = y_true == False
        accuracy_match = np.mean(y_pred[mask_0] == False)

        # Mask for Making Points
        mask_1 = y_true == True
        accuracy_make = np.mean(y_pred[mask_1] == True)
        print(
            f"{split_type} Acc: {100*acc:.2f}%, BalancedAcc: {100*balanced_accuracy:.2f}%, "
            f" (Match: {100*accuracy_match:.2f}%, Make: {100*accuracy_make:.2f})%, ",
            f"Percision: {100*precision:.2f}%, Recall: {100*recall:.2f}%"
            )
        if self.results_dict is not None:
            self.results_dict[split_type]["acc"].append(acc)
            self.results_dict[split_type]["precision"].append(precision)
            self.results_dict[split_type]["recall"].append(recall)
            self.results_dict[split_type]["f1_score"].append(f1)
            self.results_dict[split_type]["balanced_accuracy"].append(balanced_accuracy)
            self.results_dict[split_type]["accuracy_match"].append(accuracy_match)
            self.results_dict[split_type]["accuracy_make"].append(accuracy_make)
        return acc, precision, recall, f1, balanced_accuracy
    
    def add_model(self, model_name):
        self.results_dict["model_name"].append(model_name)
        
    def plot_acc_metrics(self):
        plt.bar(self.results_dict["model_name"], self.results_dict["test"]["precision"])
        plt.xlabel("Technique")
        plt.ylabel("Percision")
        plt.xticks(rotation=45)
        plt.show()

        plt.bar(self.results_dict["model_name"], self.results_dict["test"]["recall"])
        plt.xlabel("Technique")
        plt.ylabel("Recall")
        plt.xticks(rotation=45)
        plt.show()

        plt.bar(self.results_dict["model_name"], self.results_dict["test"]["f1_score"])
        plt.xlabel("Technique")
        plt.ylabel("F1 Score")
        plt.xticks(rotation=45)
        plt.show()

        plt.bar(self.results_dict["model_name"], self.results_dict["test"]["balanced_accuracy"])
        plt.xlabel("Technique")
        plt.ylabel("Balanced Accuracy")
        plt.xticks(rotation=45)
        plt.show()

        plt.bar(self.results_dict["model_name"], self.results_dict["test"]["accuracy_match"])
        plt.xlabel("Technique")
        plt.ylabel("Accuracy of Match")
        plt.xticks(rotation=45)
        plt.show()

        plt.bar(self.results_dict["model_name"], self.results_dict["test"]["accuracy_make"])
        plt.xlabel("Technique")
        plt.ylabel("Accuracy of Make")
        plt.xticks(rotation=45)
        plt.show()
        
    def save_results(self, filename):
        with open(filename, "w") as f:
            json.dump(self.results_dict, f, indent=4)  # indent is optional, for pretty printing


