
class RuleCleaner:
    def __init__(self):
        pass
    
    # Algorithm: Rule-based techniques based on distance to matched road
    def predict(self, X_test_all, threshold):
        return (X_test_all.matching_score >= threshold)
        
    def find_best_threshold(self, X_test_all, y_test, threshold_list):
        
        acc_list = []
        for threshold in threshold_list:
            X_test_all['making'] = (X_test_all.matching_score >= threshold)
            test_acc = (y_test == X_test_all.making).sum() / len(X_test_all)
            print(threshold, ': ', test_acc)
            acc_list.append(test_acc)
        return acc_list
    
    