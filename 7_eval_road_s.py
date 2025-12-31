# %%
import pandas as pd
import numpy as np
import joblib
import os
import pickle
from sklearn.model_selection import train_test_split
from load_config import load_config
from MapCleanClassifier import MapCleanClassifier
from MapEvaluator import MapEvaluator
from RuleCleaner import RuleCleaner
from MapMaker import MapMaker
from sklearn.preprocessing import StandardScaler

# config = load_config('configs/jakarta_m.json', same_gamma=False)
config = load_config(same_gamma=False)
# config = load_config('configs/jakarta_m.json', same_gamma=False)

# Constants
THRESHOLD = 0.5
RULE_CLEANER_THRESHOLD = 10

# Filenames
FEATURES_FILENAME = f"city_{config['CITY']}_pro/\
featureEx_d{config['D']}n{config['N']}_SERg{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.csv"

ROADNETWORK_FILENAME = f"city_{config['CITY']}_pro/graph_g{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl"

#%% Load road network & features
with open(ROADNETWORK_FILENAME, 'rb') as file:
    road_network = pickle.load(file)
df = pd.read_csv(FEATURES_FILENAME)
# %%
NPOINTDF_FILENAME = f"city_{config['CITY']}_pro/nPointsDf_g{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.csv"
nPoints_df = pd.read_csv(NPOINTDF_FILENAME)

ORIGINAL_FILENAME = f"city_{config['CITY']}_pro/OriginalDFs_g{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl"
gdf_original_dfs = pd.read_pickle(ORIGINAL_FILENAME)

# %%
nPoints_df.drop(columns=['Unnamed: 0'], inplace=True)
gdf_original_dfs = gdf_original_dfs.to_frame()

new_df = pd.concat([df, nPoints_df, gdf_original_dfs], axis=1)

new_df.drop(columns=['nPoints'], inplace=True)

df = new_df

# %% Preprocessing the input GPS Dataframe
print('Preprocessing input GPS dataframe')
# df = pd.concat([df, ture_matched_road_id], axis=1)
df = df.sample(frac=1, random_state=42)
df = df.reset_index(drop=True)

# %%
# df.nMatchPoints = df.nMatchPoints.astype(int)
# df.nMakePoints = df.nMakePoints.astype(int)

# %%
o_df = df.original_dataframe
df_n = df.drop(columns=['original_dataframe'])

# %%
df_n = df_n.loc[df.index.repeat(df_n['N'])].reset_index(drop=True)
df_n
# %%
o_df_exp = pd.concat(o_df.to_list(), ignore_index=True)

# %%
o_df_exp = o_df_exp[['ture_matched_road_id', 'ture_road_geometry', 
            'ture_distance_to_matched_road', 'ture_toBeMatched', 
            'Perfect', 'ture_isAddedNoise', 'matched_road_id', 
            'distance_to_matched_road', 'road_geometry']]

# %%
df_all = pd.concat([df_n, o_df_exp], axis=1)

# %% Getting Train Test Val
print('Getting Train Val Test ...')
X_train = df_all[(df_all.type == "good") | (df_all.type == "bad")].copy()
X_test = df_all[df_all.type == "uncertain"].copy()

y_train = np.full(len(X_train), False, dtype=bool)
y_train[X_train.type == "good"] = True

# Sample 20% from X_train and y_train as validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    train_size=0.8,
    stratify=y_train,  # Optional: keep class balance
    random_state=42,  # For reproducibility
)
X_test = X_test.sample(frac=1, random_state=42)
# %%
y_test = ~X_test['ture_toBeMatched']
y_train_true = ~X_train['ture_toBeMatched']
y_val_true = ~X_val['ture_toBeMatched']
y_pred_train = np.full(len(y_train), False)
y_pred_val = np.full(len(y_val), False)
# %%
matchedRoad_test = X_test.ture_matched_road_id.to_numpy()
matchedRoad_train = X_train.ture_matched_road_id.to_numpy()
matchedRoad_val = X_val.ture_matched_road_id.to_numpy()

# %%
X_test.rename(columns={'matched_road_dest': 'matched_road_dst'}, inplace=True)
X_train.rename(columns={'matched_road_dest': 'matched_road_dst'}, inplace=True)
X_val.rename(columns={'matched_road_dest': 'matched_road_dst'}, inplace=True)

# %%

# X_test.drop(columns=["ture_toBeMatched", "type"], inplace=True)
# X_train.drop(columns=["ture_toBeMatched", "type"], inplace=True)
# X_val.drop(columns=["ture_toBeMatched", "type"], inplace=True)

# X_train = X_train.to_numpy()
# X_val = X_val.to_numpy()
# X_test = X_test.to_numpy()

# The first 7 are the point features
# The second 7 are the unweighted nearby features
# The third 7 are the weighted nearby features
# The last column is the repeat count

# # Extract the repeat counts (e.g., from the last column)
# repeat_counts = X_train[:, -2].astype(int)
# X_train = np.repeat(X_train, repeat_counts, axis=0)
# y_train = np.repeat(y_train, repeat_counts)
# y_train_true = np.repeat(y_train_true, repeat_counts)
# repeat_counts = X_val[:, -2].astype(int)
# X_val = np.repeat(X_val, repeat_counts, axis=0)
# y_val = np.repeat(y_val, repeat_counts)
# y_val_true = np.repeat(y_val_true, repeat_counts)
# repeat_counts = X_test[:, -2].astype(int)
# X_test = np.repeat(X_test, repeat_counts, axis=0)
# y_test = np.repeat(y_test, repeat_counts)

# Trim the last column
# matchedRoad_train = X_train[:, -1]
# matchedRoad_val = X_val[:, -1]
# matchedRoad_test = X_test[:, -1]
# X_train = X_train[:, :-2].astype(float)
# X_val = X_val[:, :-2].astype(float)
# X_test = X_test[:, :-2].astype(float)

# perm = np.random.permutation(len(X_train))
# X_train = X_train[perm]
# y_train = y_train[perm]
# y_train_true = y_train_true[perm]
# matchedRoad_train = matchedRoad_train[perm]
# perm = np.random.permutation(len(X_val))
# X_val = X_val[perm]
# y_val = y_val[perm]
# y_val_true = y_val_true[perm]
# matchedRoad_val = matchedRoad_val[perm]
# perm = np.random.permutation(len(X_test))
# X_test = X_test[perm]
# y_test = y_test[perm]
# matchedRoad_test = matchedRoad_test[perm]



col_names = [
    "lon",
    "lat",
    "speed",
    "angle",
    "matching_score",
    "matched_road_src",
    "matched_road_dst",
    "matched_road_angle",
    "r_p_sim",
    "W_c",
    "W_s",
    "W_ms",
    "W_as",
    "W_ss",
    "W_r",
    "W_Dir",
    "N_c",
    "N_s",
    "N_ms",
    "N_as",
    "N_ss",
    "N_r",
    "N_Dir",
]
X_train_all = X_train[col_names]
X_val_all = X_val[col_names]
X_test_all = X_test[col_names]

# %% Intialize the Evaluator
print('Initializing Map Evaluator...')
evaluator = MapEvaluator()
total_n_of_matched_roads = len(np.unique(np.concatenate([matchedRoad_train, matchedRoad_val, matchedRoad_test])))
evaluator.set_total_n_matched_roads(total_n_of_matched_roads)

print('Initializing Map Maker as well...')
map_maker = MapMaker(road_network)
# %% Rule Cleaning
print('Rule Cleaning')

rule_cleaner = RuleCleaner()
evaluator.add_model('Rule Cleaner')

y_predict_rule_test = rule_cleaner.predict(X_test_all, RULE_CLEANER_THRESHOLD)

evaluator.get_metrics(y_test, y_predict_rule_test, "test")

# false_make_points, true_make_points = evaluator.get_make_points(X_test_all, y_test, y_predict_rule_test)
# new_wrong_roads, n_new_wrong_roads = map_maker.infer_roads(false_make_points, skip_snapping=True)
# road_network_acc, whole_road_network_acc = evaluator.get_road_accuracy(road_network, n_new_wrong_roads, matchedRoad_test, y_test, y_predict_rule_test, threshold=THRESHOLD)
# print(f'Road Acc: {road_network_acc}, Whole Road Network Acc: {whole_road_network_acc}')

# %%
# MatchedRN_acc, wholeRN_acc, road_acc, road_detection_acc, wrong_roads_prec, points, labels = evaluator.get_wholeRoadNetwork_accuracy(road_network, X_test_all, matchedRoad_test, y_test, y_predict, threshold=THRESHOLD)
# print('Road Acc: ', wholeRN_acc, road_acc, ', RDetectionAcc: ', road_detection_acc, ', WrongPrec: ', wrong_roads_prec)

y_predict_rule_train = rule_cleaner.predict(X_train_all, RULE_CLEANER_THRESHOLD)
y_predict_rule_val = rule_cleaner.predict(X_val_all, RULE_CLEANER_THRESHOLD)

acc = evaluator.compute_overall_acc(y_predict_rule_train, y_predict_rule_val, y_predict_rule_test, y_train_true, y_val_true, y_test)
print(f'Overall Accuracy: {100*acc:.2f}')

# %%
# MAP MAKING
evaluator.add_model("Map Making")
make_all_predict_test = np.array(len(y_test) * [True])
make_all_predict_train = np.array(len(y_train) * [True])
make_all_predict_val = np.array(len(y_val) * [True])

evaluator.get_metrics(y_train, make_all_predict_train, "train")
evaluator.get_metrics(y_val, make_all_predict_val, "val")
evaluator.get_metrics(y_test, make_all_predict_test, "test")

# MatchedRN_acc, wholeRN_acc, road_acc, road_detection_acc, wrong_roads_prec, points, labels = evaluator.get_wholeRoadNetwork_accuracy(road_network, X_test_all, matchedRoad_test, y_test, make_all_predict_test, threshold=THRESHOLD)
# print('Road Acc: ', MatchedRN_acc, wholeRN_acc, road_acc, ', RDetectionAcc: ', road_detection_acc, ', WrongPrec: ', wrong_roads_prec)

# false_make_points, true_make_points = evaluator.get_make_points(X_test_all, y_test, make_all_predict_test)
# new_wrong_roads, n_new_wrong_roads = map_maker.infer_roads(false_make_points, skip_snapping=True)
# road_network_acc, whole_road_network_acc = evaluator.get_road_accuracy(road_network, n_new_wrong_roads, matchedRoad_test, y_test, make_all_predict_test, threshold=THRESHOLD)
# print(f'Road Acc: {road_network_acc}, Whole Road Network Acc: {whole_road_network_acc}')

acc = evaluator.compute_overall_acc(y_pred_train, y_pred_val, make_all_predict_test, y_train_true, y_val_true, y_test)
print('Overall Accuracy: ', acc)

# %%
# MAP MATCHING
evaluator.add_model("Map Matching")
match_predict_test = np.array(len(y_test) * [False])
match_predict_train = np.array(len(y_train) * [False])
match_predict_val = np.array(len(y_val) * [False])

evaluator.get_metrics(y_train, match_predict_train, "train")
evaluator.get_metrics(y_val, match_predict_val, "val")
evaluator.get_metrics(y_test, match_predict_test, "test")

# MatchedRN_acc, wholeRN_acc, road_acc, road_detection_acc, wrong_roads_prec, points, labels = evaluator.get_wholeRoadNetwork_accuracy(road_network, X_test_all, matchedRoad_test, y_test, match_predict_test, threshold=THRESHOLD)
# print('Road Acc: ', MatchedRN_acc, wholeRN_acc, road_acc, ', RDetectionAcc: ', road_detection_acc, ', WrongPrec: ', wrong_roads_prec)

# false_make_points, true_make_points = evaluator.get_make_points(X_test_all, y_test, match_predict_test)
# new_wrong_roads, n_new_wrong_roads = map_maker.infer_roads(false_make_points, skip_snapping=True)
# road_network_acc, whole_road_network_acc = evaluator.get_road_accuracy(road_network, n_new_wrong_roads, matchedRoad_test, y_test, match_predict_test, threshold=THRESHOLD)
# print(f'Road Acc: {road_network_acc}, Whole Road Network Acc: {whole_road_network_acc}')

acc = evaluator.compute_overall_acc(y_pred_train, y_pred_val, match_predict_test, y_train_true, y_val_true, y_test)
print('Overall Accuracy: ', acc)

#%%
drop_columns = ['lat', 'lon', 'matched_road_dst', 'matched_road_src', 'matched_road_angle', 'angle']

def compute_r_p_sim(df):
    a = df['angle'].values
    b = df['matched_road_angle'].values
    # X_train_all['r_p_sim'] = (a - b + 180) % 360 - 180
    d = np.abs(a - b) % 360
    df['r_p_sim'] = np.where(d <= 180, d, 360 - d)
    return df

def normalize_W_Features(df):
    df['W_s'] = df['W_s'] / df['W_c']
    df['W_ss'] = df['W_ss'] / df['W_c']
    df['W_ms'] = df['W_ms'] / df['W_c']
    df['W_as'] = df['W_as'] / df['W_c']
    df['N_s'] = df['N_s'] / df['N_c']
    df['N_ss'] = df['N_ss'] / df['N_c']
    df['N_ms'] = df['N_ms'] / df['N_c']
    df['N_as'] = df['N_as'] / df['N_c']
    return df


X_train_all = compute_r_p_sim(X_train_all)
X_val_all = compute_r_p_sim(X_val_all)
X_test_all = compute_r_p_sim(X_test_all)

X_train_all = normalize_W_Features(X_train_all)
X_val_all = normalize_W_Features(X_val_all)
X_test_all = normalize_W_Features(X_test_all)


scaler = StandardScaler()
num_cols = ["speed", "matching_score", "r_p_sim"]  # numeric columns
scaler.fit(X_train_all[num_cols])

X_train_all[num_cols] = scaler.transform(X_train_all[num_cols])
X_val_all[num_cols]   = scaler.transform(X_val_all[num_cols])
X_test_all[num_cols]  = scaler.transform(X_test_all[num_cols])

# %% POINTS FEATURES ONLY
print("\n           POINT FEATURES ONLY\n")

evaluator.add_model("MapClean-P")
model_p = MapCleanClassifier()

columns=[
        "lon",
        "lat",
        "speed",
        "angle",
        "matching_score",
        "matched_road_src",
        "matched_road_dst",
        "matched_road_angle",
        "r_p_sim",
    ]

# columns = [x for x in columns if x not in drop_columns]

X_train_p = X_train_all[columns].copy()
X_val_p = X_val_all[columns].copy()
X_test_p = X_test_all[columns].copy()

model_p.fit(X_train_p, y_train, X_val_p, y_val)

y_predict_p_train = model_p.predict(X_train_p)
evaluator.get_metrics(y_train, y_predict_p_train, "train")

y_predict_p_val = model_p.predict(X_val_p)
evaluator.get_metrics(y_val, y_predict_p_val, "val")

y_predict_p_test = model_p.predict(X_test_p)
evaluator.get_metrics(y_test, y_predict_p_test, "test")

# MatchedRN_acc, wholeRN_acc, road_acc, road_detection_acc, wrong_roads_prec, points, labels = evaluator.get_wholeRoadNetwork_accuracy(road_network, X_test_all, matchedRoad_test, y_test, y_predict_p_test, threshold=THRESHOLD)
# print('Road Acc: ', MatchedRN_acc, wholeRN_acc, road_acc, ', RDetectionAcc: ', road_detection_acc, ', WrongPrec: ', wrong_roads_prec)
# MatchedRN_acc, wholeRN_acc, road_acc, road_detection_acc, wrong_roads_prec, points, labels = evaluator.get_road_accuracy(road_network, X_test_all, matchedRoad_test, y_test, y_predict_p_test, threshold=THRESHOLD)

# false_make_points, true_make_points = evaluator.get_make_points(X_test_all, y_test, y_predict_p_test)
# new_wrong_roads, n_new_wrong_roads = map_maker.infer_roads(false_make_points, skip_snapping=True)
# road_network_acc, whole_road_network_acc = evaluator.get_road_accuracy(road_network, n_new_wrong_roads, matchedRoad_test, y_test, y_predict_p_test, threshold=THRESHOLD)
# print(f'Road Acc: {road_network_acc}, Whole Road Network Acc: {whole_road_network_acc}')

acc = evaluator.compute_overall_acc(y_pred_train, y_pred_val, y_predict_p_test, y_train_true, y_val_true, y_test)
print('Overall Accuracy: ', acc)

# %%
# TODO: you may also think of dropping as well the angle
X_train_pos = X_train_all[['lat', 'lon', 'angle']].copy()
X_val_pos = X_val_all[['lat', 'lon', 'angle']].copy()
X_test_pos = X_test_all[['lat', 'lon', 'angle']].copy()

X_train_all = X_train_all.drop(columns=drop_columns)
X_val_all = X_val_all.drop(columns=drop_columns)
X_test_all = X_test_all.drop(columns=drop_columns)


# %%
# MODEL AT n=0 (unweighted nearby features)
print("")
print("           POINT + UNWEIGHTED FEATURES ONLY")
print("")
evaluator.add_model("MapClean-U")

model_n0 = MapCleanClassifier()

columns = [
    "lon",
    "lat",
    "speed",
    "angle",
    "matching_score",
    "matched_road_src",
    "matched_road_dst",
    "matched_road_angle",
    "r_p_sim",
    "N_c",
    "N_s",
    "N_ms",
    "N_as",
    "N_ss",
    "N_r",
    "N_Dir",
]
columns = [x for x in columns if x not in drop_columns]

X_train_n0 = X_train_all[columns].copy()
X_val_n0 = X_val_all[columns].copy()
X_test_n0 = X_test_all[columns].copy()

model_n0.fit(X_train_n0, y_train, X_val_n0, y_val)

y_predict_n0_train = model_n0.predict(X_train_n0)
evaluator.get_metrics(y_train, y_predict_n0_train, "train")

y_predict_n0_val = model_n0.predict(X_val_n0)
evaluator.get_metrics(y_val, y_predict_n0_val, "val")

y_predict_n0_test = model_n0.predict(X_test_n0)
evaluator.get_metrics(y_test, y_predict_n0_test, "test")

# MatchedRN_acc, wholeRN_acc, road_acc, road_detection_acc, wrong_roads_prec, points, labels = evaluator.get_wholeRoadNetwork_accuracy(road_network, pd.concat([X_test_all, X_test_pos], axis=1), matchedRoad_test, y_test, y_predict_n0_test, threshold=THRESHOLD)
# print('Road Acc: ', MatchedRN_acc, wholeRN_acc, road_acc, ', RDetectionAcc: ', road_detection_acc, ', WrongPrec: ', wrong_roads_prec)

# false_make_points, true_make_points = evaluator.get_make_points(pd.concat([X_test_all, X_test_pos], axis=1), y_test, y_predict_n0_test)
# new_wrong_roads, n_new_wrong_roads = map_maker.infer_roads(false_make_points, skip_snapping=True)
# road_network_acc, whole_road_network_acc = evaluator.get_road_accuracy(road_network, n_new_wrong_roads, matchedRoad_test, y_test, y_predict_n0_test, threshold=THRESHOLD)
# print(f'Road Acc: {road_network_acc}, Whole Road Network Acc: {whole_road_network_acc}')

acc = evaluator.compute_overall_acc(y_pred_train, y_pred_val, y_predict_n0_test, y_train_true, y_val_true, y_test)
print('Overall Accuracy: ', acc)
# %%
# MODEL AT n=1
print("")
print("           POINT + WEIGHTED (n=1) FEATURES ONLY")
print("")
evaluator.add_model("MapClean")
model_n1 = MapCleanClassifier()

columns = [
    "lon",
    "lat",
    "speed",
    "angle",
    "matching_score",
    "matched_road_src",
    "matched_road_dst",
    "matched_road_angle",
    "r_p_sim",
    "W_c",
    "W_s",
    "W_ms",
    "W_as",
    "W_ss",
    "W_r",
    "W_Dir"
]

columns = [x for x in columns if x not in drop_columns]

X_train_n1 = X_train_all[columns].copy()
X_val_n1 = X_val_all[columns].copy()
X_test_n1 = X_test_all[columns].copy()

model_n1.fit(X_train_n1, y_train, X_val_n1, y_val)

y_predict_n1_train = model_n1.predict(X_train_n1)
evaluator.get_metrics(y_train, y_predict_n1_train, "train")

y_predict_n1_val = model_n1.predict(X_val_n1)
evaluator.get_metrics(y_val, y_predict_n1_val, "val")

y_predict_n1_test = model_n1.predict(X_test_n1)
evaluator.get_metrics(y_test, y_predict_n1_test, "test")

# MatchedRN_acc, wholeRN_acc, road_acc, road_detection_acc, wrong_roads_prec, points, labels = evaluator.get_wholeRoadNetwork_accuracy(road_network, pd.concat([X_test_all, X_test_pos], axis=1), matchedRoad_test, y_test, y_predict_n1_test, threshold=THRESHOLD)
# print('Road Acc: ', wholeRN_acc, road_acc, ', RDetectionAcc: ', road_detection_acc, ', WrongPrec: ', wrong_roads_prec)
# false_make_points, true_make_points = evaluator.get_make_points(pd.concat([X_test_all, X_test_pos], axis=1), y_test, y_predict_n1_test)
# new_wrong_roads, n_new_wrong_roads = map_maker.infer_roads(false_make_points, skip_snapping=True)
# road_network_acc, whole_road_network_acc = evaluator.get_road_accuracy(road_network, n_new_wrong_roads, matchedRoad_test, y_test, y_predict_n1_test, threshold=THRESHOLD)

# print(f'Road Acc: {road_network_acc}, Whole Road Network Acc: {whole_road_network_acc}')
acc = evaluator.compute_overall_acc(y_pred_train, y_pred_val, y_predict_n1_test, y_train_true, y_val_true,y_test)
print('Overall Accuracy: ', acc)


# %% MODEL AT n=1 and n=0 and points
print("")
print("           POINT + UNWEIGHTED + WEIGHTED (n=1) FEATURES")
print("")
evaluator.add_model("all")

model_all = MapCleanClassifier()
model_all.fit(X_train_all, y_train, X_val_all, y_val, X_test_all, y_test)

y_predict_all_train = model_all.predict(X_train_all)
evaluator.get_metrics(y_train, y_predict_all_train, "train")

y_predict_all_val = model_all.predict(X_val_all)
evaluator.get_metrics(y_val, y_predict_all_val, "val")

y_predict_all_test = model_all.predict(X_test_all)
evaluator.get_metrics(y_test, y_predict_all_test, "test")

# MatchedRN_acc, wholeRN_acc, road_acc, road_detection_acc, wrong_roads_prec, points, labels = evaluator.get_wholeRoadNetwork_accuracy(road_network, pd.concat([X_test_all, X_test_pos], axis=1), matchedRoad_test, y_test, y_predict_all_test, threshold=THRESHOLD)
# print('Road Acc: ', wholeRN_acc, road_acc, ', RDetectionAcc: ', road_detection_acc, ', WrongPrec: ', wrong_roads_prec)

# false_make_points, true_make_points = evaluator.get_make_points(pd.concat([X_test_all, X_test_pos], axis=1), y_test, y_predict_all_test)
# new_wrong_roads, n_new_wrong_roads = map_maker.infer_roads(false_make_points, skip_snapping=True)
# road_network_acc, whole_road_network_acc = evaluator.get_road_accuracy(road_network, n_new_wrong_roads, matchedRoad_test, y_test, y_predict_all_test, threshold=THRESHOLD)

acc = evaluator.compute_overall_acc(y_pred_train, y_pred_val, y_predict_all_test, y_train_true, y_val_true, y_test)
print('Overall Accuracy: ', acc)


# %%
# Saving all the results and the predictions
folder_name = f"city_{config['CITY']}_pro/modelResults_d{config['D']}n{config['N']}_SERg{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}/"
print(folder_name)
print('Saving everything')
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

np.save(folder_name+"y_val_true.npy", y_val_true)
np.save(folder_name+"y_train_true.npy", y_train_true)
np.save(folder_name+"y_pred_train.npy", y_pred_train) 
np.save(folder_name+"y_pred_val.npy", y_pred_val)

X_train.road_geometry = X_train.road_geometry.apply(lambda x: x.wkt)
X_test.road_geometry = X_test.road_geometry.apply(lambda x: x.wkt)
X_val.road_geometry = X_val.road_geometry.apply(lambda x: x.wkt)

X_train.to_parquet(folder_name+"X_train.parquet")
X_train_p.to_parquet(folder_name+"X_train_p.parquet")
X_train_n0.to_parquet(folder_name+"X_train_n0.parquet")
X_train_n1.to_parquet(folder_name+"X_train_n1.parquet")
X_train_all.to_parquet(folder_name+"X_train_all.parquet")

np.save(folder_name+"y_train.npy", y_train)
np.save(folder_name+"y_predict_p_train.npy", y_predict_p_train)
np.save(folder_name+"y_predict_n0_train.npy", y_predict_n0_train)
np.save(folder_name+"y_predict_n1_train.npy", y_predict_n1_train) 
np.save(folder_name+"y_predict_all_train.npy", y_predict_all_train)
np.save(folder_name + "y_predict_rule_train.npy", y_predict_rule_train)

X_test.to_parquet(folder_name+"X_test.parquet")
X_test_p.to_parquet(folder_name+"X_test_p.parquet")
X_test_n0.to_parquet(folder_name+"X_test_n0.parquet")
X_test_n1.to_parquet(folder_name+"X_test_n1.parquet")
X_test_all.to_parquet(folder_name+"X_test_all.parquet")
np.save(folder_name+"y_test.npy", y_test)
np.save(folder_name + "y_predict_p_test.npy", y_predict_p_test)
np.save(folder_name + "y_predict_n0_test.npy", y_predict_n0_test)
np.save(folder_name + "y_predict_n1_test.npy", y_predict_n1_test)
np.save(folder_name + "y_predict_all_test.npy", y_predict_all_test)
np.save(folder_name + "y_predict_rule_test.npy", y_predict_rule_test)

X_val.to_parquet(folder_name+"X_val.parquet")
X_val_p.to_parquet(folder_name+"X_val_p.parquet")
X_val_n0.to_parquet(folder_name+"X_val_n0.parquet")
X_val_n1.to_parquet(folder_name+"X_val_n1.parquet")
X_val_all.to_parquet(folder_name+"X_val_all.parquet")
np.save(folder_name+"y_val.npy", y_val)
np.save(folder_name+"y_predict_p_val.npy", y_predict_p_val)
np.save(folder_name+"y_predict_n0_val.npy", y_predict_n0_val)
np.save(folder_name+"y_predict_n1_val.npy", y_predict_n1_val)
np.save(folder_name+"y_predict_all_val.npy", y_predict_all_val)
np.save(folder_name + "y_predict_rule_val.npy", y_predict_rule_val)

evaluator.save_results(folder_name + "metrics.json")

joblib.dump(model_p, folder_name + "model_p.pkl")
joblib.dump(model_n0, folder_name + "model_n0.pkl")
joblib.dump(model_n1, folder_name + "model_n1.pkl")
joblib.dump(model_all, folder_name + "model_all.pkl")

np.save(folder_name+"matchedRoad_train.npy", matchedRoad_train)
np.save(folder_name+"matchedRoad_val.npy", matchedRoad_val)
np.save(folder_name + "matchedRoad_test.npy", matchedRoad_test)

# %%
evaluator.plot_acc_metrics()
# %%
