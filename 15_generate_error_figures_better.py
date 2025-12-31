# %%
from generate_error_functions import (
    get_results_sigma, 
    get_results_gamma, 
    plot_error_experiment, 
    plot_n_d_experiment, 
    plot_superpoint_size_experiment,
    plot_n_d_experiment_1n
)

from load_config import load_config

import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 12,               # small but readable for papers
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,        # thinner axes lines
})

THRESHOLD = 0.5
GAMMA_LIST = [0.04, 0.08, 0.12, 0.16]
SIGMA_LIST = [1, 5, 10, 15, 20] #, 25]
metric = 'overall_acc' # 'accuracy_make', 'accuracy_match', 'balanced_accuracy', 'recall', 'f1_score'
metric = 'matchedRoadNetwork_acc' # 'accuracy_make', 'accuracy_match', 'balanced_accuracy', 'recall', 'f1_score'
y_label = "Road Network Accuracy"

metric_list = ['accuracy_match', 'accuracy_make']
y_label_list = ['Matching Acc', 'Making Acc']

metric_list = ['overall_acc', 'matchedRoadNetwork_acc']
metric_list = ['overall_acc', 'road_network_acc']
y_label_list = ['Accuracy (%)', 'Accuracy (%)']
metric_list = ['acc', 'accuracy_match', 'accuracy_make', 'overall_acc', 'road_network_acc']
y_label_list = ['acc', 'Accuracy (%)', 'Accuracy (%)', 'Accuracy (%)', 'Accuracy (%)']


forbidden_list = ['all', 'Map Matching', 'Map Making']

# %%
config = load_config('configs/chicago_s.json', same_gamma=False)
plot_error_experiment(config, get_results_gamma, GAMMA_LIST, metric_list, y_label_list, forbidden_list, 'Removed roads (%)', 'roadRemoval')
config['GAMMA_O'] = 0.05
plot_error_experiment(config, get_results_sigma, SIGMA_LIST, metric_list, y_label_list, forbidden_list, 'Noise Level (m)', 'noise')

# %%
config = load_config('configs/singapore_s.json', same_gamma=False)
plot_error_experiment(config, get_results_gamma, GAMMA_LIST, metric_list, y_label_list, forbidden_list, 'Removed roads (%)', 'roadRemoval')
config['GAMMA_O'] = 0.05
plot_error_experiment(config, get_results_sigma, SIGMA_LIST, metric_list, y_label_list, forbidden_list, 'Noise Level (m)', 'noise')

# %%
from generate_error_functions import get_results_superpoint_size, MARKER_SIZE

# config = load_config('configs/singapore1m.json', same_gamma=False)
# config = load_config('configs/chicago.json', same_gamma=False)
# config = load_config('configs/singapore_area.json', same_gamma=False)
config = load_config('configs/jakarta_better.json', same_gamma=False)

# config['BETA'] = 0.7
# config['DELTA'] = 5 
# config['GAMMA_O'] = 0.05 # 0.05
# config['SIGMA_O'] = 0.05
# config['D'] = 100
GAMMA_LIST = [0.05, 0.1, 0.15, 0.2]
SIGMA_LIST = [5, 10, 15, 20] #, 25]
metric_list = ['f1_score_neg', 'f1_score_pos']
metric_list = ['f1_score_neg', 'f1_score_pos', 'precision_neg', 'precision_pos', 'recall_neg', 'recall_pos']
y_label_list = ['F1 score', 'F1 score', 'Precision', 'Precision', 'Recall', 'Recall']
import matplotlib.pyplot as plt
def plot_error_experiment(config, get_results_var, VAR_LIST, metric_list, y_label_list, forbidden_list, x_label, name):
    temp_results_dict, models_list = get_results_var(config, VAR_LIST[0])

    markers_dict = {
        'MapClean': 'o',
        'MapClean-U': 's',
        'MapClean-P': '^',
        'Time': 'X',
        'Rule Cleaner': '*', 
        'Rule-Filter': '*', 
        'Rule Filter (thr=1)': '*',
        'Rule Filter (thr=10)': '*',
        'Rule Filter (thr=100)': '*'
    }
    
    line_styles_dict = {
        'MapClean': 'solid',
        'MapClean-U': '--',
        'MapClean-P': '--',
        'Time': '-',
        'Rule Cleaner': '-', 
        'Rule-Filter': '-', 
        'Rule Filter (thr=1)': '-',
        'Rule Filter (thr=10)': '--',
        'Rule Filter (thr=100)': ':'
    }

    for metric, y_label in zip(metric_list, y_label_list):
        fig, ax = plt.subplots(figsize=(2.6, 2.1))
        counter = 0
        for idx, S in enumerate(models_list):
            print(S)
            if S == 'Rule Cleaner': 
                S = 'Rule-Filter'
            if S in forbidden_list: continue
            y_labelslist = []
            for VAR in VAR_LIST:
                temp_results_dict, models_list = get_results_var(config, VAR)
                y_labelslist.append(temp_results_dict['test'][metric][idx])


            if name == 'roadRemoval': var_LIST = [int(100*v) for v in VAR_LIST]
            else: var_LIST = VAR_LIST
            print(y_labelslist)
            ax.plot(
                var_LIST,
                y_labelslist,
                label=S if S != 'Rule Cleaner' else 'Rule Filter (thr=10)',
                marker=markers_dict[S],
                linestyle=line_styles_dict[S],
                markersize=MARKER_SIZE,
                linewidth=2 if S == 'MapClean' else 1,
                markerfacecolor='white' if S != 'MapClean' else 'black',
                markeredgewidth=0.9,
                color='black'
            )
            counter += 1
        plt.tight_layout()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks(var_LIST)
        ax.set_xticklabels(var_LIST) #, rotation=45)
        # ax.grid(False)
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        
        fig.subplots_adjust(top=0.75)
        
        # Get current handles and labels
        handles, labels = ax.get_legend_handles_labels()

        # Desired order by label name
        desired_order = [
            "MapClean",
            "Rule-Filter",
            "MapClean-P",
            "MapClean-U",
            "Rule Filter (thr=1)",
            'Rule Filter (thr=10)',
            'Rule Filter (thr=100)'
        ]

        # Reorder
        label_to_handle = dict(zip(labels, handles))
        ordered_handles = [label_to_handle[l] for l in desired_order if l in label_to_handle]
        ordered_labels  = [l for l in desired_order if l in label_to_handle]

        # Legend in 2 rows, centered above plot
        ax.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.35, 1.5),
            ncol=2,
            frameon=False,
            fontsize=12,
            labelspacing=0.1,
            borderpad=0.0,
            columnspacing=0.3,
            # markerscale=0.8,  # smaller
            handlelength=1.2,  # default is ~2.0
            handletextpad=0.1,   # ← space between symbol and text
        )
        
        plt.savefig(
            f"NewFigures/{config['CITY']}_{name}_{metric}.eps", 
            bbox_inches='tight', 
            pad_inches=0,
            format='eps')
        plt.show()

plot_error_experiment(config, get_results_sigma, SIGMA_LIST, metric_list, y_label_list, forbidden_list, 'Noise Level (m)', 'noise')
y_label_list = ['F1 score', 'F1 score', 'Precision', 'Precision', 'Recall', 'Recall']
plot_error_experiment(config, get_results_gamma, GAMMA_LIST, metric_list, y_label_list, forbidden_list, 'Removed roads (%)', 'roadRemoval')

# %%
import json
config = load_config('configs/jakarta_m.json', same_gamma=False)
config['GAMMA_O'] = 0.05

N_LIST = [0, 2]#, 3, 4]
D_LIST = [0, 25, 50, 75, 100, 125, 150, 175, 200]
D_LIST = [0, 50, 150, 200] # 100
plot_n_d_experiment(config, N_LIST, D_LIST, metric_list, y_label_list)

# %%
import matplotlib.pyplot as plt
from generate_error_functions import get_results_superpoint_size, MARKER_SIZE

def plot_superpoint_size_experiment(config, SUPER_POINT_LIST, metric_list, y_label_list):
    markers_dict = {
        'MapClean': 'o',
        'MapClean-U': 's',
        'MapClean-P': '^',
        'Time': 'X',
        'Rule Cleaner': '*', 
        'Rule-Filter': '*', 
        'Rule Filter (thr=1)': '*',
        'Rule Filter (thr=10)': '*',
        'Rule Filter (thr=100)': '*'
    }
    
    line_styles_dict = {
        'MapClean': 'solid',
        'MapClean-U': '--',
        'MapClean-P': '--',
        'Time': '-',
        'Rule Cleaner': '-', 
        'Rule-Filter': '-', 
        'Rule Filter (thr=1)': '-',
        'Rule Filter (thr=10)': '--',
        'Rule Filter (thr=100)': ':'
    }

    counter = 0

    forbidden_list = ['all', 'Map Matching', 'Map Making', 'Rule Cleaner']

    x_label = r'$SuperPoint$ Size ($m$)'
    temp_results_dict, models_list = get_results_superpoint_size(config, SUPER_POINT_LIST[0])
    print(temp_results_dict)
    for metric, y_label in zip(metric_list, y_label_list):
        ss = 0
        fig, ax = plt.subplots(figsize=(2.6, 2.1))
        ax2 = ax.twinx()
        counter = 0
        for idx, S in enumerate(models_list):
            if S in forbidden_list: continue
            y_labels_list = []
            execution_time_list = []
            for SUPER_POINT in SUPER_POINT_LIST:
                ss += 1
                temp_results_dict, models_list = get_results_superpoint_size(config, SUPER_POINT)
                print(temp_results_dict)
                y_labels_list.append(temp_results_dict['test'][metric][idx])

                filename = f"city_{config['CITY']}_pro/featureExTime_d{config['D']}n{config['N']}_SERg{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{SUPER_POINT}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.txt"
                with open(filename, "r") as f:
                    execution_time = float(f.read().strip())
                execution_time_list.append(execution_time/60/60)

            ax.plot(
                SUPER_POINT_LIST,
                y_labels_list,
                label=S,
                marker=markers_dict[S],
                linestyle=line_styles_dict[S],
                markersize=MARKER_SIZE,
                markeredgewidth=0.9,
                linewidth=2 if S == 'MapClean' else 1,
                markerfacecolor='black' if S == 'MapClean' else 'white',
                color='black'
            )
            counter += 1
        ax2.plot(
            SUPER_POINT_LIST,
            execution_time_list,
            label=f'Time',
            marker=markers_dict['Time'],
            linestyle=line_styles_dict['Time'],
            markersize=MARKER_SIZE,
            markeredgewidth=0.9,
            linewidth=2 if S == 'MapClean' else 1,
            markerfacecolor= 'black' if S == 'MapClean' else 'white', 
            color='black'
        )
        print(execution_time_list)
        counter += 1
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        fig.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="upper center",
            bbox_to_anchor=(0.52, 1.0),   # move legend higher or lower
            ncol=2,                        # number of columns → 2 rows automatically
            frameon=False,
            fontsize=12,
            labelspacing=0.2,
            borderpad=0.3,
            columnspacing=0.4,
        )
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xscale('log')
        ax.set_xticks(SUPER_POINT_LIST)
        ax.set_xticklabels(SUPER_POINT_LIST)#, rotation=45)
        fig.subplots_adjust(top=0.7)
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        ax2.set_ylabel("Time (hours)")
        # --- make second axis black ---
        # ax.set_ylim(83, 100)
        ax2.spines['right'].set_visible(True)
        plt.savefig(f"NewFigures/{config['CITY']}_superpoint_size_{metric}.eps", bbox_inches='tight', format='eps')
        plt.show()

# config = load_config('configs/jakarta_m.json', same_gamma=False)
config = load_config('configs/jakarta_better.json', same_gamma=False)
# config['DELTA_O'] = 5
# config['DELTA'] = config['DELTA_O']
# config['BETA_O'] = 0.7
# config['BETA'] = config['BETA_O']
# config['GAMMA_O'] = 0.05
# SUPER_POINT_LIST = [1, 5, 10, 20, 30, 40, 50, 100]
SUPER_POINT_LIST = [1, 10, 100]
SUPER_POINT_LIST = [1, 10, 100]
metric_list = ['whole_network_acc', 'road_errors', 'new_wrong_roads', 'road_detec_acc', 'acc', 'accuracy_match', 'accuracy_make', 'overall_acc', 'road_network_acc']
y_label_list = metric_list#['acc', 'Accuracy (%)', 'Accuracy (%)', 'Accuracy (%)', 'Accuracy (%)']
metric_list = ['f1_score_pos', 'f1_score_neg']
y_label_list = ['Accuracy (%)', 'Accuracy (%)']
metric_list = ['f1_score_neg', 'f1_score_pos', 'precision_neg', 'precision_pos', 'recall_neg', 'recall_pos']
y_label_list = ['F1 score', 'F1 score', 'Precision', 'Precision', 'Recall', 'Recall']
plot_superpoint_size_experiment(config, SUPER_POINT_LIST, metric_list, y_label_list)

# %%
import matplotlib.pyplot as plt
from generate_error_functions import get_results_superpoint_size, MARKER_SIZE
import json
# config = load_config('configs/jakarta_m.json', same_gamma=False)
config = load_config('configs/singapore1m.json', same_gamma=False)
config = load_config('configs/chicago.json', same_gamma=False)
config = load_config('configs/jakarta_better.json', same_gamma=False)

# config['GAMMA_O'] = 0.05
# config['GAMMA'] = 0.2
config['GAMMA_O'] = 0.15
# config['GAMMA'] = 0.2
# config['SIGMA'] = 15

metric_list = ['overall_acc', 'whole_network_acc']
y_label_list = ['Accuracy (%)', 'Accuracy (%)']
def get_results_n_d(config, N=None, D=None):
    if N is None:
        N = config['N']
    if D is None:
        D = config['D']

    if N <= 0:
        # folder_name = f"city_{config['CITY']}_pro/modelResults_d{D}_n{2}_C{config['CELL_WIDTH']}_SERg{int(100 * config['GAMMA'])}s{config['SIGMA']}p{int(100 * config['P_NOISE'])}gr{int(config['REMOVAL_ROADS_GROUPING'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION'])}_PEb{int(100 * config['BETA'])}g{config['DELTA']}_OERg{int(100 * config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100 * config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}_sup{config['SUPER_POINT_SIZE']}/"
        folder_name = f"city_{config['CITY']}_pro/modelResults_d{D}n{2}_SERg{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}/"
    else:
        # folder_name = f"city_{config['CITY']}_pro/modelResults_d{D}_n{N}_C{config['CELL_WIDTH']}_SERg{int(100 * config['GAMMA'])}s{config['SIGMA']}p{int(100 * config['P_NOISE'])}gr{int(config['REMOVAL_ROADS_GROUPING'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION'])}_PEb{int(100 * config['BETA'])}g{config['DELTA']}_OERg{int(100 * config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100 * config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}_sup{config['SUPER_POINT_SIZE']}/"
        folder_name = f"city_{config['CITY']}_pro/modelResults_d{D}n{N}_SERg{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}/"
    with open(folder_name+"metrics.json", "r") as f:
        results = json.load(f)
    models_list = results['model_name']
    del results['model_name']
    return results, models_list

def plot_n_d_experiment_1n(config, best_N, D_LIST, metric_list, y_label_list):
    '''
    VAR : N
    VAR2: D
    '''

    x_label = r"$d$ (m)"


    temp_results_dict, models_list = get_results_n_d(config, best_N, D_LIST[1])
    
    markers_dict = {
        'MapClean': 'o',
        'MapClean-U': 's',
        'MapClean-P': '^',
        'Time': 'X',
        'Rule Cleaner': '*', 
        'Rule-Filter': '*', 
        'Rule Filter (thr=1)': '*',
        'Rule Filter (thr=10)': '*',
        'Rule Filter (thr=100)': '*'
    }
    
    line_styles_dict = {
        'MapClean': 'solid',
        'MapClean-U': '--',
        'MapClean-P': '--',
        'Time': '-',
        'Rule Cleaner': '-', 
        'Rule-Filter': '-', 
        'Rule Filter (thr=1)': '-',
        'Rule Filter (thr=10)': '--',
        'Rule Filter (thr=100)': ':'
    }

    for metric, y_label in zip(metric_list, y_label_list):
        ss = 0
        fig, ax = plt.subplots(figsize=(2.6, 2.1))
        ax2 = ax.twinx()
        counter = 0
        
        temp_results_dict, models_list = get_results_n_d(config, best_N, 150)
        print(temp_results_dict)
        idx = models_list.index("MapClean-P")
        y_lab_d0 = len(D_LIST) * [temp_results_dict['test'][metric][idx]]
        # y_lab_d0 = len(D_LIST) * [0.88]
        
        ax.plot(
            D_LIST,
            # [100*a for a in y_lab_d0],
            y_lab_d0,
            label=f'MapClean-P',
            marker=markers_dict['MapClean-P'],
            linestyle=line_styles_dict['MapClean-P'],
            markersize=MARKER_SIZE,
            # color=colors_ax[counter % len(colors_ax)],
            markeredgewidth=0.9,
            linewidth=1,
            markerfacecolor='white' if ss % 2 == 0 else 'black',
            color='black'
        )
        counter += 1
        
        for N in [0, best_N]:
            ss += 1
            y_labels_list = []
            is_d0_taken = False

            if not is_d0_taken:
                temp_results_dict, models_list = get_results_n_d(config, N)
                idx = models_list.index("MapClean-P")
                # for k2 in temp_results_dict['test']:
                y_labels_list = [temp_results_dict['test'][metric][idx]]
                is_d0_taken = True

            execution_time_list = [0]
            for D in D_LIST[1:]:
                temp_results_dict, models_list = get_results_n_d(config, N, D)

                # for k2 in temp_results_dict['test']:
                if N == 0:
                    idx = models_list.index("MapClean-U")
                    y_labels_list.append(temp_results_dict['test'][metric][idx])
                elif N == -1:
                    idx = models_list.index("all")
                    y_labels_list.append(temp_results_dict['test'][metric][idx])
                else:
                    idx = models_list.index("MapClean")
                    y_labels_list.append(temp_results_dict['test'][metric][idx])
                    # if k2 not in y_labels_dict: y_labels_dict[k2] = []
                # if N == 0: N = 2
                if N <= 0: continue
                # filename = f"city_{config['CITY']}_pro/featureExTime_d{D}_n{N}_C{config['CELL_WIDTH']}_SERg{int(100 * config['GAMMA'])}s{config['SIGMA']}p{int(100 * config['P_NOISE'])}gr{int(config['REMOVAL_ROADS_GROUPING'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION'])}_PEb{int(100 * config['BETA'])}g{config['DELTA']}_OERg{int(100 * config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100 * config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}_sup{config['SUPER_POINT_SIZE']}.txt"
                filename = f"city_{config['CITY']}_pro/featureExTime_d{D}n{N}_SERg{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.txt"
                with open(filename, "r") as f:
                    execution_time = float(f.read().strip())
                execution_time_list.append(execution_time)

            if N == 0: S = 'MapClean-U'
            if N == -1: S = 'MapClean-A'
            if N > 0: S = 'MapClean'
            if N == 1: S = 'MapClean(n=1)'

            print(y_labels_list)
            ax.plot(
                D_LIST,
                # [100*a for a in y_labels_list],
                y_labels_list,
                # y_labels_list,
                label=S,
                marker=markers_dict[S],
                linestyle=line_styles_dict[S],
                markersize=MARKER_SIZE,
                # color=colors_ax[counter % len(colors_ax)],
                markeredgewidth=0.9,
                linewidth=2 if N == 2 else 1,
                markerfacecolor='black' if S == 'MapClean' else 'white',
                color='black'
            )
            counter += 1
            if N <= 0: continue
            ax2.plot(
                D_LIST,
                execution_time_list,
                label=f'Time',
                marker=markers_dict['Time'],
                linestyle=line_styles_dict['Time'],
                markersize=MARKER_SIZE,
                markeredgewidth=0.9,
                linewidth=2 if N == 2 else 1,
                markerfacecolor='white', 
                color='black'
            )
            counter += 1
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        fig.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="upper center",
            bbox_to_anchor=(0.52, 0.98),   # move legend higher or lower
            ncol=2,                        # number of columns → 2 rows automatically
            frameon=False,
            fontsize=12,
            labelspacing=0.2,
            borderpad=0.3,
            columnspacing=0.4,
        )
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks(D_LIST)
        # ax.set_xticklabels(D_LIST, rotation=60)
        fig.subplots_adjust(top=0.7)
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        ax2.set_ylabel("Time (sec)")
        # plt.tight_layout()
        ax2.spines['right'].set_visible(True)
        plt.savefig(f"NewFigures/{config['CITY']}_n_d_{metric}.eps", bbox_inches='tight', format='eps')
        plt.show()
D_LIST = [0, 25, 50, 75, 100, 125, 150, 175, 200]
D_LIST = [0, 50, 100, 150, 200] # 100

metric_list = ['f1_score_pos', 'f1_score_neg'] #, 'f1_score_pos', 'precision_neg', 'precision_pos', 'recall_neg', 'recall_pos']
y_label_list = 2*['F1 score']

plot_n_d_experiment_1n(config, 2, D_LIST, metric_list, y_label_list)

# %%
