import json
import matplotlib.pyplot as plt

MARKER_SIZE = 8

def adjust_config(config):
    # Add derived values
    config["DELTA"] = config["DELTA_O"]
    config["BETA"] = config["BETA_O"]
    # config["GAMMA"] = round(
    #     config["GAMMA_O"]
    #     / (config["P_NOISE_O"] * (1 - config["GAMMA_O"]) + config["GAMMA_O"]),
    #     2,
    # )
    config["MAX_ROAD_LENGTH"] = config["MAX_ROAD_LENGTH_O"]
    config["MU"] = config["MU_O"]
    # config["SIGMA"] = config["SIGMA_O"]
    config["P_NOISE"] = 1
    config["REMOVAL_ROADS_GROUPING_O"] = config["REMOVAL_ROAD_MAXLENGTH_OPTION_O"]
    config["REMOVAL_ROADS_GROUPING"] = config["REMOVAL_ROADS_GROUPING_O"]
    config["REMOVAL_ROAD_MAXLENGTH_OPTION"] = config["REMOVAL_ROAD_MAXLENGTH_OPTION_O"]
    return config


def get_results_sigma(config, SIGMA_O=None):
    if SIGMA_O is None: 
        SIGMA_O = config['SIGMA_O']
    # SIGMA = SIGMA_O
    # SIGMA = config['SIGMA']
    # folder_name = f"city_{config['CITY']}_pro/modelResults_d{config['D']}_n{config['N']}_C{config['CELL_WIDTH']}_SERg{int(100 * config['GAMMA'])}s{config['SIGMA']}p{int(100 * config['P_NOISE'])}gr{int(config['REMOVAL_ROADS_GROUPING'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION'])}_PEb{int(100 * config['BETA'])}g{config['DELTA']}_OERg{int(100 * config['GAMMA_O'])}s{SIGMA_O}p{int(100 * config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}_sup{config['SUPER_POINT_SIZE']}/"
    folder_name = f"city_{config['CITY']}_pro/modelResults_d{config['D']}n{config['N']}_SERg{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{SIGMA_O}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}/"
    with open(folder_name+"metrics.json", "r") as f:
        results = json.load(f)
    models_list = results['model_name']
    del results['model_name']
    return results, models_list

def get_results_gamma(config, GAMMA_O=None):
    if GAMMA_O is None: 
        GAMMA_O = config['GAMMA_O']
    # GAMMA = round(GAMMA_O / (config["P_NOISE_O"] * (1 - GAMMA_O) + GAMMA_O), 2)
    # folder_name = f"city_{config['CITY']}_pro/modelResults_d{config['D']}_n{config['N']}_C{config['CELL_WIDTH']}_SERg{int(100 * config['GAMMA'])}s{config['SIGMA']}p{int(100 * config['P_NOISE'])}gr{int(config['REMOVAL_ROADS_GROUPING'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION'])}_PEb{int(100 * config['BETA'])}g{config['DELTA']}_OERg{int(100 * GAMMA_O)}s{config['SIGMA_O']}p{int(100 * config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}_sup{config['SUPER_POINT_SIZE']}/"
    folder_name = f"city_{config['CITY']}_pro/modelResults_d{config['D']}n{config['N']}_SERg{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*GAMMA_O)}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}/"
    with open(folder_name+"metrics.json", "r") as f:
        results = json.load(f)
    models_list = results['model_name']
    del results['model_name']
    return results, models_list

def get_results_n_d(config, N=None, D=None):
    if N is None:
        N = config['N']
    if D is None:
        D = config['D']

    if N == 0:
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

def get_results_superpoint_size(config, SUPER_POINT_SIZE=None):
    if SUPER_POINT_SIZE is None: 
        SUPER_POINT_SIZE = config['SUPER_POINT_SIZE']
    # folder_name = f"city_{config['CITY']}_pro/modelResults_d{config['D']}_n{config['N']}_C{config['CELL_WIDTH']}_SERg{int(100 * config['GAMMA'])}s{config['SIGMA']}p{int(100 * config['P_NOISE'])}gr{int(config['REMOVAL_ROADS_GROUPING'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION'])}_PEb{int(100 * config['BETA'])}g{config['DELTA']}_OERg{int(100 * config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100 * config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}_sup{SUPER_POINT_SIZE}/"
    folder_name = f"city_{config['CITY']}_pro/modelResults_d{config['D']}n{config['N']}_SERg{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{SUPER_POINT_SIZE}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}/"
    with open(folder_name+"metrics.json", "r") as f:
        results = json.load(f)
    models_list = results['model_name']
    del results['model_name']
    return results, models_list


def plot_error_experiment(config, get_results_var, VAR_LIST, metric_list, y_label_list, forbidden_list, x_label, name):
    temp_results_dict, models_list = get_results_var(config, VAR_LIST[0])

    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*']  # circle, square, triangle_up, diamond, triangle_down, X, plus_filled, star
    line_styles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 2, 5, 2))]  # 8 distinct styles

    markers = ['o', 's', '^', '*'] # 'v', 'X', 'P', '*']  # circle, square, triangle_up, diamond, triangle_down, X, plus_filled, star
    line_styles = ['-', '--', ':', '-.'] #, (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 2, 5, 2))]  # 8 distinct styles
    markers.reverse()
    line_styles.reverse()

    for metric, y_label in zip(metric_list, y_label_list):
        fig, ax = plt.subplots(figsize=(2.6, 2.1))
        counter = 0
        for idx, S in enumerate(models_list):
            if S in forbidden_list: continue
            y_labels_dict = dict()
            for VAR in VAR_LIST:
                temp_results_dict, models_list = get_results_var(config, VAR)

                for k2 in temp_results_dict['test']:
                    if k2 not in y_labels_dict: y_labels_dict[k2] = []
                    y_labels_dict[k2].append(100*temp_results_dict['test'][k2][idx])

            # ax.plot(SIGMA_LIST, y_labels_dict[metric], label=f'{S}', marker=markers[counter], color='black', linestyle=line_styles[counter], markerfacecolor='none', markersize=5)
            ax.plot(
                VAR_LIST,
                y_labels_dict[metric],
                label=S,
                marker=markers[counter],
                linestyle=line_styles[counter],
                markersize=MARKER_SIZE,
                linewidth=2 if S == 'MapClean' else 1,
                markerfacecolor='white' if idx % 2 == 0 else 'black',
                markeredgewidth=0.9,
                color='black'
            )
            counter += 1
        plt.tight_layout()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks(VAR_LIST)
        ax.set_xticklabels(VAR_LIST) #, rotation=45)
        # ax.grid(False)
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        
        fig.subplots_adjust(top=0.75)

        # Legend in 2 rows, centered above plot
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.45, 1.35),   # move legend higher or lower
            ncol=2,                        # number of columns → 2 rows automatically
            frameon=False,
            fontsize=8,
            labelspacing=0.2,
            borderpad=0.3,
            columnspacing=0.5,
        )
        
        plt.savefig(f"NewFigures/{config['CITY']}_{name}_{metric}.eps", bbox_inches='tight', format='eps')
        plt.show()
        
def plot_n_d_experiment(config, N_LIST, D_LIST, metric_list, y_label_list):
    '''
    VAR : N
    VAR2: D
    '''

    x_label = r"$d$ (m)"
    
    temp_results_dict, models_list = get_results_n_d(config, N_LIST[0], D_LIST[1])

    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*']  # circle, square, triangle_up, diamond, triangle_down, X, plus_filled, star
    line_styles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 2, 5, 2))]  # 8 distinct styles

    markers = ['o', 's', '^', '*'] # 'v', 'X', 'P', '*']  # circle, square, triangle_up, diamond, triangle_down, X, plus_filled, star
    line_styles = ['-', '--', ':', '-.'] #, (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 2, 5, 2))]  # 8 distinct styles
    markers.reverse()
    line_styles.reverse()

    for metric, y_label in zip(metric_list, y_label_list):
        ss = 0
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax2 = ax.twinx()
        counter = 0
        for N in N_LIST:
            ss += 1
            y_labels_dict = dict()
            is_d0_taken = False

            if not is_d0_taken:
                temp_results_dict, models_list = get_results_n_d(config, N)
                idx = models_list.index("MapClean-P")
                for k2 in temp_results_dict['test']:
                    y_labels_dict[k2] = [temp_results_dict['test'][k2][idx]]
                is_d0_taken = True

            execution_time_list = [0]
            for D in D_LIST[1:]:
                temp_results_dict, models_list = get_results_n_d(config, N, D)

                for k2 in temp_results_dict['test']:
                    if N == 0:
                        idx = models_list.index("MapClean-U")
                        y_labels_dict[k2].append(temp_results_dict['test'][k2][idx])
                    else:
                        idx = models_list.index("MapClean")
                        y_labels_dict[k2].append(temp_results_dict['test'][k2][idx])
                    # if k2 not in y_labels_dict: y_labels_dict[k2] = []
                # if N == 0: N = 2
                if N == 0: continue
                # filename = f"city_{config['CITY']}_pro/featureExTime_d{D}_n{N}_C{config['CELL_WIDTH']}_SERg{int(100 * config['GAMMA'])}s{config['SIGMA']}p{int(100 * config['P_NOISE'])}gr{int(config['REMOVAL_ROADS_GROUPING'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION'])}_PEb{int(100 * config['BETA'])}g{config['DELTA']}_OERg{int(100 * config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100 * config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}_sup{config['SUPER_POINT_SIZE']}.txt"
                filename = f"city_{config['CITY']}_pro/featureExTime_d{D}n{N}_SERg{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.txt"
                with open(filename, "r") as f:
                    execution_time = float(f.read().strip())
                execution_time_list.append(execution_time)

            ax.plot(
                D_LIST,
                [100*a for a in y_labels_dict[metric]],
                label=f'MC(n={N})-Acc',
                marker=markers[counter % len(markers)],
                linestyle=line_styles[counter % len(line_styles)],
                markersize=MARKER_SIZE,
                # color=colors_ax[counter % len(colors_ax)],
                markeredgewidth=0.9,
                linewidth=2 if N == 2 else 1,
                markerfacecolor='white' if ss % 2 == 0 else 'black',
                color='black'
            )
            counter += 1
            if N == 0: continue
            ax2.plot(
                D_LIST,
                execution_time_list,
                label=f'MC(n={N})-Time',
                marker=markers[counter % len(markers)],
                linestyle=line_styles[counter % len(line_styles)],
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
            bbox_to_anchor=(0.52, 1.15),   # move legend higher or lower
            ncol=2,                        # number of columns → 2 rows automatically
            frameon=False,
            fontsize=8,
            labelspacing=0.2,
            borderpad=0.3,
            columnspacing=0.4,
        )
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks(D_LIST)
        ax.set_xticklabels(D_LIST, rotation=60)
        fig.subplots_adjust(top=0.7)
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        ax2.spines['right'].set_visible(True)
        ax2.set_ylabel("Time (sec)")
        plt.tight_layout()
        plt.savefig(f"NewFigures/{config['CITY']}_n_d_{metric}.eps", bbox_inches='tight', format='eps')
        plt.show()

def plot_n_d_experiment_1n(config, best_N, D_LIST, metric_list, y_label_list):
    '''
    VAR : N
    VAR2: D
    '''

    x_label = r"$d$ (m)"


    temp_results_dict, models_list = get_results_n_d(config, best_N, D_LIST[1])

    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*']  # circle, square, triangle_up, diamond, triangle_down, X, plus_filled, star
    line_styles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 2, 5, 2))]  # 8 distinct styles

    markers = ['o', 's', '^', '*'] # 'v', 'X', 'P', '*']  # circle, square, triangle_up, diamond, triangle_down, X, plus_filled, star
    line_styles = ['-', '--', ':', '-.'] #, (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 2, 5, 2))]  # 8 distinct styles
    markers.reverse()
    line_styles.reverse()

    for metric, y_label in zip(metric_list, y_label_list):
        ss = 0
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax2 = ax.twinx()
        counter = 0
        
        temp_results_dict, models_list = get_results_n_d(config, best_N)
        idx = models_list.index("MapClean-P")
        y_lab_d0 = len(D_LIST) * [temp_results_dict['test'][metric][idx]]
        
        ax.plot(
            D_LIST,
            [100*a for a in y_lab_d0],
            label=f'MapClean-P',
            marker=markers[counter % len(markers)],
            linestyle=line_styles[counter % len(line_styles)],
            markersize=MARKER_SIZE,
            # color=colors_ax[counter % len(colors_ax)],
            markeredgewidth=0.9,
            linewidth=2 if best_N == 2 else 1,
            markerfacecolor='white' if ss % 2 == 0 else 'black',
            color='black'
        )
        counter += 1
        
        for N in [0, best_N]:
            ss += 1
            y_labels_dict = dict()
            is_d0_taken = False

            if not is_d0_taken:
                temp_results_dict, models_list = get_results_n_d(config, N)
                idx = models_list.index("MapClean-P")
                for k2 in temp_results_dict['test']:
                    y_labels_dict[k2] = [temp_results_dict['test'][k2][idx]]
                is_d0_taken = True

            execution_time_list = [0]
            for D in D_LIST[1:]:
                temp_results_dict, models_list = get_results_n_d(config, N, D)

                for k2 in temp_results_dict['test']:
                    if N == 0:
                        idx = models_list.index("MapClean-U")
                        y_labels_dict[k2].append(temp_results_dict['test'][k2][idx])
                    else:
                        idx = models_list.index("MapClean")
                        y_labels_dict[k2].append(temp_results_dict['test'][k2][idx])
                    # if k2 not in y_labels_dict: y_labels_dict[k2] = []
                # if N == 0: N = 2
                if N == 0: continue
                # filename = f"city_{config['CITY']}_pro/featureExTime_d{D}_n{N}_C{config['CELL_WIDTH']}_SERg{int(100 * config['GAMMA'])}s{config['SIGMA']}p{int(100 * config['P_NOISE'])}gr{int(config['REMOVAL_ROADS_GROUPING'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION'])}_PEb{int(100 * config['BETA'])}g{config['DELTA']}_OERg{int(100 * config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100 * config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}_sup{config['SUPER_POINT_SIZE']}.txt"
                filename = f"city_{config['CITY']}_pro/featureExTime_d{D}n{N}_SERg{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.txt"
                with open(filename, "r") as f:
                    execution_time = float(f.read().strip())
                execution_time_list.append(execution_time)

            ax.plot(
                D_LIST,
                [100*a for a in y_labels_dict[metric]],
                label='MapClean-U' if N == 0 else 'MapClean',
                marker=markers[counter % len(markers)],
                linestyle=line_styles[counter % len(line_styles)],
                markersize=MARKER_SIZE,
                # color=colors_ax[counter % len(colors_ax)],
                markeredgewidth=0.9,
                linewidth=2 if N == 2 else 1,
                markerfacecolor='white' if ss % 2 == 0 else 'black',
                color='black'
            )
            counter += 1
            if N == 0: continue
            ax2.plot(
                D_LIST,
                execution_time_list,
                label=f'Time',
                marker=markers[counter % len(markers)],
                linestyle=line_styles[counter % len(line_styles)],
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
            bbox_to_anchor=(0.52, 1.08),   # move legend higher or lower
            ncol=2,                        # number of columns → 2 rows automatically
            frameon=False,
            fontsize=8,
            labelspacing=0.2,
            borderpad=0.3,
            columnspacing=0.4,
        )
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks(D_LIST)
        ax.set_xticklabels(D_LIST, rotation=60)
        fig.subplots_adjust(top=0.7)
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        ax2.spines['right'].set_visible(True)
        ax2.set_ylabel("Time (sec)")
        plt.tight_layout()
        plt.savefig(f"NewFigures/{config['CITY']}_n_d_{metric}.eps", bbox_inches='tight', format='eps')
        plt.show()

def plot_superpoint_size_experiment(config, SUPER_POINT_LIST, metric_list, y_label_list):

    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*']  # circle, square, triangle_up, diamond, triangle_down, X, plus_filled, star
    line_styles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 2, 5, 2))]  # 8 distinct styles

    counter = 0

    forbidden_list = ['all', 'Map Matching', 'Map Making', 'Rule Cleaner']

    # metric_list = ['overall_acc', 'matchedRoadNetwork_acc']
    # y_label_list = ['Accuracy (%)', 'Accuracy (%)']
    x_label = r'$SuperPoint$ Size ($m$)'
    temp_results_dict, models_list = get_results_superpoint_size(config, SUPER_POINT_LIST[0])

    for metric, y_label in zip(metric_list, y_label_list):
        ss = 0
        fig, ax = plt.subplots(figsize=(2.6, 2.1))
        ax2 = ax.twinx()
        counter = 0
        for idx, S in enumerate(models_list):
            if S in forbidden_list: continue
            y_labels_dict = dict()
            execution_time_list = []
            for SUPER_POINT in SUPER_POINT_LIST:
                ss += 1
                temp_results_dict, models_list = get_results_superpoint_size(config, SUPER_POINT)

                for k2 in temp_results_dict['test']:
                    if k2 not in y_labels_dict: y_labels_dict[k2] = []
                    y_labels_dict[k2].append(100*temp_results_dict['test'][k2][idx])
                
                # filename = f"city_{config['CITY']}_pro/featureExTime_d{config['D']}_n{config['N']}_C{config['CELL_WIDTH']}_SERg{int(100 * config['GAMMA'])}s{config['SIGMA']}p{int(100 * config['P_NOISE'])}gr{int(config['REMOVAL_ROADS_GROUPING'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION'])}_PEb{int(100 * config['BETA'])}g{config['DELTA']}_OERg{int(100 * config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100 * config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}_sup{SUPER_POINT}.txt"
                filename = f"city_{config['CITY']}_pro/featureExTime_d{config['D']}n{config['N']}_SERg{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{SUPER_POINT}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.txt"
                with open(filename, "r") as f:
                    execution_time = float(f.read().strip())
                execution_time_list.append(execution_time)

            ax.plot(
                SUPER_POINT_LIST,
                y_labels_dict[metric],
                label=S,
                marker=markers[counter % len(markers)],
                linestyle=line_styles[counter % len(line_styles)],
                markersize=MARKER_SIZE,
                # color=colors_ax[counter % len(colors_ax)],
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
            marker=markers[counter % len(markers)],
            linestyle=line_styles[counter % len(line_styles)],
            markersize=MARKER_SIZE,
            markeredgewidth=0.9,
            linewidth=2 if S == 'MapClean' else 1,
            markerfacecolor= 'black' if S == 'MapClean' else 'white', 
            color='black'
        )
        counter += 1
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        fig.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="upper center",
            bbox_to_anchor=(0.52, 0.9),   # move legend higher or lower
            ncol=2,                        # number of columns → 2 rows automatically
            frameon=False,
            fontsize=8,
            labelspacing=0.2,
            borderpad=0.3,
            columnspacing=0.4,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks(SUPER_POINT_LIST)
        ax.set_xticklabels(SUPER_POINT_LIST)#, rotation=45)
        fig.subplots_adjust(top=0.7)
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        ax2.set_ylabel("Time (sec)")
        ax.set_xscale('log')
        # --- make second axis black ---
        ax2.spines['right'].set_visible(True)
        plt.savefig(f"NewFigures/{config['CITY']}_superpoint_size_{metric}.eps", bbox_inches='tight', format='eps')
        plt.show()