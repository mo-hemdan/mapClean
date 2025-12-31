import argparse
import json

def load_config(filepath=None, same_sigma=False, same_gamma=True):

    if filepath is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", help="Path to config file", default="config.json")

        parser.add_argument("--var", type=str)
        parser.add_argument("--value", type=str)

        parser.add_argument("--var2", type=str)
        parser.add_argument("--value2", type=str)

        args = parser.parse_args()

        # Load config
        with open(args.config) as f:
            config = json.load(f)

        # Update config programmatically
        def apply_argument(var, value):
            if var is None:
                return

            try:
                val = eval(value)
            except Exception:
                val = value  # leave as string

            config[var] = val
            print(f"Run File with args: {var}: {value}, type={type(val)}")

        apply_argument(args.var, args.value)
        apply_argument(args.var2, args.value2)
    
    else:
        with open(filepath) as f:
            config = json.load(f)

    config = adjust_config(config, same_sigma, same_gamma)

    return config

def adjust_config(config, same_sigma, same_gamma):
    
    # Add derived values
    config["DELTA"] = config["DELTA_O"]
    config["BETA"] = config["BETA_O"]
    if same_gamma:
        config["GAMMA"] = round(
            config["GAMMA_O"]
            / (config["P_NOISE_O"] * (1 - config["GAMMA_O"]) + config["GAMMA_O"]),
            2,
        )
    config["MAX_ROAD_LENGTH"] = config["MAX_ROAD_LENGTH_O"]
    config["MU"] = config["MU_O"]
    if same_sigma: config["SIGMA"] = config["SIGMA_O"]
    config["P_NOISE"] = 1
    config["REMOVAL_ROADS_GROUPING_O"] = config["REMOVAL_ROAD_MAXLENGTH_OPTION_O"]
    config["REMOVAL_ROADS_GROUPING"] = config["REMOVAL_ROADS_GROUPING_O"]
    config["REMOVAL_ROAD_MAXLENGTH_OPTION"] = config["REMOVAL_ROAD_MAXLENGTH_OPTION_O"]

    return config

