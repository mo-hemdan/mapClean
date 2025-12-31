This is a project to build a machine learning based spatial cleaning framework for map data (i.e. GPS points and road networks). 

To the run the system, you need first to prepare the dataset. That includes matching the dataset to a road network which is the first step of the spatial data cleaning framework. You can do that with the following command which downloads the road network associated with the area, match the data as well. You need to enter the city name at the top of the file. 
```shell
python 0_preprocess.py
```

To run the system you need a config.json file which contains the important parameters needed to run the system. The following was the best parameters we experimented with along with input data characteristics for experimentation

```json
{
  "CITY": "Jakarta_better",

  "SUPER_POINT_SIZE": 10,
  "CELL_WIDTH": 20,

  "DELTA_O": 10,
  "BETA_O": 0.8,

  "GAMMA_O": 0.05,
  "GAMMA": 0.1,
  "MAX_ROAD_LENGTH_O": 300,
  "MU_O": 0,
  "SIGMA_O": 15,
  "SIGMA": 5,
  "P_NOISE_O": 0.5,
  "RANDOM_SEED": 80,

  "REMOVAL_ROADS_GROUPING_O": true,
  "REMOVAL_ROAD_MAXLENGTH_OPTION_O": true,

  "INJECT_ERROR_TO_AREAS": false,
  "INJECT_NOISE_TO_REGIONS": false,
  "REMOVAL_AREA_PERC": 0.24,
  "SQUARE_SIDE_LENGTH": 1500,
  "MIN_LAT": -6.4129002796,
  "MAX_LAT": -6.12338103986,
  "MIN_LON": 106.64209758,
  "MAX_LON": 107.0168747189,
  "NEAREST_ROAD2ROAD_RANGE": 100,

  "D": 150,
  "N": 2,

  "VERBOSE": true
}
```

Once the config is prepared. You can run the system using the run_system_on_config.sh command 

```shell
./run_system_on_config.sh config.json
```

All stages of the system will run and results will be reported for rule-based filter, MapClean-P, MapClean-U and MapClean the full version. 