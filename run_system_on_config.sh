#!/bin/bash

# Exit if any command fails
set -e

# Get arguments
CONFIG_FILENAME=$1

# Optionally echo them for logging
echo "Running for $CONFIG_FILENAME"

# conda activate datalab
python 1_generate_error_dataset.py --config $CONFIG_FILENAME
python 2_super_points_index.py --config $CONFIG_FILENAME
python 3_extract_perfect_points.py --config $CONFIG_FILENAME
python 4_inject_system_error.py --config $CONFIG_FILENAME
g++ 6_feature_extraction_s.cpp -o 6_feature_extraction_s
./6_feature_extraction_s  $CONFIG_FILENAME
python 7_eval_road_s.py --config $CONFIG_FILENAME