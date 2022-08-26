#!/usr/bin/env bash

CONFIG=${CONFIG:-example_node}
GRID=${GRID:-example}
REPEAT=${REPEAT:-3}
MAX_JOBS=${MAX_JOBS:-8}
SLEEP=${SLEEP:-1}
MAIN=${MAIN:-main}

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python pytorch_geometric/graphgym/configs_gen.py --config pytorch_geometric/graphgym/configs/pyg/${CONFIG}.yaml \
  --grid pytorch_geometric/graphgym/grids/${GRID}.txt \
  --out_dir pytorch_geometric/graphgym/configs
#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
bash pytorch_geometric/graphgym/parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash pytorch_geometric/graphgym/parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash pytorch_geometric/graphgym/parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN

# aggregate results for the batch
python pytorch_geometric/graphgym/agg_batch.py --dir pytorch_geometric/graphgym/results/${CONFIG}_grid_${GRID}
