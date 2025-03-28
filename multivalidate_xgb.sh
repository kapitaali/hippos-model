#!/bin/sh
# easy rider
#./venv/bin/python3 validate_multi_xgb.py \
#    --learning-rates 0.02 0.05 0.1 \
#    --scale-pos-weights 7 10 13 \
#    --max-depths 5 6 7 \
#    --output-dir ./results/xgb_validation_run1 \
#    --results-lookup-path ./horse-race-predictor/racedata/xgb/race_results_lookup.pkl \
#    --data-path ./horse-race-predictor/racedata/scraped_race_data_2018-2024.json \
#    --primary-metric auc \
#    --early-stopping-rounds 50 \
#    --n-estimators 1000 \
#    --save-best-model
    # Add --debug 10 for more verbose logging if needed
    
# more combos
venv/bin/python3 validate_multi_xgb.py \
    --lr 0.01 0.015 0.02 0.025 0.03 0.04 \
    --spw 4 6 8 10 12 14 \
    --depth 4 5 6 7 8 \
    --subsample 0.7 0.8 0.9 \
    --colsample-bytree 0.7 0.8 0.9 \
    --n-estimators 2500 \
    --early-stopping-rounds 75 \
    --output-dir ./results/xgb_validation_run3_expanded \
    --results-lookup-path ./horse-race-predictor/racedata/xgb/race_results_lookup.pkl \
    --data-path ./horse-race-predictor/racedata/scraped_race_data_2018-2024.json \
    --primary-metric auc \
    --save-best-model \
    -d 20 # INFO level logging
