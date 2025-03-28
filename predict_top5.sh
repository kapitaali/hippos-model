#!/bin/sh
venv/bin/python3 predict_8feat_xgb.py $1 \
    --model-path ./results/xgb_validation_run1/best_model.ubj \
    --scaler-path ./results/xgb_validation_run1/best_scaler.pkl \
    --track-map-path ./results/xgb_validation_run1/best_track_map.json \
    --output-file ./predictions/preds_$1_best.json \
    --top-n-table 5
