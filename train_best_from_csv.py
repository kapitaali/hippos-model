import json
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from tqdm import tqdm
import logging
import ijson
import multiprocessing
import argparse
from pathlib import Path
import time
import pandas as pd

# --- Import functions from the 19-feature training script ---
# CHANGE THIS FILENAME if you named your 19-feature training script differently
try:
    from xg_train_19feat import load_results_lookup, stream_json_data, process_batch
    print("--- Successfully imported functions from xg_train_19feat ---")
except ImportError as e:
    print(f"ERROR: Could not import from xg_train_19feat.py: {e}. Ensure it's accessible and corrected.")
    exit(1)

# --- Argument Parsing (Updated Defaults for 19 Features) ---
parser = argparse.ArgumentParser(description="Train the final best 19-feature XGBoost model based on validation summary.")

# Input Paths
parser.add_argument('--summary-csv-path', type=str, required=True,
                    help='Path to the 19-feature validation_summary.csv file')
parser.add_argument('--data-path', type=str, default='./horse-race-predictor/racedata/scraped_race_data_2018-2024.json',
                    help='Path to the original combined race data JSON')
parser.add_argument('--results-lookup-path', type=str, default='./horse-race-predictor/racedata/xgb/race_results_lookup.pkl',
                    help='Path to REQUIRED precomputed race results lookup table') # Use the same lookup

# Output Paths (Updated Defaults)
parser.add_argument('--output-dir', type=str, default='./final_model_artifacts/19feat_best',
                    help='Directory to save the final 19-feature model, scaler, and track map')

# Metric for Selecting Best
parser.add_argument('--primary-metric', type=str, default='auc',
                    choices=['auc', 'logloss', 'f1', 'precision', 'recall', 'accuracy'],
                    help="Metric used to determine the best run in the summary CSV")

# Other Settings
parser.add_argument('--debug', '-d', type=int, default=20, choices=[10, 20, 30, 40, 50],
                    help='Debug level')

print(f"--- EXECUTING SCRIPT: {os.path.abspath(__file__)} ---")
print("--- Parsing arguments ---")
args = parser.parse_args()
print("--- Arguments parsed ---")
# --- END OF Argument Parsing Section ---

# --- Setup Output Directory and Logging ---
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / f"final_train_19feat_{time.strftime('%Y%m%d_%H%M%S')}.log" # Add identifier
logging.basicConfig(
    level=args.debug,
    format='%(levelname)s:%(name)s:%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[ logging.FileHandler(LOG_FILE), logging.StreamHandler() ]
)
logger = logging.getLogger(__name__)
logger.info("Starting final 19-feature model training based on validation summary.")
logger.info("Arguments: %s", vars(args))

# --- Define which metrics improve when higher ---
higher_is_better_metrics = {'auc', 'f1', 'precision', 'recall', 'accuracy'}
INPUT_SIZE = 19 # Define expected feature size

# --- GPU Check ---
try:
    import subprocess
    subprocess.check_output('nvidia-smi')
    gpu_support = True
    logger.info("NVIDIA GPU detected.")
except Exception:
    gpu_support = False
    logger.info("No NVIDIA GPU detected.")

# --- Load Lookup ---
results_lookup = load_results_lookup(args.results_lookup_path)
if results_lookup is None: logger.error("Lookup missing. Exiting."); exit(1)

# --- Read Summary CSV and Find Best Parameters ---
best_params = None
best_iteration = None
summary_path = Path(args.summary_csv_path)
metric_col = f'val_{args.primary_metric}'

if not summary_path.exists():
    logger.error(f"Validation summary CSV not found: {summary_path}. Cannot determine best parameters.")
    exit(1)

logger.info(f"Reading validation summary: {summary_path}")
try:
    summary_df = pd.read_csv(summary_path)
    # Infer parameter names from columns (needed for reconstruction)
    known_cols = {'run_id', 'best_iteration', 'actual_rounds', 'stopped_early', 'val_logloss',
                  'val_auc', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1',
                  'training_time_s', 'status'}
    param_names_from_csv = [col for col in summary_df.columns if col not in known_cols]

    successful_runs_df = summary_df[summary_df['status'] == 'COMPLETED'].copy()
    if successful_runs_df.empty: raise ValueError("No successful runs found in the summary CSV.")
    if metric_col not in successful_runs_df.columns: raise ValueError(f"Metric column '{metric_col}' not found in summary CSV.")

    sort_ascending = False if args.primary_metric in higher_is_better_metrics else True
    successful_runs_df.sort_values(by=metric_col, ascending=sort_ascending, inplace=True, na_position='last')

    best_run = successful_runs_df.iloc[0]
    best_score = best_run[metric_col]

    if pd.notna(best_score):
        best_params = {}
        logger.info("--- Best Parameters Identified from CSV ---")
        for p_name in param_names_from_csv:
            if p_name not in best_run: continue # Skip if param somehow missing
            val = best_run[p_name]
            # Attempt type conversion based on standard XGBoost params
            if p_name in ['max_depth', 'n_estimators']: val = int(val)
            elif p_name in ['learning_rate', 'scale_pos_weight', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']: val = float(val) # Added common regularization params just in case
            best_params[p_name] = val
            logger.info(f"  {p_name}: {val}")

        best_iteration = int(best_run['best_iteration'])
        logger.info(f"  Best Iteration (used for n_estimators): {best_iteration}")
        logger.info(f"  Best Score ({args.primary_metric}): {best_score:.5f}")

        # CRITICAL: Override n_estimators with best_iteration
        if 'n_estimators' in best_params:
             logger.info(f"Overriding n_estimators ({best_params['n_estimators']}) with best_iteration: {best_iteration}")
        else:
             logger.info(f"Setting n_estimators for final training to best_iteration: {best_iteration}")
        best_params['n_estimators'] = best_iteration

    else: raise ValueError(f"The best score found for metric '{args.primary_metric}' was NaN.")

except FileNotFoundError: logger.error(f"Validation summary CSV not found: {summary_path}"); exit(1)
except ValueError as e: logger.error(f"Error processing summary CSV: {e}"); exit(1)
except Exception as e: logger.error(f"Unexpected error reading summary CSV: {e}", exc_info=True); exit(1)

if best_params is None or best_iteration is None: logger.error("Could not determine best parameters/iteration. Exiting."); exit(1)


# --- Load ALL Data and Prepare for Final Training ---
logger.info(f"Loading ALL data for final training (expecting {INPUT_SIZE} features)...")
all_features_list, all_labels_list = [], []
track_map = {} # Reset track map, will be rebuilt from all data
total_samples, total_wins = 0, 0
try:
    # Use the IMPORTED 19-feature stream/process functions
    batch_generator = stream_json_data(args.data_path, batch_size=10000)
    data_loader = tqdm(batch_generator, desc="Loading Full Data for Final Train")
    for batch_idx, race_batch in enumerate(data_loader):
        # Process batch should yield 19 features now
        features_np, labels_np, current_track_map = process_batch(
             race_batch, results_lookup, scaler=None, track_map=track_map
        )
        track_map.update(current_track_map) # Accumulate track map from all data
        if features_np is not None and labels_np is not None and features_np.shape[0] > 0:
             if features_np.shape[1] != INPUT_SIZE:
                 raise ValueError(f"Feature shape mismatch in batch {batch_idx+1}! Expected {INPUT_SIZE}, got {features_np.shape[1]}.")
             all_features_list.append(features_np); all_labels_list.append(labels_np)
             total_samples += len(labels_np)
             data_loader.set_postfix({"total_samples": total_samples})

    if not all_features_list: raise ValueError("No valid data collected after processing.")

    X_full = np.vstack(all_features_list); del all_features_list
    y_full = np.concatenate(all_labels_list); del all_labels_list
    logger.info(f"Full dataset loaded: {X_full.shape[0]} samples, {X_full.shape[1]} features.")
    if X_full.shape[1] != INPUT_SIZE: # Final check
        raise ValueError(f"Final feature matrix shape mismatch! Expected {INPUT_SIZE} cols, got {X_full.shape[1]}.")
    import gc; gc.collect()

    # Fit scaler on the FULL **19-feature** dataset
    logger.info("Fitting StandardScaler on FULL 19-feature data...")
    final_scaler = StandardScaler()
    final_scaler.fit(X_full)
    logger.info("Scaler fitted on full data.")

    # Scale the FULL dataset
    logger.info("Scaling FULL 19-feature data...")
    X_full_scaled = final_scaler.transform(X_full); del X_full
    gc.collect()
    logger.info("Full data scaling complete.")

except Exception as e:
    logger.error(f"Error during final data loading/preparation: {e}", exc_info=True)
    exit(1)


# --- Train Final Model ---
logger.info("--- Training Final Model with Best Parameters ---")
logger.info(f"Using Parameters: {best_params}")
logger.info(f"Training for exactly {best_params['n_estimators']} rounds.")

final_model = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    random_state=42,
    nthread=max(1, multiprocessing.cpu_count() - 1),
    # Unpack best hyperparameters (includes n_estimators=best_iteration)
    **best_params
)

# Set device/tree method
if gpu_support: final_model.set_params(device='cuda', tree_method='hist'); logger.info("Using GPU.")
else: final_model.set_params(tree_method='hist'); logger.info("Using CPU.")

start_time = time.time()
try:
    # Train on the entire dataset
    # Set verbose high enough to see progress but not overwhelming for many rounds
    verbose_freq = max(1, best_params['n_estimators'] // 20)
    logger.info(f"Starting final fit (verbose frequency: {verbose_freq})...")
    final_model.fit(X_full_scaled, y_full, verbose=verbose_freq)
    training_time = time.time() - start_time
    logger.info(f"Final model training completed in {training_time:.2f}s.")

    # --- Save Final Artifacts ---
    logger.info(f"Saving final model and artifacts to: {OUTPUT_DIR}")
    try:
        # Add 19feat suffix to filenames
        model_path = OUTPUT_DIR / "final_model_19feat.ubj"
        scaler_path = OUTPUT_DIR / "final_scaler_19feat.pkl"
        track_map_path = OUTPUT_DIR / "final_track_map_19feat.json"
        final_params_path = OUTPUT_DIR / "final_params_19feat.json"

        # Model
        final_model.save_model(model_path)
        logger.info(f"Final model saved: {model_path}")

        # Scaler (fitted on all data)
        with open(scaler_path, 'wb') as f: pickle.dump(final_scaler, f)
        logger.info(f"Final scaler saved: {scaler_path}")

        # Track Map (built from all data)
        with open(track_map_path, 'w') as f: json.dump(track_map, f, indent=2)
        logger.info(f"Final track map saved: {track_map_path}")

        # Parameters used for this final model
        params_to_save = {
            'selected_metric': args.primary_metric,
            'best_params_from_validation': best_params, # Includes n_estimators=best_iteration
            'best_iteration_used': best_iteration,
            'validation_score': best_score # Score achieved during validation
        }
        with open(final_params_path, 'w') as f: json.dump(params_to_save, f, indent=2)
        logger.info(f"Final parameters info saved: {final_params_path}")

    except Exception as save_err:
        logger.error(f"Failed saving final artifacts: {save_err}", exc_info=True)

except Exception as final_train_err:
    logger.error(f"FAILED final model training: {final_train_err}", exc_info=True)

logger.info("Script finished.")

# --- END OF FILE ---
