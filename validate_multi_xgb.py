# --- START OF FILE validate_multi_xgb.py (Final - All Metrics, Resumable, Correct Saving) ---

import json
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, precision_recall_fscore_support
import numpy as np
import pickle
from tqdm import tqdm
import logging
import ijson
import multiprocessing
import argparse
import itertools
from pathlib import Path
import time
import pandas as pd

# --- Import functions from xg_train ---
# Ensure xg_train.py is in the same directory or accessible via PYTHONPATH
# and that it has been corrected (parsing inside main, logger global)
try:
    from xg_train import stream_json_data, process_batch
    print("--- Successfully imported functions from xg_train ---")
except ImportError as e:
    print(f"ERROR: Could not import from xg_train.py: {e}. Please ensure it's accessible and corrected.")
    # Define dummy functions or exit
    def stream_json_data(*args, **kwargs): raise NotImplementedError("stream_json_data missing")
    def process_batch(*args, **kwargs): raise NotImplementedError("process_batch missing")
    exit(1) # Exit if imports fail

# --- Argument Parsing (Full Corrected Definition with All Metrics) ---
parser = argparse.ArgumentParser(description="RESUMABLE - Train/validate multiple XGBoost models.")

# Data Paths
parser.add_argument('--data-path', type=str, default='./horse-race-predictor/racedata/scraped_race_data_2018-2024.json',
                    help='Path to combined race data JSON')
parser.add_argument('--results-lookup-path', type=str, default='./horse-race-predictor/racedata/xgb/race_results_lookup.pkl',
                    help='Path to REQUIRED precomputed race results lookup table')
parser.add_argument('--output-dir', type=str, default='./horse-race-predictor/racedata/xgb/xgb_multi_validation',
                    help='Directory to save results summary, best model, scaler, and track map')

# Hyperparameters to Vary
parser.add_argument('--learning-rates', '--lr', type=float, nargs='+', default=[0.03, 0.05, 0.1],
                    help='List of learning rates (eta) to try')
parser.add_argument('--scale-pos-weights', '--spw', type=float, nargs='+', default=[5.0, 10.0, 15.0],
                    help='List of scale_pos_weight values to try')
parser.add_argument('--max-depths', '--depth', type=int, nargs='+', default=[5, 6, 7],
                    help='List of max_depth values to try')
parser.add_argument('--subsample', type=float, nargs='+', default=[0.8],
                    help='List of subsample ratios to try')
parser.add_argument('--colsample-bytree', type=float, nargs='+', default=[0.8],
                    help='List of colsample_bytree ratios to try')

# Fixed Hyperparameters
parser.add_argument('--n-estimators', type=int, default=500,
                    help='Maximum number of boosting rounds (used with manual best iter check)')

# Other Settings
parser.add_argument('--val-split', type=float, default=0.2, help='Validation set split ratio')
parser.add_argument('--early-stopping-rounds', type=int, default=30, help='Patience rounds for manual best iteration check (logging only)')
parser.add_argument('--primary-metric', type=str, default='auc',
                    choices=['auc', 'logloss', 'f1', 'precision', 'recall', 'accuracy'], # <-- EXPANDED choices
                    help="Metric to select best model (higher is better for all except logloss)")
parser.add_argument('--save-best-model', action='store_true', help='Save the best performing model artifacts found *in this run*')
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
LOG_FILE = OUTPUT_DIR / f"validation_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=args.debug,
    format='%(levelname)s:%(name)s:%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[ logging.FileHandler(LOG_FILE), logging.StreamHandler() ]
)
logger = logging.getLogger(__name__)
logger.info("Starting RESUMABLE multi-model validation run.")
logger.info("Arguments: %s", vars(args))

# --- Define which metrics improve when higher ---
higher_is_better_metrics = {'auc', 'f1', 'precision', 'recall', 'accuracy'}

# --- Define paths for results files ---
RESULTS_CSV_PATH = OUTPUT_DIR / "validation_summary.csv"
BEST_PARAMS_PATH = OUTPUT_DIR / "best_params.json" # Path for potentially reloading best score

# --- GPU Check ---
try:
    import subprocess
    subprocess.check_output('nvidia-smi')
    gpu_support = True
    logger.info("NVIDIA GPU detected.")
except Exception:
    gpu_support = False
    logger.info("No NVIDIA GPU detected.")

# --- Load Essential Lookup Data ---
def load_results_lookup(lookup_load_path):
    lookup_load_path = Path(lookup_load_path)
    if not lookup_load_path.exists(): logger.error(f"Lookup file NOT FOUND: {lookup_load_path}"); return None
    try:
        logger.info(f"Loading lookup table: {lookup_load_path}...")
        with open(lookup_load_path, 'rb') as f: results_lookup = pickle.load(f)
        logger.info("Lookup loaded."); return results_lookup
    except Exception as e: logger.error(f"Failed loading lookup: {e}", exc_info=True); return None

results_lookup = load_results_lookup(args.results_lookup_path)
if results_lookup is None: logger.error("Lookup missing. Exiting."); exit(1)

# --- Load Data, Scale, Split (ONCE) ---
def load_and_prepare_data(data_path, results_lookup, val_split):
    logger.info("Loading and preparing full dataset...")
    all_features_list, all_labels_list, track_map = [], [], {}
    total_samples, total_wins = 0, 0
    try:
        batch_generator = stream_json_data(data_path, batch_size=10000) # Use a reasonable batch size
        data_loader = tqdm(batch_generator, desc="Loading Full Data")

        for batch_idx, race_batch in enumerate(data_loader):
            features_np, labels_np, current_track_map = process_batch(
                 race_batch, results_lookup, scaler=None, track_map=track_map # Don't scale yet
            )
            track_map.update(current_track_map) # Accumulate track map

            if features_np is not None and labels_np is not None and features_np.shape[0] > 0:
                all_features_list.append(features_np)
                all_labels_list.append(labels_np)
                batch_wins = int(np.sum(labels_np))
                total_wins += batch_wins
                total_samples += len(labels_np)
                data_loader.set_postfix({"total_wins": total_wins, "total_samples": total_samples})

        if not all_features_list:
            raise ValueError("No valid data collected after processing the entire dataset.")

        X = np.vstack(all_features_list); del all_features_list
        y = np.concatenate(all_labels_list); del all_labels_list
        logger.info(f"Full dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
        logger.info(f"Total wins: {total_wins} ({total_wins / X.shape[0] * 100:.2f}%)")

        if total_wins == 0:
            logger.error("CRITICAL: No wins found in dataset even with lookup. Cannot train.")
            return None, None, None, None, None, None

        # Split data BEFORE scaling
        logger.info(f"Splitting data (Validation split: {val_split:.1f})")
        stratify_option = y if total_wins > 0 and total_wins < len(y) else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=stratify_option
        )
        del X, y; import gc; gc.collect()
        logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

        # Fit scaler ONLY on training data
        logger.info("Fitting StandardScaler on training data...")
        scaler = StandardScaler(); scaler.fit(X_train)
        logger.info("Scaler fitted.")

        # Scale both training and validation data
        logger.info("Scaling training and validation data...")
        X_train_scaled = scaler.transform(X_train); del X_train
        X_val_scaled = scaler.transform(X_val); del X_val; gc.collect()
        logger.info("Data scaling complete.")

        # Return prepared data and tools
        return X_train_scaled, X_val_scaled, y_train, y_val, scaler, track_map

    except Exception as e:
        logger.error(f"Error during data loading/preparation: {e}", exc_info=True)
        return None, None, None, None, None, None

X_train_scaled, X_val_scaled, y_train, y_val, scaler, track_map = load_and_prepare_data(
    args.data_path, results_lookup, args.val_split
)
if X_train_scaled is None: logger.error("Data preparation failed. Exiting."); exit(1)

# --- Generate FULL Hyperparameter Grid ---
param_grid = {
    'learning_rate': args.learning_rates,
    'scale_pos_weight': args.scale_pos_weights,
    'max_depth': args.max_depths,
    'subsample': args.subsample,
    'colsample_bytree': args.colsample_bytree,
    'n_estimators': [args.n_estimators],
}
param_names = list(param_grid.keys())
all_param_combinations_tuples = list(itertools.product(*param_grid.values()))
total_combinations_planned = len(all_param_combinations_tuples)
logger.info(f"Generated {total_combinations_planned} total hyperparameter combinations planned.")
logger.debug(f"Parameter Grid: {param_grid}")

# --- Load Existing Results and Filter Combinations ---
completed_combos_set = set()
# --- Initialize best_score based on metric type ---
best_score = -np.inf if args.primary_metric in higher_is_better_metrics else np.inf
best_params = None # Tracks overall best params found across runs

if RESULTS_CSV_PATH.exists() and os.path.getsize(RESULTS_CSV_PATH) > 0:
    logger.info(f"Found existing results file: {RESULTS_CSV_PATH}. Loading completed runs...")
    try:
        existing_df = pd.read_csv(RESULTS_CSV_PATH)
        if not existing_df.empty:
            param_cols_in_csv = [col for col in param_names if col in existing_df.columns]
            if len(param_cols_in_csv) != len(param_names):
                logger.warning("CSV columns mismatch current parameter names. Resuming might be incomplete.")

            # Convert loaded columns to object type before creating tuples to handle mixed types gracefully
            existing_params_df = existing_df[param_cols_in_csv].astype(object)
            completed_combos_set = set(map(tuple, existing_params_df.to_numpy()))
            logger.info(f"Loaded {len(completed_combos_set)} completed combinations from previous runs.")

            # Initialize overall best score from the loaded file
            metric_col = f'val_{args.primary_metric}'
            if metric_col in existing_df.columns:
                existing_df_sorted = existing_df.sort_values(by=metric_col, ascending=(args.primary_metric == 'logloss'), na_position='last')
                if not existing_df_sorted.empty:
                    best_run_so_far = existing_df_sorted.iloc[0]
                    if pd.notna(best_run_so_far[metric_col]):
                        best_score = best_run_so_far[metric_col]
                        # Reconstruct best_params dict carefully
                        best_params = {}
                        for p_name in param_names:
                            if p_name in best_run_so_far:
                                val = best_run_so_far[p_name]
                                # Attempt type conversion based on original grid types
                                if p_name in ['max_depth', 'n_estimators']: val = int(val)
                                elif p_name in ['learning_rate', 'scale_pos_weight', 'subsample', 'colsample_bytree']: val = float(val)
                                best_params[p_name] = val
                        logger.info(f"Initialized overall best score ({args.primary_metric}) from existing file: {best_score:.5f}")
                        logger.info(f"Initialized overall best params from existing file: {best_params}")
                    else: logger.warning(f"Best score for metric '{args.primary_metric}' in CSV was NaN. Re-initializing."); best_params = None; best_score = -np.inf if args.primary_metric in higher_is_better_metrics else np.inf
                else: logger.warning("Could not sort existing CSV. Re-initializing best score."); best_params = None; best_score = -np.inf if args.primary_metric in higher_is_better_metrics else np.inf
            else: logger.warning(f"Metric '{metric_col}' not in CSV. Cannot initialize best score."); best_params = None; best_score = -np.inf if args.primary_metric in higher_is_better_metrics else np.inf

    except Exception as e:
        logger.error(f"Error loading/processing existing CSV: {e}. Starting fresh.", exc_info=True)
        completed_combos_set = set(); best_params = None
        best_score = -np.inf if args.primary_metric in higher_is_better_metrics else np.inf

# Filter the list of combinations to run
param_combinations_to_run = [
    combo for combo in all_param_combinations_tuples if combo not in completed_combos_set
]
num_to_run = len(param_combinations_to_run)
num_already_done = total_combinations_planned - num_to_run

if num_already_done > 0: logger.info(f"Skipping {num_already_done} combinations already completed.")
logger.info(f"Starting loop for {num_to_run} remaining combinations...")

# --- Training and Evaluation Loop ---
best_model_this_run_details = {} # Track best model found *in this execution* for saving
num_cores = multiprocessing.cpu_count()

# Open CSV in append mode, write header if file is new/empty
file_exists = RESULTS_CSV_PATH.exists() and os.path.getsize(RESULTS_CSV_PATH) > 0
# Define header names based on potential keys in run_result
header_names = ['run_id'] + param_names + ['best_iteration', 'actual_rounds', 'stopped_early',
                'val_logloss', 'val_auc', 'val_accuracy', 'val_precision', 'val_recall',
                'val_f1', 'training_time_s', 'status']
csv_file = open(RESULTS_CSV_PATH, 'a', newline='')
writer = pd.DataFrame(columns=header_names) # Use pandas to write header easily
if not file_exists:
    writer.to_csv(csv_file, index=False, header=True, lineterminator='\n')
    csv_file.flush()

try:
    for i, combo_tuple in enumerate(param_combinations_to_run):
        current_params = dict(zip(param_names, combo_tuple))
        # Generate consistent run ID based on overall count
        run_id = f"run_{num_already_done + i + 1:0{len(str(total_combinations_planned))}d}"
        logger.info(f"--- {run_id}/{total_combinations_planned}: Testing Params: {current_params} ---")

        xgb_model = xgb.XGBClassifier( objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
                                       random_state=42, nthread=max(1, num_cores - 1), **current_params )
        if gpu_support: xgb_model.set_params(device='cuda', tree_method='hist'); logger.debug("Using GPU")
        else: xgb_model.set_params(tree_method='hist'); logger.debug("Using CPU")
        eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
        start_time = time.time()
        run_status = "COMPLETED"

        try:
            # --- Fit without explicit early stopping ---
            xgb_model.fit( X_train_scaled, y_train, eval_set=eval_set, verbose=False )
            training_time = time.time() - start_time

            # --- Evaluate ---
            eval_results = xgb_model.evals_result()
            val_logloss_history = eval_results['validation_1']['logloss']
            if not val_logloss_history:
                logger.error(f"({run_id}) No validation results. Training failed early.")
                run_status = "FAILED_NO_RESULTS"; raise ValueError("No results")

            best_iteration_idx = np.argmin(val_logloss_history)
            actual_rounds = len(val_logloss_history)
            stopped_early = (best_iteration_idx < actual_rounds - 1) and \
                            (actual_rounds - (best_iteration_idx + 1) >= args.early_stopping_rounds)
            final_val_logloss = val_logloss_history[best_iteration_idx]
            try: # Predict using best iteration
                 booster = xgb_model.get_booster()
                 dval = xgb.DMatrix(X_val_scaled)
                 y_pred_proba = booster.predict(dval, iteration_range=(0, best_iteration_idx + 1))
            except Exception as pred_err:
                 logger.warning(f"({run_id}) Iteration predict failed ({pred_err}). Using final state.")
                 y_pred_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]

            # Calculate all metrics
            final_val_auc = roc_auc_score(y_val, y_pred_proba)
            y_pred_binary = (y_pred_proba > 0.5).astype(int)
            final_val_accuracy = accuracy_score(y_val, y_pred_binary)
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_binary, average='binary', zero_division=0)

            logger.info(f"({run_id}) Train Time: {training_time:.2f}s | Rounds: {actual_rounds} | BestIter: {best_iteration_idx + 1} {'(Sim. Early Stop)' if stopped_early else ''}")
            logger.info(f"({run_id}) Val LogLoss: {final_val_logloss:.5f} | Val AUC: {final_val_auc:.5f} | Val Acc: {final_val_accuracy:.5f} | P: {precision:.5f} | R: {recall:.5f} | F1: {f1:.5f}")

            # --- Check for best model based on chosen primary metric ---
            # Use locals() to dynamically get the value of the variable matching the metric name
            current_score = locals().get(f'final_val_{args.primary_metric}', np.nan if args.primary_metric in higher_is_better_metrics else np.inf)

            # Check if this score is better than the overall best score found so far (including previous runs)
            is_better = (args.primary_metric in higher_is_better_metrics and current_score > best_score) or \
                        (args.primary_metric == 'logloss' and current_score < best_score)

            if is_better and pd.notna(current_score):
                logger.info(f"({run_id}) *** Overall best model updated ({args.primary_metric}: {current_score:.5f}) ***")
                best_score = current_score; best_params = current_params # Update overall best trackers
                # Store details for potentially saving *this specific model* later
                best_model_this_run_details = { 'run_id': run_id, 'params': current_params, 'score': best_score,
                                                'best_iteration': best_iteration_idx + 1,
                                                'booster_object': xgb_model.get_booster() if args.save_best_model else None }

        except Exception as train_err:
             logger.error(f"({run_id}) FAILED training/evaluation: {train_err}", exc_info=True)
             run_status = "FAILED"
             training_time = time.time() - start_time # Record time even on failure
             best_iteration_idx, actual_rounds, stopped_early = -1, 0, False
             final_val_logloss, final_val_auc, final_val_accuracy = np.nan, np.nan, np.nan
             precision, recall, f1 = np.nan, np.nan, np.nan

        # --- Append result to CSV IMMEDIATELY ---
        run_result = { 'run_id': run_id, **current_params, 'best_iteration': best_iteration_idx + 1,
                       'actual_rounds': actual_rounds, 'stopped_early': stopped_early,
                       'val_logloss': final_val_logloss, 'val_auc': final_val_auc, 'val_accuracy': final_val_accuracy,
                       'val_precision': precision, 'val_recall': recall, 'val_f1': f1,
                       'training_time_s': training_time, 'status': run_status }
        pd.DataFrame([run_result]).to_csv(csv_file, index=False, header=False, float_format='%.5f', lineterminator='\n')
        csv_file.flush()

finally:
    csv_file.close() # Ensure the CSV file is closed

# --- Summarize and Save FINAL Best Results ---
logger.info("--- Multi-validation run finished processing combinations ---")
logger.info(f"Reading final summary from: {RESULTS_CSV_PATH}")
try:
    final_results_df = pd.read_csv(RESULTS_CSV_PATH)
    successful_runs_df = final_results_df[final_results_df['status'] == 'COMPLETED'].copy()

    if not successful_runs_df.empty:
        # Determine sort order based on the primary metric
        sort_ascending = False if args.primary_metric in higher_is_better_metrics else True
        successful_runs_df.sort_values(by=f'val_{args.primary_metric}', ascending=sort_ascending, inplace=True, na_position='last')

        # Get the overall best run details from the potentially combined CSV
        overall_best_run = successful_runs_df.iloc[0]
        overall_best_score = overall_best_run[f'val_{args.primary_metric}']

        if pd.notna(overall_best_score):
             # Reconstruct overall best params correctly (handle potential type issues from CSV read)
             overall_best_params = {}
             for p_name in param_names:
                 if p_name in overall_best_run:
                     val = overall_best_run[p_name]
                     try: # Attempt conversion based on original grid types
                         if isinstance(param_grid[p_name][0], int): val = int(val)
                         elif isinstance(param_grid[p_name][0], float): val = float(val)
                     except Exception: logger.warning(f"Could not convert param {p_name} value {val} to original type.")
                     overall_best_params[p_name] = val

             overall_best_iteration = int(overall_best_run['best_iteration'])
             overall_best_run_id = overall_best_run['run_id']

             logger.info(f"--- Overall Best Model Identified (Across All Runs) ---")
             logger.info(f"  Run ID: {overall_best_run_id}")
             logger.info(f"  Score ({args.primary_metric}): {overall_best_score:.5f}")
             logger.info(f"  Parameters: {overall_best_params}")
             logger.info(f"  Best Iteration: {overall_best_iteration}")

             # --- Save the best model FOUND IN *THIS* RUN (if requested) ---
             if args.save_best_model:
                 # Check if the best overall run ID matches the best run found *in this execution*
                 if best_model_this_run_details and best_model_this_run_details.get('run_id') == overall_best_run_id:
                      logger.info(f"Saving the best model artifacts (Run ID: {overall_best_run_id}) from this execution...")
                      best_booster = best_model_this_run_details.get('booster_object')
                      if best_booster:
                          try:
                              # Define paths using OUTPUT_DIR
                              best_model_path = OUTPUT_DIR / "best_model.ubj"
                              best_scaler_path = OUTPUT_DIR / "best_scaler.pkl"
                              best_track_map_path = OUTPUT_DIR / "best_track_map.json"
                              best_params_path = OUTPUT_DIR / "best_params.json"

                              # Save the booster
                              best_booster.save_model(best_model_path)
                              logger.info(f"Saved Best Model Booster to: {best_model_path}")

                              # Save the corresponding scaler and track map
                              with open(best_scaler_path, 'wb') as f: pickle.dump(scaler, f)
                              logger.info(f"Saved Scaler to: {best_scaler_path}")
                              with open(best_track_map_path, 'w') as f: json.dump(track_map, f, indent=2)
                              logger.info(f"Saved Track Map to: {best_track_map_path}")

                              # Save the parameters and score of the model *actually saved*
                              params_to_save = {
                                  'best_score_metric': args.primary_metric,
                                  'best_score': overall_best_score, # Use overall best score
                                  'best_params': overall_best_params, # Use overall best params
                                  'best_iteration': int(overall_best_iteration) # Cast to standard Python int
                              }
                              with open(best_params_path, 'w') as f: json.dump(params_to_save, f, indent=2)
                              logger.info(f"Saved Best Parameters to: {best_params_path}")

                          except Exception as save_err:
                              logger.error(f"Failed saving best model artifacts: {save_err}", exc_info=True)
                      else:
                          logger.warning("Saving best model requested, but booster object was not available for the best run in this execution.")
                 # If the overall best was from a previous run, check if artifacts exist
                 elif os.path.exists(OUTPUT_DIR / "best_model.ubj"):
                     logger.info("Best overall model determined from previous run(s). Artifacts should already exist.")
                 else:
                     logger.warning(f"Best overall model (Run ID: {overall_best_run_id}) was from a previous run, but its artifacts were not found in {OUTPUT_DIR}. Not saving artifacts from this execution.")

             else:
                  logger.info("Saving best model was not requested (--save-best-model not set).")

        else:
             logger.warning("Best score found in summary was NaN. Cannot determine or save best parameters/model.")
    else:
        logger.warning("No successful runs found in the results summary.")

except FileNotFoundError:
    logger.error(f"Final summary CSV not found at {RESULTS_CSV_PATH}. Cannot determine best model.")
except Exception as read_final_err:
     logger.error(f"Could not read/process final summary CSV {RESULTS_CSV_PATH}: {read_final_err}", exc_info=True)

logger.info("Script finished.")

# --- END OF FILE ---
