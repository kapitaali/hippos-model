import json
import os
import xgboost as xgb # Uncommented
from sklearn.preprocessing import StandardScaler # Uncommented
from sklearn.model_selection import train_test_split # Uncommented
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, precision_recall_fscore_support # Uncommented
import numpy as np
import pickle
from tqdm import tqdm
import logging
import ijson # Uncommented
import multiprocessing
import argparse
import itertools
from pathlib import Path
import time
import pandas as pd # Uncommented

# --- Import functions from xg_train ---
# Ensure xg_train.py is in the same directory or accessible via PYTHONPATH
# and that it has been corrected (parsing inside main, logger global)
try:
    from xg_train import stream_json_data, process_batch
    print("--- Successfully imported functions from xg_train ---")
except ImportError:
    print("ERROR: Could not import from xg_train.py. Please ensure it's accessible and corrected.")
    # Define dummy functions or exit
    def stream_json_data(*args, **kwargs): raise NotImplementedError("stream_json_data missing")
    def process_batch(*args, **kwargs): raise NotImplementedError("process_batch missing")
    exit(1) # Exit if imports fail

# --- Argument Parsing (Full Corrected Definition) ---
# Using 'parser' variable name again, but ensure no conflict due to xg_train fix
parser = argparse.ArgumentParser(description="Train and validate multiple XGBoost models with varying hyperparameters.")

# Data Paths
parser.add_argument('--data-path', type=str, default='./horse-race-predictor/racedata/scraped_race_data_2018-2024.json',
                    help='Path to combined race data JSON')
parser.add_argument('--results-lookup-path', type=str, default='./horse-race-predictor/racedata/8feat/xgb/race_results_lookup.pkl',
                    help='Path to REQUIRED precomputed race results lookup table')
parser.add_argument('--output-dir', type=str, default='./horse-race-predictor/racedata/8feat/xgb_multi_validation',
                    help='Directory to save results summary, best model, scaler, and track map')

# Hyperparameters to Vary
parser.add_argument('--learning-rates', '--lr', type=float, nargs='+', default=[0.03, 0.05, 0.1],
                    help='List of learning rates (eta) to try')
parser.add_argument('--scale-pos-weights', '--spw', type=float, nargs='+', default=[5.0, 10.0, 15.0],
                    help='List of scale_pos_weight values to try')
parser.add_argument('--max-depths', '--depth', type=int, nargs='+', default=[5, 6, 7],
                    help='List of max_depth values to try')

# Fixed Hyperparameters
parser.add_argument('--n-estimators', type=int, default=500,
                    help='Maximum number of boosting rounds (used with manual best iter check)')
parser.add_argument('--subsample', type=float, nargs='+', default=[0.8], help='List of subsample ratios to try')
parser.add_argument('--colsample-bytree', type=float, nargs='+', default=[0.8], help='List of colsample_bytree ratios to try')
# --- END ADDED ---

# Fixed Hyperparameters (Now only n_estimators is fixed by default)
parser.add_argument('--n-estimators', type=int, default=500, help='Maximum number of boosting rounds')


# Other Settings
parser.add_argument('--val-split', type=float, default=0.2, help='Validation set split ratio')
# NOTE: early_stopping_rounds is used by the script logic, not passed to fit anymore
parser.add_argument('--early-stopping-rounds', type=int, default=30, help='Patience rounds for manual best iteration check (if needed)')
parser.add_argument('--primary-metric', type=str, default='auc', choices=['auc', 'logloss'],
                    help="Metric to select the best model ('auc' higher is better, 'logloss' lower is better)")
parser.add_argument('--save-best-model', action='store_true', help='Save the best performing model, scaler, and track map')
parser.add_argument('--debug', type=int, default=20, choices=[10, 20, 30, 40, 50],
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
logger.info("Starting multi-model validation run.")
logger.info("Arguments: %s", vars(args))

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
    # ... (Keep function content exactly as in previous successful script) ...
    logger.info("Loading and preparing full dataset...")
    all_features_list, all_labels_list, track_map = [], [], {}
    total_samples, total_wins = 0, 0
    try:
        batch_generator = stream_json_data(data_path, batch_size=10000)
        data_loader = tqdm(batch_generator, desc="Loading Full Data")
        for batch_idx, race_batch in enumerate(data_loader):
            features_np, labels_np, current_track_map = process_batch(
                 race_batch, results_lookup, scaler=None, track_map=track_map
            )
            track_map.update(current_track_map)
            if features_np is not None and labels_np is not None and features_np.shape[0] > 0:
                all_features_list.append(features_np); all_labels_list.append(labels_np)
                batch_wins = int(np.sum(labels_np)); total_wins += batch_wins
                total_samples += len(labels_np)
                data_loader.set_postfix({"total_wins": total_wins, "total_samples": total_samples})
        if not all_features_list: raise ValueError("No valid data collected.")
        X = np.vstack(all_features_list); del all_features_list
        y = np.concatenate(all_labels_list); del all_labels_list
        logger.info(f"Full dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
        logger.info(f"Total wins: {total_wins} ({total_wins / X.shape[0] * 100:.2f}%)")
        if total_wins == 0: logger.error("CRITICAL: No wins. Cannot train."); return None,None,None,None,None,None
        logger.info(f"Splitting data (Validation split: {val_split:.1f})")
        stratify_option = y if total_wins > 0 and total_wins < len(y) else None
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42, stratify=stratify_option)
        del X, y; import gc; gc.collect()
        logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        logger.info("Fitting StandardScaler on training data...")
        scaler = StandardScaler(); scaler.fit(X_train)
        logger.info("Scaling training and validation data...")
        X_train_scaled = scaler.transform(X_train); del X_train
        X_val_scaled = scaler.transform(X_val); del X_val; gc.collect()
        logger.info("Data scaling complete.")
        return X_train_scaled, X_val_scaled, y_train, y_val, scaler, track_map
    except Exception as e:
        logger.error(f"Data loading/preparation error: {e}", exc_info=True); return None,None,None,None,None,None


X_train_scaled, X_val_scaled, y_train, y_val, scaler, track_map = load_and_prepare_data(
    args.data_path, results_lookup, args.val_split
)
if X_train_scaled is None: logger.error("Data preparation failed. Exiting."); exit(1)

# --- Generate Hyperparameter Grid ---
param_grid = {
    'learning_rate': args.learning_rates,
    'scale_pos_weight': args.scale_pos_weights,
    'max_depth': args.max_depths,
    'subsample': args.subsample,             # Added key
    'colsample_bytree': args.colsample_bytree, # Added key
    'n_estimators': [args.n_estimators],     # Still fixed based on arg
}
param_names = list(param_grid.keys())
param_combinations = list(itertools.product(*param_grid.values()))
total_combinations = len(param_combinations)
logger.info(f"Generated {total_combinations} hyperparameter combinations to test.")
logger.debug(f"Parameter Grid: {param_grid}")

# --- Training and Evaluation Loop ---
results_list = []
best_score = -np.inf if args.primary_metric == 'auc' else np.inf
best_params = None
best_model_details = {}
num_cores = multiprocessing.cpu_count()

logger.info("Starting hyperparameter tuning loop...")
for i, combo in enumerate(param_combinations):
    current_params = dict(zip(param_names, combo))
    run_id = f"run_{i+1:0{len(str(total_combinations))}d}"
    logger.info(f"--- {run_id}/{total_combinations}: Testing Params: {current_params} ---")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
        random_state=42, nthread=max(1, num_cores - 1), **current_params
    )
    if gpu_support: xgb_model.set_params(device='cuda', tree_method='hist'); logger.debug("Using GPU")
    else: xgb_model.set_params(tree_method='hist'); logger.debug("Using CPU")
    eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
    start_time = time.time()
    try:
        # --- Fit without explicit early stopping in call ---
        xgb_model.fit(
            X_train_scaled, y_train, eval_set=eval_set, verbose=False
        )
        training_time = time.time() - start_time

        # --- Evaluate using manual best iteration check ---
        eval_results = xgb_model.evals_result()
        val_logloss_history = eval_results['validation_1']['logloss']

        if not val_logloss_history:
             logger.error(f"({run_id}) No validation results. Training failed early.")
             results_list.append({'run_id': run_id, **current_params, 'val_logloss': np.nan, 'val_auc': np.nan, 'status': 'FAILED_NO_RESULTS'})
             continue # Skip to next combo

        # Check if validation loss improved enough for early stopping criteria (manual check)
        best_iteration_idx = np.argmin(val_logloss_history)
        actual_rounds = len(val_logloss_history)
        stopped_early = (best_iteration_idx < actual_rounds - 1) and \
                        (actual_rounds - (best_iteration_idx + 1) >= args.early_stopping_rounds)

        final_val_logloss = val_logloss_history[best_iteration_idx]
        final_train_logloss = eval_results['validation_0']['logloss'][best_iteration_idx]

        try:
             booster = xgb_model.get_booster()
             dval = xgb.DMatrix(X_val_scaled)
             y_pred_proba = booster.predict(dval, iteration_range=(0, best_iteration_idx + 1))
        except Exception as pred_err:
             logger.warning(f"({run_id}) Iteration predict failed ({pred_err}). Using final state.")
             y_pred_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]

        final_val_auc = roc_auc_score(y_val, y_pred_proba)
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        final_val_accuracy = accuracy_score(y_val, y_pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_binary, average='binary', zero_division=0)

        logger.info(f"({run_id}) Train Time: {training_time:.2f}s | Rounds: {actual_rounds} | BestIter: {best_iteration_idx + 1} {'(Early Stopped)' if stopped_early else ''}")
        logger.info(f"({run_id}) Val LogLoss: {final_val_logloss:.4f} | Val AUC: {final_val_auc:.4f} | Val Acc: {final_val_accuracy:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

        run_result = { 'run_id': run_id, **current_params, 'best_iteration': best_iteration_idx + 1,
                       'actual_rounds': actual_rounds, 'stopped_early': stopped_early,
                       'val_logloss': final_val_logloss, 'val_auc': final_val_auc, 'val_accuracy': final_val_accuracy,
                       'val_precision': precision, 'val_recall': recall, 'val_f1': f1, 'training_time_s': training_time }
        results_list.append(run_result)

        # --- Check for best model ---
        current_score = final_val_auc if args.primary_metric == 'auc' else final_val_logloss
        is_better = (args.primary_metric == 'auc' and current_score > best_score) or \
                    (args.primary_metric == 'logloss' and current_score < best_score)
        if is_better:
            logger.info(f"({run_id}) *** New best model found ({args.primary_metric}: {current_score:.4f}) ***")
            best_score = current_score; best_params = current_params
            # Get a fresh booster object representing the best iteration state
            best_booster = xgb_model.get_booster()
            best_booster.best_iteration = best_iteration_idx # Set attribute for clarity if needed
            best_model_details = { 'run_id': run_id, 'params': current_params, 'score': best_score,
                                   'best_iteration': best_iteration_idx + 1,
                                   # Store the booster object, not the classifier if saving later
                                   'booster_object': best_booster if args.save_best_model else None }

    except Exception as train_err:
        logger.error(f"({run_id}) FAILED with params {current_params}: {train_err}", exc_info=True)
        results_list.append({'run_id': run_id, **current_params, 'val_logloss': np.nan, 'val_auc': np.nan, 'status': 'FAILED'})

# --- Summarize and Save Results ---
logger.info("--- Multi-validation run finished ---")
results_df = pd.DataFrame(results_list)
sort_ascending = True if args.primary_metric == 'logloss' else False
results_df.sort_values(by=f'val_{args.primary_metric}', ascending=sort_ascending, inplace=True)
results_csv_path = OUTPUT_DIR / "validation_summary.csv"
results_df.to_csv(results_csv_path, index=False, float_format='%.5f')
logger.info(f"Validation summary saved to: {results_csv_path}")
if best_params:
    logger.info(f"Best model based on {args.primary_metric}:")
    logger.info(f"  Run ID: {best_model_details.get('run_id', 'N/A')}")
    logger.info(f"  Score ({args.primary_metric}): {best_score:.4f}")
    logger.info(f"  Parameters: {best_params}")
    logger.info(f"  Best Iteration: {best_model_details.get('best_iteration', 'N/A')}")

    if args.save_best_model and best_model_details.get('booster_object'):
        logger.info("Saving the best model booster, scaler, and track map...")
        best_booster = best_model_details['booster_object']
        try:
            # Save Booster directly
            best_model_path = OUTPUT_DIR / "best_model.ubj"
            best_booster.save_model(best_model_path)
            logger.info(f"Best model booster saved to: {best_model_path}")

            best_scaler_path = OUTPUT_DIR / "best_scaler.pkl";
            with open(best_scaler_path, 'wb') as f: pickle.dump(scaler, f)
            logger.info(f"Scaler saved: {best_scaler_path}")

            best_track_map_path = OUTPUT_DIR / "best_track_map.json";
            with open(best_track_map_path, 'w') as f: json.dump(track_map, f, indent=2)
            logger.info(f"Track map saved: {best_track_map_path}")

            best_params_path = OUTPUT_DIR / "best_params.json";
            # --- MODIFIED Dictionary Creation ---
            best_params_data = {
                'best_score_metric': args.primary_metric,
                'best_score': best_score, # Floats are usually fine
                'best_params': best_params, # Dictionary of params
                'best_iteration': int(best_model_details.get('best_iteration', -1)) # Cast to standard Python int
            }
            # --- END MODIFIED Dictionary ---
            with open(best_params_path, 'w') as f:
                json.dump(best_params_data, f, indent=2) # Save the modified dict
            logger.info(f"Best parameters saved: {best_params_path}")

        except Exception as save_err: logger.error(f"Failed saving best artifacts: {save_err}", exc_info=True)
    elif args.save_best_model: logger.warning("Saving best requested, but object unavailable.")
else: logger.warning("No successful model runs completed.")
logger.info("Script finished.")

# --- END OF FILE validate_multi_xgb.py (Final) ---
