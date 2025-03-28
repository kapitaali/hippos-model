# --- START OF FILE xg_train.py (Corrected - Complete) ---

import json
import os
from glob import glob
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from tqdm import tqdm
import logging # Keep logging import
import ijson
import multiprocessing
import argparse # Keep argparse import
import requests
from datetime import datetime
from pathlib import Path
import time

# --- Define logger instance globally ---
# Configured inside main() after parsing args
logger = logging.getLogger(__name__)

# --- Keep Constants and API Setup at Top Level ---
BASE_URL = "https://heppa.hippos.fi/heppa2_backend"
HEADERS = {'Content-Type': 'application/json'}
CACHE_DIR = Path("./cache_train")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# RESULTS_LOOKUP_PATH defined inside main() based on args

# --- Keep Function Definitions at Top Level ---

def fetch_api_data(endpoint, base_url=BASE_URL, headers=HEADERS, cache_dir=CACHE_DIR, retries=2, delay=5):
    """Fetch data from API with caching and retries."""
    # Sanitize endpoint for filename
    safe_endpoint = "".join(c if c.isalnum() else "_" for c in endpoint)
    cache_key = f"api_{safe_endpoint}"
    # Limit length if needed, potentially using a hash
    if len(cache_key) > 150:
         import hashlib
         cache_key = f"api_{hashlib.md5(endpoint.encode()).hexdigest()}"

    cache_file = Path(cache_dir) / f"{cache_key}.json"

    if cache_file.exists():
        try:
            logger.debug(f"Loading cached data for {endpoint} from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
             logger.warning(f"Corrupt cache file found: {cache_file}. Deleting and re-fetching.")
             cache_file.unlink() # Delete corrupt file
        except Exception as e:
            logger.error(f"Error reading cache file {cache_file}: {e}. Re-fetching.")

    url = f"{base_url}{endpoint}"
    for attempt in range(retries + 1):
        try:
            logger.debug(f"Fetching (attempt {attempt+1}/{retries+1}): {url}")
            response = requests.get(url, headers=headers, timeout=20) # Increased timeout
            # Check for common HTML error pages before raising for status
            if '<html' in response.text[:100].lower():
                 logger.error(f"Received HTML instead of JSON from {url}. Status: {response.status_code}. Body: {response.text[:200]}...")
                 if attempt < retries:
                     logger.info(f"Waiting {delay}s before retry...")
                     time.sleep(delay)
                     continue
                 else:
                     return None # Failed after retries

            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            logger.debug(f"Successfully fetched data for {endpoint}. Status: {response.status_code}")

            # Save to cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(data, f) # Save compact JSON
                logger.debug(f"Cached data to {cache_file}")
            except Exception as e:
                 logger.error(f"Failed to write cache file {cache_file}: {e}")

            return data # Success

        except requests.exceptions.Timeout:
             logger.warning(f"Timeout fetching {url} on attempt {attempt+1}")
             if attempt < retries:
                 logger.info(f"Waiting {delay}s before retry...")
                 time.sleep(delay)
             else:
                 logger.error(f"Timeout fetching {url} after {retries+1} attempts.")
                 return None # Failed after retries
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url} on attempt {attempt+1}: {e}")
            if attempt < retries:
                 logger.info(f"Waiting {delay}s before retry...")
                 time.sleep(delay)
            else:
                logger.error(f"Failed fetching {url} after {retries+1} attempts.")
                return None # Failed after retries
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode JSON response from {url}. Status: {response.status_code}. Error: {e}. Response text: {response.text[:500]}...")
             return None
    return None

def get_race_results_from_api(race_date, track_code, start_number):
    """Fetches results for a specific race start and returns a mapping."""
    endpoint = f"/race/{race_date}/{track_code}/start/{start_number}"
    horses_data = fetch_api_data(endpoint)
    if not horses_data or not isinstance(horses_data, list):
        logger.warning(f"No valid horse data via API for {race_date}/{track_code}/{start_number}"); return None
    results = {}
    for horse in horses_data:
        if isinstance(horse, dict):
            prog_num = horse.get('programNumber'); placing = horse.get('placing')
            if prog_num is not None: results[str(prog_num)] = placing
            else: logger.debug(f"Horse missing pgm# in API resp for {race_date}/{track_code}/{start_number}: {horse.get('horseName')}")
        else: logger.warning(f"Unexpected item type in API horse list for {race_date}/{track_code}/{start_number}: {type(horse)}")
    if not results: logger.warning(f"Extracted no pgm#->placing pairs from API for {race_date}/{track_code}/{start_number}"); return None
    logger.debug(f"API results fetched for {race_date}/{track_code}/{start_number}: {len(results)} horses"); return results

def build_results_lookup(data_path, lookup_save_path):
    """Scans the JSON data, fetches results from API, and saves the lookup table."""
    logger.info(f"Building race results lookup table from {data_path}...")
    unique_races = set(); total_races_scanned = 0
    logger.info("Scanning JSON for unique race identifiers (date, track, start)...")
    try:
        with open(data_path, 'rb') as f:
            parser = ijson.items(f, 'item')
            for monthly_dict in tqdm(parser, desc="Scanning Months"):
                if not isinstance(monthly_dict, dict): continue
                for date_key, date_data in monthly_dict.items():
                    if not isinstance(date_data, dict): continue
                    try: datetime.strptime(date_key, "%Y-%m-%d")
                    except ValueError: logger.warning(f"Skipping invalid date key: {date_key}"); continue
                    for track_code, track_data in date_data.items():
                         if not isinstance(track_data, dict): continue
                         for start_number, race_data in track_data.items():
                             if isinstance(race_data, dict) and str(start_number).isdigit():
                                 unique_races.add((date_key, track_code, str(start_number))); total_races_scanned += 1
        logger.info(f"Found {len(unique_races)} unique race starts after scanning {total_races_scanned} total race entries in JSON.")
    except FileNotFoundError: logger.error(f"Training data file not found: {data_path}"); raise
    except Exception as e: logger.error(f"Error scanning JSON file {data_path}: {e}", exc_info=True); raise
    results_lookup = {}; api_failures = 0
    logger.info(f"Fetching results from API for {len(unique_races)} unique races (this may take a long time)...")
    for race_date, track_code, start_number in tqdm(unique_races, desc="Fetching API Results"):
        race_results = get_race_results_from_api(race_date, track_code, start_number)
        # time.sleep(0.05) # Optional delay
        if race_results is not None:
            if race_date not in results_lookup: results_lookup[race_date] = {}
            if track_code not in results_lookup[race_date]: results_lookup[race_date][track_code] = {}
            results_lookup[race_date][track_code][start_number] = race_results
        else: api_failures += 1; logger.warning(f"Failed API lookup for {race_date}/{track_code}/{start_number}.")
    logger.info(f"Finished fetching API results. Success: {len(unique_races) - api_failures}, Failures: {api_failures}")
    try:
        logger.info(f"Saving results lookup table to {lookup_save_path}...")
        lookup_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lookup_save_path, 'wb') as f: pickle.dump(results_lookup, f)
        logger.info("Lookup table saved successfully.")
    except Exception as e: logger.error(f"Failed to save lookup table: {e}", exc_info=True)
    return results_lookup

def load_results_lookup(lookup_load_path):
    """Loads the precomputed results lookup table."""
    lookup_load_path = Path(lookup_load_path)
    if not lookup_load_path.exists(): logger.error(f"Results lookup file not found: {lookup_load_path}"); return None
    try:
        logger.info(f"Loading results lookup table from {lookup_load_path}...")
        with open(lookup_load_path, 'rb') as f: results_lookup = pickle.load(f)
        logger.info("Lookup table loaded successfully."); return results_lookup
    except Exception as e: logger.error(f"Failed to load lookup table: {e}", exc_info=True); return None


def stream_json_data(file_path, batch_size=10000):
    """Streams race data, adding context."""
    try:
        with open(file_path, 'rb') as f:
            parser = ijson.items(f, 'item'); races_batch = []; total_races_yielded = 0
            logger.info("Starting JSON stream parsing for data processing...")
            for monthly_dict in parser:
                if not isinstance(monthly_dict, dict): continue

                current_date = None # Initialize before date loop
                for date_key, date_data in monthly_dict.items():
                    if not isinstance(date_data, dict): continue
                    current_date = date_key # Assign the actual date key

                    current_track = None # Initialize before track loop
                    for track_code, track_data in date_data.items():
                         if not isinstance(track_data, dict): continue
                         current_track = track_code # Assign the actual track code

                         for start_number, race_data in track_data.items():
                             if isinstance(race_data, dict) and ('race_info' in race_data or 'starts' in race_data):
                                 # Check if context variables were assigned properly
                                 if current_date is None or current_track is None:
                                     logger.warning(f"Context missing for start {start_number}. Skipping entry. Date: {current_date}, Track: {current_track}")
                                     continue

                                 race_data['_context'] = {'date': current_date, 'track': current_track, 'start': str(start_number)}
                                 races_batch.append(race_data); total_races_yielded += 1
                                 if len(races_batch) >= batch_size:
                                     logger.debug(f"Yielding batch {len(races_batch)}. Total: {total_races_yielded}")
                                     yield races_batch; races_batch = []
            # Handle final batch
            if races_batch: logger.debug(f"Yielding final batch {len(races_batch)}. Total: {total_races_yielded}"); yield races_batch
        logger.info(f"Finished streaming JSON. Total races yielded: {total_races_yielded}")
    except FileNotFoundError: logger.error(f"Data file not found: {file_path}"); raise
    except Exception as e: logger.error(f"Error reading/parsing JSON {file_path}: {e}", exc_info=True); raise


def process_batch(races_with_context, results_lookup, scaler=None, track_map=None):
    """Processes a batch of race dictionaries to extract features and CORRECT labels."""
    features = []; labels = []; win_count = 0; missing_results_count = 0
    if track_map is None: track_map = {}
    for race_entry in races_with_context:
        context = race_entry.get('_context')
        if not context: logger.warning("Skipping entry missing _context"); continue
        race_date = context['date']; track_code = context['track']; start_number = context['start']
        race_info = race_entry.get('race_info', {}); starts = race_entry.get('starts', [])
        race_results_api = results_lookup.get(race_date, {}).get(track_code, {}).get(start_number)
        if race_results_api is None:
            logger.debug(f"Results missing in lookup: {race_date}/{track_code}/{start_number}. Assigning label 0.")
            missing_results_count += len(starts)
        # logger.debug(f"Processing Race - {race_date}/{track_code}/{start_number}, Starts: {len(starts)}") # Reduce noise
        if not starts: logger.debug("No starts list found."); continue
        if track_code not in track_map: track_map[track_code] = len(track_map)
        track_id = track_map[track_code]
        for start in starts:
            if not isinstance(start, dict): continue
            horse_data = start.get('horse_data', {})
            if not isinstance(horse_data, dict): horse_data = {}
            horse_stats = start.get('horse_stats', {}).get('total', {})
            if not isinstance(horse_stats, dict): horse_stats = {}
            driver_stats = start.get('driver_stats', {}).get('allTotal', {})
            if not isinstance(driver_stats, dict): driver_stats = {}
            program_number = horse_data.get('programNumber')
            if program_number is None: logger.warning(f"Horse missing pgm# {race_date}/{track_code}/{start_number}. Skipping."); continue
            program_number_str = str(program_number)
            label = 0
            if race_results_api is not None:
                placing = race_results_api.get(program_number_str)
                if placing == '1': label = 1; win_count += 1
                # elif placing is None: logger.debug(f"Pgm# {program_number_str} not in API results {race_date}/{track_code}/{start_number}. Label 0.") # Reduce noise
            try:
                feature_vector = [ float(horse_stats.get('winningPercent', 0.0) or 0.0), float(horse_stats.get('priceMoney', 0.0) or 0.0),
                                   float(horse_stats.get('starts', 0.0) or 0.0), float(driver_stats.get('winPercentage', 0.0) or 0.0),
                                   float(driver_stats.get('priceMoney', 0.0) or 0.0), float(driver_stats.get('starts', 0.0) or 0.0),
                                   float(horse_data.get('lane', 0.0) or 0.0), float(track_id) ]
            except (ValueError, TypeError) as e: logger.warning(f"Feature conversion error {program_number_str}, {race_date}/{track_code}/{start_number}: {e}. Skipping."); continue
            features.append(feature_vector); labels.append(label)
    if not features: logger.warning("No valid features extracted in batch."); return None, None, track_map
    features_np = np.array(features, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int32)
    logger.info(f"Batch processed: {len(features_np)} entries. Wins (API Lookup): {win_count}. Missing Results: {missing_results_count}.")
    if scaler:
        try: features_np = scaler.transform(features_np)
        except Exception as e: logger.error(f"Scaler transform error: {e}"); return None, None, track_map
    return features_np, labels_np, track_map


def train_model_xgboost(data_path, model_path, scaler_path, results_lookup,
                        # Added parameters previously derived from args
                        sample_size=100000, val_split=0.2,
                        n_estimators=100, learning_rate=0.05, scale_pos_weight=10.0,
                        max_depth=6, subsample=0.8, colsample_bytree=0.8,
                        gpu_support=False, early_stopping_rounds=20):
    """Loads data, trains an XGBoost model using API-verified labels, and saves."""
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Detected {num_cores} CPU cores.")

    if results_lookup is None: logger.error("Results lookup missing."); return

    # --- 1. Fit Scaler ---
    scaler = StandardScaler(); track_map = {}
    sample_features_list, sample_labels_list = [], []
    sampled_count, sample_wins = 0, 0
    logger.info(f"Starting scaler fitting on sample ~{sample_size} entries...")
    try:
        batch_generator = stream_json_data(data_path, batch_size=10000)
        for batch_idx, race_batch in enumerate(batch_generator):
            features_np, labels_np, current_track_map = process_batch(race_batch, results_lookup, scaler=None, track_map=track_map)
            track_map.update(current_track_map)
            if features_np is not None and labels_np is not None and features_np.shape[0] > 0:
                sample_features_list.append(features_np); sample_labels_list.append(labels_np)
                sampled_count += features_np.shape[0]; batch_wins = int(np.sum(labels_np)); sample_wins += batch_wins
                logger.info(f"Scaler fitting: Batch {batch_idx+1}, accumulated {sampled_count}. Wins: {batch_wins}")
                if sampled_count >= sample_size: logger.info(f"Reached target sample size ({sampled_count})."); break
        if not sample_features_list: raise ValueError("No valid sample data found.")
        all_sample_features = np.vstack(sample_features_list)
        all_sample_labels = np.concatenate(sample_labels_list)
        logger.info(f"Fitting scaler on {all_sample_features.shape[0]} samples.")
        logger.info(f"Wins in scaler sample: {sample_wins} ({sample_wins/all_sample_features.shape[0]*100:.2f}%)")
        if sample_wins == 0: logger.warning("No wins found in scaler sample data.")
        scaler.fit(all_sample_features); logger.info("Scaler fitted.")
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
        scaler_data_path = scaler_path.replace('.pkl', '_data.json')
        scaler_data = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}
        with open(scaler_data_path, 'w') as f: json.dump(scaler_data, f, indent=2)
        track_map_path = scaler_path.replace('.pkl', '_track_map.json')
        with open(track_map_path, 'w') as f: json.dump(track_map, f, indent=2)
        logger.info(f"Scaler saved: {scaler_path}. Track map saved: {track_map_path}")
    except Exception as e: logger.error(f"Scaler fitting/saving error: {e}", exc_info=True); return

    # --- 2. Load Full Dataset ---
    logger.info("Collecting and scaling full dataset...")
    all_features_list, all_labels_list = [], []
    total_wins, total_samples = 0, 0
    try:
        batch_generator = stream_json_data(data_path, batch_size=10000)
        data_loader = tqdm(batch_generator, desc="Loading & Scaling Full Data")
        final_track_map = track_map.copy()
        for batch_idx, race_batch in enumerate(data_loader):
            features_np, labels_np, _ = process_batch(race_batch, results_lookup, scaler=scaler, track_map=final_track_map)
            if features_np is not None and labels_np is not None and features_np.shape[0] > 0:
                all_features_list.append(features_np); all_labels_list.append(labels_np)
                batch_wins = int(np.sum(labels_np)); total_wins += batch_wins
                total_samples += len(labels_np)
                data_loader.set_postfix({"wins_in_batch": batch_wins, "total_samples": total_samples})
        if not all_features_list: raise ValueError("No valid full data collected.")
        X = np.vstack(all_features_list); y = np.concatenate(all_labels_list)
        logger.info(f"Full dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
        logger.info(f"Total wins (Full): {total_wins} ({total_wins / X.shape[0] * 100:.2f}%)")
        if total_wins == 0: logger.error("CRITICAL: No wins in full dataset. Cannot train."); return
    except Exception as e: logger.error(f"Full data loading error: {e}", exc_info=True); return

    # --- 3. Split Data ---
    logger.info(f"Splitting data (Validation split: {val_split:.1f})")
    try:
        stratify_option = y if total_wins > 0 and total_wins < len(y) else None
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42, stratify=stratify_option)
        logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        y_train = np.asarray(y_train); y_val = np.asarray(y_val)
        logger.info(f"Wins in Train: {np.sum(y_train)}, Wins in Val: {np.sum(y_val)}")
        del X, y; import gc; gc.collect()
    except Exception as e: logger.error(f"Train/test split error: {e}", exc_info=True); return

    # --- 4. Configure and Train Model ---
    logger.info("Configuring XGBoost model...")
    xgb_params = { 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'eta': learning_rate,
                   'max_depth': max_depth, 'subsample': subsample, 'colsample_bytree': colsample_bytree,
                   'scale_pos_weight': scale_pos_weight, 'random_state': 42, 'nthread': max(1, num_cores - 1) }
    if gpu_support:
        xgb_params['device'] = 'cuda'; xgb_params['tree_method'] = 'hist'
        logger.info("Using GPU via device='cuda', tree_method='hist'.")
    else:
        xgb_params['tree_method'] = 'hist'; logger.info("Using CPU ('hist').")

    model = xgb.XGBClassifier(**xgb_params, n_estimators=n_estimators, use_label_encoder=False)
    logger.info(f"Starting training up to {n_estimators} rounds (early stopping: {early_stopping_rounds} rounds)...")
    logger.info(f"Params: {xgb_params}")
    try:
        eval_set = [(X_train, y_train), (X_val, y_val)] # Use already scaled data
        model.fit(
            X_train, y_train, eval_set=eval_set, # Pass X_train directly
            # Use callbacks for XGBoost >= 1.3 (which 3.0.0 is)
            #callbacks=[xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)],
            verbose=max(10, n_estimators // 10) if n_estimators > 0 else False
        )
        # comment erroring code
        #model.fit( X_train, y_train, eval_set=eval_set, # Pass X_train directly
        #           early_stopping_rounds=early_stopping_rounds,
        #           verbose=max(10, n_estimators // 10) if n_estimators > 0 else False )
        logger.info("XGBoost training finished (completed all rounds).")

        # --- 5. Evaluate ---
        results = model.evals_result()
        val_logloss_history = results['validation_1']['logloss']
        if not val_logloss_history: # Check if training failed immediately
                logger.error("No validation results recorded. Training likely failed.")
                return
        best_iteration = np.argmin(val_logloss_history)
        best_val_logloss = val_logloss_history[best_iteration]
        best_train_logloss = results['validation_0']['logloss'][best_iteration]
        logger.info(f"Training completed {len(val_logloss_history)} rounds.")
        logger.info(f"Best Iteration found via minimum validation_1 logloss: {best_iteration + 1}")
        logger.info(f"Training LogLoss at best iteration: {best_train_logloss:.4f}")
        logger.info(f"Validation LogLoss at best iteration: {best_val_logloss:.4f}")
        logger.info("Evaluating model calibration:")
        try:
            booster = model.get_booster()
            val_preds_proba = booster.predict(xgb.DMatrix(X_val), iteration_range=(0, best_iteration + 1)) # Use X_val
        except Exception as pred_err:
             logger.warning(f"Iteration predict failed ({pred_err}). Using final state."); val_preds_proba = model.predict_proba(X_val)[:, 1] # Use X_val
        bins = np.linspace(0, 1, 11); bin_indices = np.digitize(val_preds_proba, bins[1:], right=False)
        for i in range(len(bins)-1):
            bin_mask = (bin_indices == i); count_in_bin = bin_mask.sum()
            if count_in_bin > 0:
                avg_true_in_bin = y_val[bin_mask].mean(); avg_pred_in_bin = val_preds_proba[bin_mask].mean()
                logger.info(f"  Prob Range {bins[i]:.1f}-{bins[i+1]:.1f}: Avg Pred {avg_pred_in_bin:.3f}, Avg True {avg_true_in_bin:.3f}, Count {count_in_bin}")

        # --- 6. Save Model ---
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        logger.info(f"XGBoost model saved to: {model_path}")

    except xgb.core.XGBoostError as e: logger.error(f"XGBoost error: {e}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected training/eval error: {e}", exc_info=True)


# --- main function ---
def main():
    # --- Argument Parsing MOVED INSIDE main() ---
    parser = argparse.ArgumentParser(description="Train an 8-feature XGBoost model with API-verified labels")
    parser.add_argument('--data-path', type=str, default='./horse-race-predictor/racedata/scraped_race_data_2018-2024.json', help='Path to data JSON')
    parser.add_argument('--model-path', type=str, default='./horse-race-predictor/racedata/xgb/horse_race_predictor_8feat.ubj', help='Path to save model')
    parser.add_argument('--scaler-path', type=str, default='./horse-race-predictor/racedata/xgb/scaler_8feat.pkl', help='Path to save scaler')
    parser.add_argument('--results-lookup-path', type=str, default='./horse-race-predictor/racedata/xgb/race_results_lookup.pkl', help='Path to results lookup')
    parser.add_argument('--force-rebuild-lookup', action='store_true', help='Force rebuild lookup table')
    parser.add_argument('-d', '--debug', type=int, default=20, choices=[10, 20, 30, 40, 50], help="Debug level")
    parser.add_argument('--scale-pos-weight', type=float, default=10.0, help='XGBoost scale_pos_weight')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='XGBoost eta')
    parser.add_argument('--n-estimators', type=int, default=100, help='XGBoost n_estimators')
    parser.add_argument('--max-depth', type=int, default=6, help='XGBoost max_depth')
    parser.add_argument('--subsample', type=float, default=0.8, help='XGBoost subsample')
    parser.add_argument('--colsample-bytree', type=float, default=0.8, help='XGBoost colsample_bytree')
    parser.add_argument('--sample-size', type=int, default=100000, help='Sample size for scaler')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--early-stopping-rounds', type=int, default=20, help='Early stopping rounds')
    args = parser.parse_args()

    # --- Configure Logging AFTER parsing args ---
    logging.basicConfig(level=args.debug, format='%(levelname)s:%(name)s:%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # logger instance is already global

    logger.info("--- Starting Single XGBoost Training Run ---")
    logger.info("Arguments: %s", vars(args))

    # --- Define Paths ---
    RESULTS_LOOKUP_PATH = Path(args.results_lookup_path)

    # --- Check GPU Support ---
    try:
        import subprocess
        subprocess.check_output('nvidia-smi')
        gpu_support = True
        logger.info("NVIDIA GPU detected.")
    except Exception:
        gpu_support = False
        logger.info("No NVIDIA GPU detected.")

    # --- Manage Results Lookup Table ---
    results_lookup = None
    if RESULTS_LOOKUP_PATH.exists() and not args.force_rebuild_lookup:
        results_lookup = load_results_lookup(RESULTS_LOOKUP_PATH)
    else:
        if args.force_rebuild_lookup: logger.info("Forcing rebuild lookup.")
        else: logger.info(f"Lookup not found at {RESULTS_LOOKUP_PATH}. Building...")
        try: results_lookup = build_results_lookup(args.data_path, RESULTS_LOOKUP_PATH)
        except Exception as e: logger.error(f"Failed build lookup: {e}", exc_info=True); return
    if results_lookup is None: logger.error("Lookup failed. Cannot proceed."); return
    num_dates = len(results_lookup); num_races = sum(len(ts) for dt in results_lookup.values() for ts in dt.values())
    logger.info(f"Using lookup: {num_dates} dates, {num_races} starts.")

    # --- Proceed with Training ---
    logger.info(f"Starting training...")
    train_model_xgboost(
        data_path=args.data_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        results_lookup=results_lookup,
        sample_size=args.sample_size,
        val_split=args.val_split,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        scale_pos_weight=args.scale_pos_weight,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        gpu_support=gpu_support,
        early_stopping_rounds=args.early_stopping_rounds
    )
    logger.info("--- Training Run Finished ---")

# --- Main Guard ---
if __name__ == "__main__":
    main()

# --- END OF FILE xg_train.py (Corrected - Complete) ---
