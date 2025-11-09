# --- START OF FILE xg_train_19feat.py (Corrected Streamer) ---

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
logger = logging.getLogger(__name__)

# --- Keep Constants and API Setup at Top Level ---
BASE_URL = "https://heppa.hippos.fi/heppa2_backend"
HEADERS = {'Content-Type': 'application/json'}
CACHE_DIR = Path("./cache_train")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# RESULTS_LOOKUP_PATH defined inside main() based on args

# --- Helper Functions for Feature Engineering ---

def safe_float(value, default=0.0):
    """Safely convert value to float, handling None, empty strings, etc."""
    if value is None: return default
    try: return float(value)
    except (ValueError, TypeError): return default

def calculate_earnings_per_start(money, starts):
    """Calculate earnings per start, handling division by zero."""
    money_f = safe_float(money, 0.0); starts_f = safe_float(starts, 0.0)
    return (money_f / starts_f) if starts_f > 0 else 0.0

def get_stats_for_year_or_prev(stats_array, target_year_int):
    """Finds stats for the target year or the previous year if target year is missing/empty."""
    if not stats_array or not isinstance(stats_array, list): return None
    target_year_str = str(target_year_int)
    target_year_stats = next((s for s in stats_array if s.get('year') == target_year_str), None)
    if target_year_stats and safe_float(target_year_stats.get('starts', 0)) > 0: return target_year_stats
    prev_year_str = str(target_year_int - 1)
    prev_year_stats = next((s for s in stats_array if s.get('year') == prev_year_str), None)
    if prev_year_stats and safe_float(prev_year_stats.get('starts', 0)) > 0: return prev_year_stats
    return None

def calculate_derived_stats(year_stats_dict):
    """Calculates Win%, Place%, Starts, AvgMoney/Start from a yearly stats dict."""
    if not year_stats_dict or not isinstance(year_stats_dict, dict):
        return {'win_perc': 0.0, 'place_perc': 0.0, 'starts': 0.0, 'avg_money': 0.0}
    starts = safe_float(year_stats_dict.get('starts', 0))
    first = safe_float(year_stats_dict.get('firstPlaces', 0))
    second = safe_float(year_stats_dict.get('secondPlaces', 0))
    third = safe_float(year_stats_dict.get('thirdPlaces', 0))
    money = safe_float(year_stats_dict.get('priceMoney', 0))
    win_perc = (first / starts * 100) if starts > 0 else 0.0
    place_perc = ((first + second + third) / starts * 100) if starts > 0 else 0.0
    avg_money = (money / starts) if starts > 0 else 0.0
    return {'win_perc': win_perc, 'place_perc': place_perc, 'starts': starts, 'avg_money': avg_money}


# --- Core Data Processing & Training Functions ---

def fetch_api_data(endpoint, base_url=BASE_URL, headers=HEADERS, cache_dir=CACHE_DIR, retries=2, delay=5):
    """Fetch data from API with caching and retries."""
    safe_endpoint = "".join(c if c.isalnum() else "_" for c in endpoint)
    cache_key = f"api_{safe_endpoint}"
    if len(cache_key) > 150: import hashlib; cache_key = f"api_{hashlib.md5(endpoint.encode()).hexdigest()}"
    cache_file = Path(cache_dir) / f"{cache_key}.json"
    if cache_file.exists():
        try:
            logger.debug(f"Loading cached data for {endpoint} from {cache_file}")
            with open(cache_file, 'r') as f: return json.load(f)
        except json.JSONDecodeError: logger.warning(f"Corrupt cache: {cache_file}. Re-fetching."); cache_file.unlink()
        except Exception as e: logger.error(f"Cache read error {cache_file}: {e}. Re-fetching.")
    url = f"{base_url}{endpoint}"
    for attempt in range(retries + 1):
        try:
            logger.debug(f"Fetching (attempt {attempt+1}): {url}")
            response = requests.get(url, headers=headers, timeout=20)
            if '<html' in response.text[:100].lower():
                 logger.error(f"HTML response from {url}. Status: {response.status_code}.");
                 if attempt < retries: 
                     time.sleep(delay)
                     continue
                 else: 
                     return None
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Success: {endpoint}. Status: {response.status_code}")
            try:
                with open(cache_file, 'w') as f: json.dump(data, f)
                logger.debug(f"Cached data to {cache_file}")
            except Exception as e: logger.error(f"Cache write error {cache_file}: {e}")
            return data
        except requests.exceptions.Timeout:
             logger.warning(f"Timeout {url} on attempt {attempt+1}")
             if attempt == retries: logger.error(f"Final timeout: {url}"); return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request Exception {url}: {e}")
            if attempt == retries: logger.error(f"Final Request Exception: {url}"); return None
        except json.JSONDecodeError as e:
             logger.error(f"JSON Decode Error from {url}. Status: {response.status_code}. Error: {e}.")
             return None # Bad response
        if attempt < retries: logger.info(f"Waiting {delay}s before retrying {url}..."); time.sleep(delay)
    return None

def get_race_results_from_api(race_date, track_code, start_number):
    """Fetches results for a specific race start and returns a mapping {prog_num: placing}."""
    endpoint = f"/race/{race_date}/{track_code}/start/{start_number}"
    horses_data = fetch_api_data(endpoint)
    if not horses_data or not isinstance(horses_data, list):
        logger.warning(f"No valid horse data via API for {race_date}/{track_code}/{start_number}"); return None
    results = {}
    for horse in horses_data:
        if isinstance(horse, dict):
            prog_num = horse.get('programNumber'); placing = horse.get('placing')
            if prog_num is not None: results[str(prog_num)] = placing
        else: logger.warning(f"Unexpected item type {type(horse)} in API horse list for {race_date}/{track_code}/{start_number}")
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
        logger.info(f"Found {len(unique_races)} unique starts after scanning {total_races_scanned} JSON entries.")
    except FileNotFoundError: logger.error(f"Training data file not found: {data_path}"); raise
    except Exception as e: logger.error(f"Error scanning JSON: {e}", exc_info=True); raise
    results_lookup = {}; api_failures = 0
    logger.info(f"Fetching results from API for {len(unique_races)} unique races...")
    for race_date, track_code, start_number in tqdm(unique_races, desc="Fetching API Results"):
        race_results = get_race_results_from_api(race_date, track_code, start_number)
        if race_results is not None:
            if race_date not in results_lookup: results_lookup[race_date] = {}
            if track_code not in results_lookup[race_date]: results_lookup[race_date][track_code] = {}
            results_lookup[race_date][track_code][start_number] = race_results
        else: api_failures += 1; logger.warning(f"Failed API lookup for {race_date}/{track_code}/{start_number}.")
    logger.info(f"Finished fetching API results. Success: {len(unique_races) - api_failures}, Failures: {api_failures}")
    try:
        logger.info(f"Saving results lookup: {lookup_save_path}...")
        lookup_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lookup_save_path, 'wb') as f: pickle.dump(results_lookup, f)
        logger.info("Lookup saved.")
    except Exception as e: logger.error(f"Failed saving lookup: {e}", exc_info=True)
    return results_lookup

def load_results_lookup(lookup_load_path):
    """Loads the precomputed results lookup table."""
    lookup_load_path = Path(lookup_load_path)
    if not lookup_load_path.exists(): logger.error(f"Lookup file not found: {lookup_load_path}"); return None
    try:
        logger.info(f"Loading lookup table: {lookup_load_path}...")
        with open(lookup_load_path, 'rb') as f: results_lookup = pickle.load(f)
        logger.info("Lookup loaded."); return results_lookup
    except Exception as e: logger.error(f"Failed loading lookup: {e}", exc_info=True); return None

# --- CORRECTED stream_json_data ---
def stream_json_data(file_path, batch_size=10000):
    """Streams race data, adding context. Corrected Scope."""
    try:
        with open(file_path, 'rb') as f:
            # Assuming top level is a list of monthly dictionaries
            parser = ijson.items(f, 'item');
            races_batch = [];
            total_races_yielded = 0
            logger.info("Starting JSON stream parsing for data processing...")

            for monthly_dict in parser:
                if not isinstance(monthly_dict, dict):
                    # This might happen if the top level list contains non-dicts
                    logger.warning(f"Skipping non-dictionary item in top-level list: {type(monthly_dict)}")
                    continue

                for date_key, date_data in monthly_dict.items():
                    # Validate date_key and ensure date_data is a dictionary
                    if not isinstance(date_key, str) or not date_key:
                        logger.warning(f"Invalid or missing date key: {date_key}. Skipping associated data.")
                        continue
                    if not isinstance(date_data, dict):
                        logger.warning(f"Data for date '{date_key}' is not a dictionary ({type(date_data)}). Skipping.")
                        continue
                    current_date = date_key # Assign valid date key

                    for track_code, track_data in date_data.items():
                         # Validate track_code and ensure track_data is a dictionary
                         if not isinstance(track_code, str) or not track_code:
                             logger.warning(f"Invalid or missing track code for date {current_date}: {track_code}. Skipping associated data.")
                             continue
                         if not isinstance(track_data, dict):
                            logger.warning(f"Data for track '{track_code}' on date '{current_date}' is not a dictionary ({type(track_data)}). Skipping.")
                            continue
                         current_track = track_code # Assign valid track code

                         # Now current_date and current_track are guaranteed to be valid strings
                         # Iterate through starts for this specific date/track
                         for start_number, race_data in track_data.items():
                             # Check if it looks like a valid race data entry
                             if isinstance(race_data, dict) and ('race_info' in race_data or 'starts' in race_data):
                                 # Create and add context
                                 race_context = {'date': current_date, 'track': current_track, 'start': str(start_number)}
                                 race_data['_context'] = race_context # Add it to the race data

                                 races_batch.append(race_data)
                                 total_races_yielded += 1
                                 # Yield batch if full
                                 if len(races_batch) >= batch_size:
                                     logger.debug(f"Yielding batch {len(races_batch)}. Total: {total_races_yielded}")
                                     yield races_batch; races_batch = []
                             else:
                                 logger.debug(f"Skipping non-race data for start_num '{start_number}', track '{current_track}', date '{current_date}': Type={type(race_data)}")

            # Handle final batch
            if races_batch:
                logger.debug(f"Yielding final batch {len(races_batch)}. Total: {total_races_yielded}")
                yield races_batch

        logger.info(f"Finished streaming JSON. Total valid race entries yielded: {total_races_yielded}")

        # Check if any data was actually yielded
        if total_races_yielded == 0:
             logger.error("Streaming finished, but ZERO valid race entries were yielded. Check JSON structure and parsing logic against the input file.")
             raise ValueError("No valid race entries yielded from JSON.")

    except FileNotFoundError: logger.error(f"Data file not found: {file_path}"); raise
    except Exception as e: logger.error(f"Error reading/parsing JSON file {file_path}: {e}", exc_info=True); raise
# --- End CORRECTED stream_json_data ---


# --- process_batch (MODIFIED FOR 19 FEATURES) ---
def process_batch(races_with_context, results_lookup, scaler=None, track_map=None):
    """Processes a batch of race dicts to extract 19 features and CORRECT labels."""
    features = []; labels = []; win_count = 0; missing_results_count = 0
    if track_map is None: track_map = {}
    INPUT_SIZE = 19 # Define expected size

    for race_entry in races_with_context:
        context = race_entry.get('_context')
        if not context: logger.warning("Skipping entry missing _context"); continue
        race_date = context['date']; track_code = context['track']; start_number = context['start']

        race_info = race_entry.get('race_info', {})
        starts = race_entry.get('starts', [])
        race_details = race_info.get('race', {})
        race_distance = safe_float(race_details.get('distance'), 0.0)

        race_results_api = results_lookup.get(race_date, {}).get(track_code, {}).get(start_number)
        if race_results_api is None: missing_results_count += len(starts)

        if not starts: continue

        if track_code not in track_map: track_map[track_code] = len(track_map)
        track_id = track_map[track_code]

        try:
            race_year_int = int(race_date.split('-')[0])
        except:
            logger.warning(f"Could not parse year from race_date: {race_date}. Skipping race.")
            continue

        for start in starts:
            if not isinstance(start, dict): continue

            horse_data = start.get('horse_data', {})
            horse_stats_full = start.get('horse_stats', {})
            driver_stats_full = start.get('driver_stats', {})

            if not isinstance(horse_data, dict): horse_data = {}
            if not isinstance(horse_stats_full, dict): horse_stats_full = {}
            if not isinstance(driver_stats_full, dict): driver_stats_full = {}

            program_number = horse_data.get('programNumber')
            if program_number is None: continue # Skip if no program number
            program_number_str = str(program_number)

            label = 0
            if race_results_api is not None:
                placing = race_results_api.get(program_number_str)
                if placing == '1': label = 1; win_count += 1

            try:
                # --- Original 8 Features ---
                horse_total = horse_stats_full.get('total', {})
                driver_alltotal = driver_stats_full.get('allTotal', {}) # Using allTotal based on previous scripts
                f1 = safe_float(horse_total.get('winningPercent'))
                f2 = safe_float(horse_total.get('priceMoney'))
                f3 = safe_float(horse_total.get('starts'))
                f4 = safe_float(driver_alltotal.get('winPercentage'))
                f5 = safe_float(driver_alltotal.get('priceMoney'))
                f6 = safe_float(driver_alltotal.get('starts'))
                f7 = safe_float(horse_data.get('lane'))
                f8 = float(track_id)

                # --- New Features (Calculated) ---
                horse_yearly_stats_list = horse_stats_full.get('stats', [])
                driver_yearly_stats_list = driver_stats_full.get('horseStats', [])

                recent_horse_stats_dict = get_stats_for_year_or_prev(horse_yearly_stats_list, race_year_int)
                recent_driver_stats_dict = get_stats_for_year_or_prev(driver_yearly_stats_list, race_year_int)

                recent_horse_metrics = calculate_derived_stats(recent_horse_stats_dict)
                recent_driver_metrics = calculate_derived_stats(recent_driver_stats_dict)

                f9  = recent_horse_metrics['win_perc']
                f10 = recent_horse_metrics['place_perc'] # Top 3 Place %
                f11 = recent_horse_metrics['starts']
                f12 = recent_horse_metrics['avg_money']
                f13 = recent_driver_metrics['win_perc']
                f14 = recent_driver_metrics['starts']
                # Feature 15 (Days Since Last Race) - Skipped

                # Context / Other Derived Features
                f15 = race_distance
                f16 = calculate_earnings_per_start(f2, f3) # Horse total earnings/start
                f17 = calculate_earnings_per_start(f5, f6) # Driver total earnings/start
                f18 = safe_float(horse_total.get('gallopPercentage')) # Risk factor
                f19 = safe_float(horse_total.get('disqualificationPercentage')) # Risk factor

                feature_vector = [
                    f1, f2, f3, f4, f5, f6, f7, f8,
                    f9, f10, f11, f12, f13, f14,
                    f15, f16, f17, f18, f19
                ]
                if len(feature_vector) != INPUT_SIZE: raise ValueError(f"Incorrect feature count: {len(feature_vector)}")

            except Exception as e:
                logger.warning(f"Feature extraction error {program_number_str} @ {race_date}/{track_code}/{start_number}: {e}. Skipping.", exc_info=logger.isEnabledFor(logging.DEBUG))
                continue

            features.append(feature_vector)
            labels.append(label)

    if not features: logger.warning("No valid features extracted in batch."); return None, None, track_map
    features_np = np.array(features, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int32)
    logger.info(f"Batch processed: {len(features_np)} entries. Wins: {win_count}. Missing Results: {missing_results_count}.")
    if scaler:
        try: features_np = scaler.transform(features_np)
        except Exception as e: logger.error(f"Scaler transform error: {e}"); return None, None, track_map
    return features_np, labels_np, track_map


# --- Training Function ---
def train_model_xgboost(data_path, model_path, scaler_path, results_lookup,
                        sample_size=100000, val_split=0.2,
                        n_estimators=100, learning_rate=0.05, scale_pos_weight=10.0,
                        max_depth=6, subsample=0.8, colsample_bytree=0.8,
                        gpu_support=False, early_stopping_rounds=20):
    """Trains a single XGBoost model with specified parameters and 19 features."""
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Detected {num_cores} CPU cores.")
    INPUT_SIZE = 19 # Update feature count reference
    logger.info(f"Expecting {INPUT_SIZE} features.")

    if results_lookup is None: logger.error("Results lookup missing."); return

    # --- 1. Fit Scaler ---
    scaler = StandardScaler(); track_map = {}
    sample_features_list, sample_labels_list = [], []; sampled_count, sample_wins = 0, 0
    logger.info(f"Starting scaler fitting on sample ~{sample_size} entries...")
    try:
        batch_generator = stream_json_data(data_path, batch_size=10000)
        for batch_idx, race_batch in enumerate(batch_generator):
            features_np, labels_np, current_track_map = process_batch(race_batch, results_lookup, scaler=None, track_map=track_map)
            track_map.update(current_track_map)
            if features_np is not None and labels_np is not None and features_np.shape[0] > 0:
                if features_np.shape[1] != INPUT_SIZE: logger.error(f"Scaler Fit: Feature shape mismatch! Expected {INPUT_SIZE}, got {features_np.shape[1]}."); return
                sample_features_list.append(features_np); sample_labels_list.append(labels_np)
                sampled_count += features_np.shape[0]; batch_wins = int(np.sum(labels_np)); sample_wins += batch_wins
                logger.info(f"Scaler fitting: Batch {batch_idx+1}, accumulated {sampled_count}. Wins: {batch_wins}")
                if sampled_count >= sample_size: logger.info(f"Reached target sample size ({sampled_count})."); break
        if not sample_features_list: raise ValueError("No valid sample data found.")
        all_sample_features = np.vstack(sample_features_list)
        logger.info(f"Fitting scaler on {all_sample_features.shape[0]} samples with {all_sample_features.shape[1]} features.")
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
                if features_np.shape[1] != INPUT_SIZE: logger.error(f"Full Load: Feature shape mismatch! Expected {INPUT_SIZE}, got {features_np.shape[1]}."); return
                all_features_list.append(features_np); all_labels_list.append(labels_np)
                batch_wins = int(np.sum(labels_np)); total_wins += batch_wins
                total_samples += len(labels_np)
                data_loader.set_postfix({"wins_in_batch": batch_wins, "total_samples": total_samples})
        if not all_features_list: raise ValueError("No valid full data collected.")
        X = np.vstack(all_features_list); y = np.concatenate(all_labels_list)
        logger.info(f"Full dataset loaded: {X.shape[0]} samples, {X.shape[1]} features. Wins: {total_wins} ({total_wins / X.shape[0] * 100:.2f}%)")
        if total_wins == 0: logger.error("CRITICAL: No wins in full dataset."); return
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
    if gpu_support: xgb_params['device'] = 'cuda'; xgb_params['tree_method'] = 'hist'; logger.info("Using GPU.")
    else: xgb_params['tree_method'] = 'hist'; logger.info("Using CPU.")
    model = xgb.XGBClassifier(**xgb_params, n_estimators=n_estimators, use_label_encoder=False)
    logger.info(f"Starting training for {n_estimators} rounds (manual best iter check)...")
    logger.info(f"Params: {xgb_params}")
    try:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        # Fit WITHOUT early stopping parameters
        model.fit( X_train, y_train, eval_set=eval_set,
                   verbose=max(10, n_estimators // 10) if n_estimators > 0 else False )
        logger.info("XGBoost training finished (completed all rounds).")

        # --- 5. Evaluate ---
        results = model.evals_result()
        val_logloss_history = results['validation_1']['logloss']
        if not val_logloss_history: logger.error("No validation results recorded."); return
        best_iteration = np.argmin(val_logloss_history)
        best_val_logloss = val_logloss_history[best_iteration]
        best_train_logloss = results['validation_0']['logloss'][best_iteration]
        logger.info(f"Training completed {len(val_logloss_history)} rounds.")
        logger.info(f"Best Iteration found via minimum validation_1 logloss: {best_iteration + 1}")
        logger.info(f"LogLoss at best iteration: Train={best_train_logloss:.5f}, Val={best_val_logloss:.5f}")
        logger.info("Evaluating model calibration:")
        try:
            booster = model.get_booster()
            val_preds_proba = booster.predict(xgb.DMatrix(X_val), iteration_range=(0, best_iteration + 1))
        except Exception as pred_err:
             logger.warning(f"Iteration predict failed ({pred_err}). Using final state."); val_preds_proba = model.predict_proba(X_val)[:, 1]
        bins = np.linspace(0, 1, 11); bin_indices = np.digitize(val_preds_proba, bins[1:], right=False)
        for i in range(len(bins)-1):
            bin_mask = (bin_indices == i); count_in_bin = bin_mask.sum()
            if count_in_bin > 0:
                avg_true_in_bin = y_val[bin_mask].mean(); avg_pred_in_bin = val_preds_proba[bin_mask].mean()
                logger.info(f"  Prob Range {bins[i]:.1f}-{bins[i+1]:.1f}: Avg Pred {avg_pred_in_bin:.3f}, Avg True {avg_true_in_bin:.3f}, Count {count_in_bin}")

        # --- 6. Save Model ---
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        logger.info(f"XGBoost model (final state) saved to: {model_path}")
        best_iter_path = model_path.replace('.ubj', '_best_iter.json')
        with open(best_iter_path, 'w') as f: json.dump({'best_iteration': int(best_iteration) + 1}, f)

    except xgb.core.XGBoostError as e: logger.error(f"XGBoost error: {e}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected training/eval error: {e}", exc_info=True)


# --- main function ---
def main():
    # --- Argument Parsing MOVED INSIDE main() ---
    parser = argparse.ArgumentParser(description="Train a 19-feature XGBoost model with API-verified labels")
    # Update defaults for 19 features conceptually
    parser.add_argument('--data-path', type=str, default='./horse-race-predictor/racedata/scraped_race_data_2018-2024.json', help='Path to data JSON')
    parser.add_argument('--model-path', type=str, default='./horse-race-predictor/racedata/19feat/xgb/model_19feat.ubj', help='Path to save model')
    parser.add_argument('--scaler-path', type=str, default='./horse-race-predictor/racedata/19feat/xgb/scaler_19feat.pkl', help='Path to save scaler')
    parser.add_argument('--results-lookup-path', type=str, default='./horse-race-predictor/racedata/xgb/race_results_lookup.pkl', help='Path to results lookup') # Keep using the same lookup
    parser.add_argument('--force-rebuild-lookup', action='store_true', help='Force rebuild lookup table')
    parser.add_argument('-d', '--debug', type=int, default=20, choices=[10, 20, 30, 40, 50], help="Debug level")
    parser.add_argument('--scale-pos-weight', type=float, default=4.0, help='XGBoost scale_pos_weight') # Use previously found best?
    parser.add_argument('--learning-rate', type=float, default=0.01, help='XGBoost eta') # Use previously found best?
    parser.add_argument('--n-estimators', type=int, default=2500, help='XGBoost n_estimators') # Use best iteration range?
    parser.add_argument('--max-depth', type=int, default=5, help='XGBoost max_depth') # Use previously found best?
    parser.add_argument('--subsample', type=float, default=0.7, help='XGBoost subsample') # Use previously found best?
    parser.add_argument('--colsample-bytree', type=float, default=0.8, help='XGBoost colsample_bytree') # Use previously found best?
    parser.add_argument('--sample-size', type=int, default=100000, help='Sample size for scaler')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--early-stopping-rounds', type=int, default=50, help='Early stopping rounds (for logging)') # Patience for finding best iter
    args = parser.parse_args()

    # --- Configure Logging AFTER parsing args ---
    logging.basicConfig(level=args.debug, format='%(levelname)s:%(name)s:%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # logger instance is already global

    logger.info("--- Starting Single XGBoost Training Run (19 Features) ---")
    logger.info("Arguments: %s", vars(args))

    # --- Define Paths ---
    RESULTS_LOOKUP_PATH = Path(args.results_lookup_path)
    MODEL_PATH = Path(args.model_path); MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALER_PATH = Path(args.scaler_path); SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)


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
        model_path=args.model_path, # Pass Path object
        scaler_path=args.scaler_path, # Pass Path object
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

# --- END OF FILE ---
