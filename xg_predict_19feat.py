# --- START OF FILE xg_predict_19feat.py ---

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import requests
from tqdm import tqdm
import logging
import json
import os
import pickle
import argparse
from datetime import datetime
from pathlib import Path
import time
import pandas as pd

# --- Helper Functions (Copied from xg_train_19feat.py) ---

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

# --- End Helper Functions ---


# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Predict horse race outcomes with a 19-feature XGBoost model")
parser.add_argument('date', type=str, help='Date in YYYY-MM-DD format')
parser.add_argument('--track', type=str, default=None, help='Track code (e.g., "H"); if not provided, fetched from API')
# --- UPDATED DEFAULT PATHS ---
parser.add_argument('--model-path', type=str, default='./horse-race-predictor/racedata/19feat/xgb/model_19feat.ubj',
                    help='Path to trained 19-feature XGBoost model file')
parser.add_argument('--scaler-path', type=str, default='./horse-race-predictor/racedata/19feat/xgb/scaler_19feat.pkl',
                    help='Path to 19-feature scaler pickle file')
parser.add_argument('--track-map-path', type=str, default='./horse-race-predictor/racedata/19feat/xgb/scaler_19feat_track_map.json',
                    help='Path to 19-feature track map JSON file')
parser.add_argument('--best-iter-path', type=str, default=None,
                    help='Optional path to _best_iter.json file to limit prediction rounds')
# --- END UPDATED ---
parser.add_argument('--output-file', type=str, default=None, help='Path to save predictions as JSON (optional)')
parser.add_argument('--debug', type=int, default=20, choices=[10, 20, 30, 40, 50],
                    help='Debug level: 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL (default: 20)')
parser.add_argument('--top-n-table', type=int, default=5, metavar='N',
                    help='Print a table showing the top N horses per race (default: 5). Set to 0 to disable.')

args = parser.parse_args()

# --- Setup logging, Constants, API Fetching ---
logging.basicConfig(level=args.debug, format='%(levelname)s:%(name)s:%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
BASE_URL = "https://heppa.hippos.fi/heppa2_backend"
HEADERS = {'Content-Type': 'application/json'}
CACHE_DIR = Path("./cache_predict_xgb") # Keep same cache dir
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INPUT_SIZE = 19 # Define expected feature size

# --- fetch_api_data function (Keep existing code from previous version) ---
def fetch_api_data(endpoint, base_url=BASE_URL, headers=HEADERS, cache_dir=CACHE_DIR, retries=2, delay=5):
    safe_endpoint = "".join(c if c.isalnum() else "_" for c in endpoint)
    cache_key = f"api_{safe_endpoint}"
    if len(cache_key) > 150: import hashlib; cache_key = f"api_{hashlib.md5(endpoint.encode()).hexdigest()}"
    cache_file = Path(cache_dir) / f"{cache_key}.json"
    if cache_file.exists():
        try:
            logger.debug(f"Cache HIT: {endpoint} from {cache_file}")
            with open(cache_file, 'r') as f: return json.load(f)
        except Exception as e: logger.warning(f"Cache read error {cache_file}: {e}. Re-fetching."); cache_file.unlink()
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
            data = response.json(); logger.debug(f"Fetch SUCCESS: {endpoint}. Status: {response.status_code}")
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
             logger.error(f"JSON Decode Error from {url}. Status: {response.status_code}. Error: {e}."); return None
        if attempt < retries: logger.info(f"Waiting {delay}s before retrying {url}..."); time.sleep(delay)
    return None

# --- fetch_races function (Modified to store full stats and distance) ---
def fetch_races(date, track):
    """Fetches and enriches race data for 19-feature prediction."""
    enriched_races = []
    logger.info(f"Fetching starts for {date}/{track} using /races endpoint...")
    starts_info = fetch_api_data(f"/race/{date}/{track}/races")
    if not starts_info or not isinstance(starts_info, list):
        logger.warning(f"No valid starts list found for {date}/{track} via /races")
        return None

    for start_summary in tqdm(starts_info, desc=f"Fetching Details for {track}"):
        race_details = start_summary.get('race', {})
        start_number = race_details.get('startNumber', '')
        race_distance = race_details.get('distance') # Get distance here
        if not start_number: continue

        logger.debug(f"Fetching horses for Start {start_number}")
        horses_in_start = fetch_api_data(f"/race/{date}/{track}/start/{start_number}")
        if not horses_in_start or not isinstance(horses_in_start, list):
            logger.warning(f"No horses found for {date}/{track}/start/{start_number}"); continue

        for horse_data in horses_in_start:
            if not isinstance(horse_data, dict): continue
            horse_id = horse_data.get('horseId', ''); driver_id = horse_data.get('driverId', '')

            # Store the FULL stats objects fetched from API
            horse_stats_full = fetch_api_data(f"/horse/{horse_id}/stats") if horse_id else {}
            driver_stats_full = fetch_api_data(f"/driver/{driver_id}/stats") if driver_id else {}

            enriched_races.append({
                'raceDate': date, 'trackCode': track, 'startNumber': start_number,
                'raceDistance': race_distance, # Add race distance
                'horse_data': horse_data,
                'horse_stats_full': horse_stats_full or {}, # Store full object
                'driver_stats_full': driver_stats_full or {} # Store full object
            })
            # time.sleep(0.02) # Optional delay

    logger.info(f"Fetched and enriched data for {len(enriched_races)} horse entries.")
    return enriched_races

# --- process_prediction_data function (CORRECTED for 19 features & arguments) ---
def process_prediction_data(enriched_api_data, scaler, track_map, loaded_default_track_id):
    """Processes API data into 19 scaled features for prediction."""
    raw_features = []
    horse_info_list = []
    # No need to calculate default_track_id, it's passed as loaded_default_track_id
    unknown_track_logged = set()
    logger.info(f"Processing {len(enriched_api_data)} fetched entries into 19 features...")
    INPUT_SIZE = 19 # Define expected size locally for checks

    for entry in tqdm(enriched_api_data, desc="Processing Features"):
        horse_data = entry.get('horse_data', {})
        horse_stats_full = entry.get('horse_stats_full', {}) # Use full stats
        driver_stats_full = entry.get('driver_stats_full', {}) # Use full stats
        track_code = entry.get('trackCode', 'Unknown')
        race_date = entry.get('raceDate')
        race_distance = safe_float(entry.get('raceDistance'), 0.0) # Get distance

        if not isinstance(horse_data, dict): horse_data = {}
        if not isinstance(horse_stats_full, dict): horse_stats_full = {}
        if not isinstance(driver_stats_full, dict): driver_stats_full = {}

        # Map Track Code using the passed-in default ID
        if track_code in track_map:
            track_id = track_map[track_code]
        else:
            track_id = loaded_default_track_id # Use the passed-in argument
            if track_code not in unknown_track_logged:
                logger.warning(f"Track '{track_code}' not in map. Using default ID: {track_id}.")
                unknown_track_logged.add(track_code)

        # Get Race Year for calculating recent stats
        try:
            race_year_int = int(race_date.split('-')[0])
        except:
            logger.warning(f"Invalid race date {race_date} in entry. Cannot calculate recent stats.")
            race_year_int = None

        try:
            # --- Original 8 Features ---
            horse_total = horse_stats_full.get('total', {})
            driver_alltotal = driver_stats_full.get('allTotal', {})
            f1 = safe_float(horse_total.get('winningPercent'))
            f2 = safe_float(horse_total.get('priceMoney'))
            f3 = safe_float(horse_total.get('starts'))
            f4 = safe_float(driver_alltotal.get('winPercentage'))
            f5 = safe_float(driver_alltotal.get('priceMoney'))
            f6 = safe_float(driver_alltotal.get('starts'))
            f7 = safe_float(horse_data.get('lane'))
            f8 = float(track_id)

            # --- New Features (Calculated) ---
            recent_horse_metrics = {'win_perc': 0.0, 'place_perc': 0.0, 'starts': 0.0, 'avg_money': 0.0}
            recent_driver_metrics = {'win_perc': 0.0, 'place_perc': 0.0, 'starts': 0.0, 'avg_money': 0.0}

            if race_year_int is not None:
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
            # Feature 15 (Days Since Last Race) - Still Skipped

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
            if len(feature_vector) != INPUT_SIZE:
                raise ValueError(f"Incorrect feature count: {len(feature_vector)}")

            raw_features.append(feature_vector)
            # Store corresponding info
            horse_info_list.append({
                'date': entry.get('raceDate', 'N/A'), 'track': track_code, 'start': entry.get('startNumber', 'N/A'),
                'program_number': horse_data.get('programNumber', 'N/A'), 'horse': horse_data.get('horseName', 'N/A'),
                'driver': horse_data.get('driverName', 'N/A'), 'lane': horse_data.get('lane', 'N/A')
            })

        except (ValueError, TypeError) as e:
            logger.warning(f"Feature conversion/calc error for {horse_data.get('horseName', '?')}: {e}. Skipping.", exc_info=logger.isEnabledFor(logging.DEBUG))
            continue
        except Exception as e_gen:
             logger.error(f"General Feature Error for {horse_data.get('horseName', '?')}: {e_gen}. Skipping.", exc_info=True)
             continue

    if not raw_features: logger.error("No valid features extracted."); return None, None
    features_np = np.array(raw_features, dtype=np.float32)
    logger.info(f"Extracted raw features shape: {features_np.shape}")
    if features_np.shape[1] != INPUT_SIZE:
         logger.error(f"Final feature shape mismatch! Expected {INPUT_SIZE}, got {features_np.shape[1]}. Cannot proceed.")
         return None, None
    try:
        scaled_features = scaler.transform(features_np)
        logger.info("Applied scaler transform."); return scaled_features, horse_info_list
    except Exception as e:
        logger.error(f"Scaler transform error: {e}", exc_info=True); return None, None

# --- predict_xgb function (Modified to use Booster) ---
def predict_xgb(booster_model, features, best_iteration=None):
    """Generates predictions using the loaded XGBoost Booster."""
    if features is None or features.shape[0] == 0: return None
    try:
        app_logger = logging.getLogger(__name__) # Use logger defined globally
        app_logger.info(f"Predicting probabilities for {features.shape[0]} entries...")
        dmatrix = xgb.DMatrix(features)
        iteration_args = {}
        if best_iteration is not None:
            iteration_args['iteration_range'] = (0, best_iteration) # Use 0-based index + 1 for range
            app_logger.debug(f"Predicting using iteration_range={iteration_args['iteration_range']}")
        else:
             app_logger.debug("Predicting using all booster trees.")

        probabilities = booster_model.predict(dmatrix, **iteration_args)
        app_logger.info("Prediction complete."); return probabilities
    except Exception as e: app_logger.error(f"XGBoost prediction error: {e}", exc_info=True); return None

# --- get_track_code function (Keep existing code) ---
def get_track_code(date):
    # ... (function code remains the same) ...
    endpoint = f"/race/search/{date}/{date}/"; data = fetch_api_data(endpoint)
    if data and isinstance(data, list) and len(data)>0 and data[0].get('events'):
        tracks = [e['trackCode'] for e in data[0]['events'] if isinstance(e,dict)];
        if tracks: logger.debug(f"Found tracks for {date}: {tracks}"); return tracks
    logger.error(f"No tracks for {date}."); return None

# --- create_top_n_table function (Keep existing code) ---
def create_top_n_table(predictions_list, n=5):
    # ... (function code remains the same) ...
    if not predictions_list or n <= 0: return ""
    try:
        df = pd.DataFrame(predictions_list)
        if df.empty: return "No data."
        df['start_num'] = pd.to_numeric(df['start'], errors='coerce').fillna(0).astype(int)
        all_tables = []
        for name, group in df.groupby(['date', 'track', 'start_num']):
            race_date, track_code, start_num = name
            header = f"\n--- Race {start_num} ({track_code} / {race_date}) Top {n} ---\n"
            top_n = group.sort_values('normalized_win_probability', ascending=False).head(n).copy()
            top_n['Prob (%)'] = (top_n['normalized_win_probability'] * 100).map('{:.2f}%'.format)
            top_n_formatted = top_n[['program_number', 'horse', 'driver', 'Prob (%)']]
            top_n_formatted.rename(columns={'program_number': 'Pgm', 'horse': 'Horse', 'driver': 'Driver'}, inplace=True)
            all_tables.append(header + top_n_formatted.to_string(index=False, justify='left'))
        if not all_tables: return "No groups for table."
        return "\n".join(all_tables) + "\n--- End Top N Table ---"
    except Exception as e: logger.error(f"Error creating top N table: {e}"); return "Error."


# --- Main Execution Logic ---
def main():
    date = args.date
    try: datetime.strptime(date, "%Y-%m-%d")
    except ValueError: logger.error("Invalid date format. Use YYYY-MM-DD."); return

    track = args.track
    if not track:
        logger.info(f"No track specified, trying API lookup for {date}...")
        available_tracks = get_track_code(date)
        if available_tracks: track = available_tracks[0]; logger.info(f"Using track: {track}")
        else: logger.error(f"Cannot determine track for {date}. Use --track."); return

    # --- Load Artifacts (Scaler, Track Map, Model Booster) ---
    scaler, track_map, model_booster, best_iter = None, None, None, None
    default_track_id = 0 # Initialize
    try: # Scaler
        if not os.path.exists(args.scaler_path): raise FileNotFoundError(f"Scaler not found: {args.scaler_path}")
        with open(args.scaler_path, 'rb') as f: scaler = pickle.load(f)
        logger.info(f"Loaded scaler: {args.scaler_path}")
    except Exception as e: logger.error(f"Failed loading scaler: {e}", exc_info=True); return
    try: # Track Map
        if not os.path.exists(args.track_map_path): raise FileNotFoundError(f"Track map not found: {args.track_map_path}")
        with open(args.track_map_path, 'r') as f: track_map = json.load(f)
        default_track_id = len(track_map) # Set default ID based on loaded map
        logger.info(f"Loaded track map: {args.track_map_path} ({len(track_map)} tracks)")
    except Exception as e: logger.error(f"Failed loading track map: {e}", exc_info=True); return
    try: # Model Booster
        if not os.path.exists(args.model_path): raise FileNotFoundError(f"Model not found: {args.model_path}")
        model_booster = xgb.Booster(); model_booster.load_model(args.model_path)
        logger.info(f"Loaded XGBoost model booster: {args.model_path}")
        # Optionally load best iteration
        best_iter_path = args.best_iter_path or args.model_path.replace('.ubj', '_best_iter.json')
        if os.path.exists(best_iter_path):
             try:
                  with open(best_iter_path, 'r') as f: best_iter_info = json.load(f)
                  best_iter = best_iter_info.get('best_iteration') # Get 1-based index
                  if best_iter is not None: logger.info(f"Using best_iteration={best_iter} for prediction.")
             except Exception as e_iter: logger.warning(f"Could not read best iteration from {best_iter_path}: {e_iter}")
        else: logger.info("No best iteration file specified or found. Predicting with all trees.")

    except Exception as e: logger.error(f"Failed loading model: {e}", exc_info=True); return

    # --- Fetch Data ---
    logger.info(f"Fetching race data for {date}/{track}...")
    api_data = fetch_races(date, track)
    if api_data is None: logger.error("Failed to fetch race data from API."); return
    if not api_data: logger.warning("No race data retrieved (empty list)."); print("No predictions generated."); return

    # --- Process Features (using 19-feature logic) ---
    scaled_features, horse_info_list = process_prediction_data(api_data, scaler, track_map, default_track_id)
    if scaled_features is None or horse_info_list is None: logger.error("Feature processing failed."); return
    if scaled_features.shape[0] == 0: logger.error("No valid entries after feature processing."); return

    # --- Predict (using Booster) ---
    probabilities = predict_xgb(model_booster, scaled_features, best_iteration=best_iter)
    if probabilities is None: logger.error("Prediction failed."); return

    # --- Format and Normalize Results (Keep existing code) ---
    results = []
    if len(probabilities) != len(horse_info_list):
         logger.error(f"Prediction/Info mismatch ({len(probabilities)} vs {len(horse_info_list)})."); return
    for i, prob in enumerate(probabilities):
        info = horse_info_list[i]
        results.append({ 'date': info['date'], 'track': info['track'], 'start': info['start'],
                         'program_number': info['program_number'], 'horse': info['horse'],
                         'driver': info['driver'], 'lane': info['lane'],
                         'raw_win_probability': float(prob) })

    logger.info("Normalizing probabilities..."); start_groups = {}
    for result in results:
        key = (result['date'], result['track'], result['start'])
        if key not in start_groups: start_groups[key] = []
        start_groups[key].append(result)
    final_predictions = []
    for key, group in start_groups.items():
        total_prob = sum(r['raw_win_probability'] for r in group)
        if total_prob > 1e-6:
            for result in group: result['normalized_win_probability'] = result['raw_win_probability'] / total_prob; final_predictions.append(result)
        else:
            logger.warning(f"Race {key}: Sum raw prob near zero. Using uniform."); num_horses = len(group)
            uniform_prob = 1.0 / num_horses if num_horses > 0 else 0
            for result in group: result['normalized_win_probability'] = uniform_prob; final_predictions.append(result)
    final_predictions.sort(key=lambda x: (x['date'], x['track'], int(x.get('start', 0)), -x['normalized_win_probability']))

    # --- Output Results ---
    logger.info("Predictions generated and normalized.")
    predictions_json = json.dumps(final_predictions, indent=2)
    print("\n--- Full Predictions (JSON) ---"); print(predictions_json); print("--- End Full Predictions ---")
    if args.output_file:
        try:
            output_path = Path(args.output_file); output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f: f.write(predictions_json)
            logger.info(f"Predictions saved to {output_path}")
        except Exception as e: logger.error(f"Failed writing to {args.output_file}: {e}", exc_info=True)
    if args.top_n_table > 0:
        table_output = create_top_n_table(final_predictions, n=args.top_n_table)
        print(table_output)

if __name__ == "__main__":
    main()

# --- END OF FILE xg_predict_19feat.py ---
