# --- START OF FILE predict_8feat_xgb.py ---

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
import pandas as pd # Added Pandas

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Predict horse race outcomes with an 8-feature XGBoost model")
parser.add_argument('date', type=str, help='Date in YYYY-MM-DD format')
parser.add_argument('--track', type=str, default=None, help='Track code (e.g., "H"); if not provided, fetched from API')
parser.add_argument('--model-path', type=str, default='./horse-race-predictor/racedata/xgb/horse_race_predictor_8feat.ubj',
                    help='Path to trained XGBoost model file')
parser.add_argument('--scaler-path', type=str, default='./horse-race-predictor/racedata/xgb/scaler_8feat.pkl',
                    help='Path to scaler pickle file')
parser.add_argument('--track-map-path', type=str, default='./horse-race-predictor/racedata/xgb/scaler_8feat_track_map.json',
                    help='Path to track map JSON file')
parser.add_argument('--output-file', type=str, default=None, help='Path to save predictions as JSON (optional)')
parser.add_argument('--debug', type=int, default=20, choices=[10, 20, 30, 40, 50],
                    help='Debug level: 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL (default: 20)')
# New argument for the table
parser.add_argument('--top-n-table', type=int, default=5, metavar='N',
                    help='Print a table showing the top N horses per race (default: 5). Set to 0 to disable.')

args = parser.parse_args()

# --- Setup logging, Constants, API Fetching (Keep existing code) ---
logging.basicConfig(level=args.debug, format='%(levelname)s:%(name)s:%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
BASE_URL = "https://heppa.hippos.fi/heppa2_backend"
HEADERS = {'Content-Type': 'application/json'}
CACHE_DIR = Path("./cache_predict_xgb")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- fetch_api_data function (Keep existing code) ---
def fetch_api_data(endpoint, base_url=BASE_URL, headers=HEADERS, cache_dir=CACHE_DIR, retries=2, delay=5):
    # ... (function code from previous version) ...
    safe_endpoint = "".join(c if c.isalnum() else "_" for c in endpoint)
    cache_key = f"api_{safe_endpoint}"
    if len(cache_key) > 150:
         import hashlib
         cache_key = f"api_{hashlib.md5(endpoint.encode()).hexdigest()}"
    cache_file = Path(cache_dir) / f"{cache_key}.json"
    if cache_file.exists():
        try:
            logger.debug(f"Loading cached data for {endpoint} from {cache_file}")
            with open(cache_file, 'r') as f: return json.load(f)
        except json.JSONDecodeError:
             logger.warning(f"Corrupt cache file: {cache_file}. Re-fetching."); cache_file.unlink()
        except Exception as e: logger.error(f"Error reading cache {cache_file}: {e}. Re-fetching.")
    url = f"{base_url}{endpoint}"
    for attempt in range(retries + 1):
        try:
            logger.debug(f"Fetching (attempt {attempt+1}/{retries+1}): {url}")
            response = requests.get(url, headers=headers, timeout=20)
            if '<html' in response.text[:100].lower():
                 logger.error(f"HTML response from {url}. Status: {response.status_code}. Body: {response.text[:200]}...")
                 if attempt < retries: time.sleep(delay); continue
                 else: return None
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Success: {endpoint}. Status: {response.status_code}")
            try:
                with open(cache_file, 'w') as f: json.dump(data, f)
                logger.debug(f"Cached data to {cache_file}")
            except Exception as e: logger.error(f"Failed to write cache {cache_file}: {e}")
            return data
        except requests.exceptions.Timeout:
             logger.warning(f"Timeout fetching {url} on attempt {attempt+1}")
             if attempt < retries: time.sleep(delay)
             else: logger.error(f"Timeout after {retries+1} attempts: {url}"); return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url} on attempt {attempt+1}: {e}")
            if attempt < retries: time.sleep(delay)
            else: logger.error(f"Failed fetching {url} after {retries+1} attempts."); return None
        except json.JSONDecodeError as e:
             logger.error(f"JSON Decode Error from {url}. Status: {response.status_code}. Error: {e}. Response: {response.text[:500]}...")
             return None
    return None

# --- fetch_races function (Keep existing code) ---
def fetch_races(date, track):
    # ... (function code from previous version) ...
    enriched_races = []
    logger.info(f"Fetching starts for {date}/{track} using /races endpoint...")
    starts_info = fetch_api_data(f"/race/{date}/{track}/races")
    if not starts_info or not isinstance(starts_info, list):
        logger.warning(f"No valid starts list found for {date}/{track} via /races")
        return enriched_races
    for start_summary in tqdm(starts_info, desc=f"Fetching Horses for {track}"):
        start_number = start_summary.get('race', {}).get('startNumber', '')
        if not start_number: continue
        logger.debug(f"Fetching horses for Start {start_number}")
        horses_in_start = fetch_api_data(f"/race/{date}/{track}/start/{start_number}")
        if not horses_in_start or not isinstance(horses_in_start, list):
            logger.warning(f"No horses found for {date}/{track}/start/{start_number}"); continue
        for horse_data in horses_in_start:
            if not isinstance(horse_data, dict): continue
            horse_id = horse_data.get('horseId', ''); driver_id = horse_data.get('driverId', '')
            horse_stats = fetch_api_data(f"/horse/{horse_id}/stats") if horse_id else {}
            driver_stats = fetch_api_data(f"/driver/{driver_id}/stats") if driver_id else {}
            enriched_races.append({
                'raceDate': date, 'trackCode': track, 'startNumber': start_number,
                'horse_data': horse_data, 'horse_stats': horse_stats or {},
                'driver_stats': driver_stats or {}
            })
            # time.sleep(0.02) # Optional delay
    logger.info(f"Fetched and enriched data for {len(enriched_races)} horse entries.")
    return enriched_races


# --- process_prediction_data function (Keep existing code) ---
def process_prediction_data(enriched_api_data, scaler, track_map):
    # ... (function code from previous version) ...
    raw_features = []
    horse_info_list = []
    default_track_id = len(track_map)
    unknown_track_logged = set()
    logger.info(f"Processing {len(enriched_api_data)} fetched horse entries into features...")
    for entry in tqdm(enriched_api_data, desc="Processing Features"):
        horse_data = entry.get('horse_data', {})
        horse_stats = entry.get('horse_stats', {}).get('total', {})
        driver_stats = entry.get('driver_stats', {}).get('allTotal', {})
        track_code = entry.get('trackCode', 'Unknown')
        if track_code in track_map: track_id = track_map[track_code]
        else:
            track_id = default_track_id
            if track_code not in unknown_track_logged:
                logger.warning(f"Track '{track_code}' not in map. Using default ID: {track_id}.")
                unknown_track_logged.add(track_code)
        try:
            feature_vector = [
                float(horse_stats.get('winningPercent', 0.0) or 0.0),
                float(horse_stats.get('priceMoney', 0.0) or 0.0),
                float(horse_stats.get('starts', 0.0) or 0.0),
                float(driver_stats.get('winPercentage', 0.0) or 0.0),
                float(driver_stats.get('priceMoney', 0.0) or 0.0),
                float(driver_stats.get('starts', 0.0) or 0.0),
                float(horse_data.get('lane', 0.0) or 0.0),
                float(track_id)
            ]
            raw_features.append(feature_vector)
            horse_info_list.append({
                'date': entry.get('raceDate', 'Unknown'), 'track': track_code,
                'start': entry.get('startNumber', 'Unknown'),
                'program_number': horse_data.get('programNumber', 'N/A'),
                'horse': horse_data.get('horseName', 'Unknown'),
                'driver': horse_data.get('driverName', 'Unknown'),
                'lane': horse_data.get('lane', 'N/A')
            })
        except (ValueError, TypeError) as e:
            logger.warning(f"Feature conversion error for {horse_data.get('horseName', '?')}: {e}. Skipping.")
            continue
    if not raw_features: logger.error("No valid features extracted."); return None, None
    features_np = np.array(raw_features, dtype=np.float32)
    logger.info(f"Extracted raw features shape: {features_np.shape}")
    try:
        scaled_features = scaler.transform(features_np)
        logger.info("Applied scaler transform."); return scaled_features, horse_info_list
    except Exception as e: logger.error(f"Scaler transform error: {e}", exc_info=True); return None, None

# --- predict_xgb function (Keep existing code) ---
def predict_xgb(model, features):
    # ... (function code from previous version) ...
    if features is None or features.shape[0] == 0: return None
    try:
        logger.info(f"Predicting probabilities for {features.shape[0]} entries...")
        probabilities = model.predict_proba(features)[:, 1]
        logger.info("Prediction complete."); return probabilities
    except Exception as e: logger.error(f"XGBoost prediction error: {e}", exc_info=True); return None

# --- get_track_code function (Keep existing code) ---
def get_track_code(date):
    # ... (function code from previous version) ...
    endpoint = f"/race/search/{date}/{date}/"
    data = fetch_api_data(endpoint)
    if data and isinstance(data, list) and len(data) > 0 and data[0].get('events'):
        tracks = [event['trackCode'] for event in data[0]['events'] if isinstance(event, dict)]
        if tracks: logger.debug(f"Found tracks for {date}: {tracks}"); return tracks
    logger.error(f"No tracks found for {date} via API."); return None

# --- create_top_n_table function (NEW) ---
def create_top_n_table(predictions_list, n=5):
    """Creates a formatted string table of the top N horses per race using Pandas."""
    if not predictions_list:
        return "No predictions available to generate table."
    if n <= 0:
        return "" # Return empty string if table is disabled

    try:
        df = pd.DataFrame(predictions_list)
        if df.empty:
             return "No data in predictions list."

        # Ensure 'start' is treated numerically for sorting if possible, handle N/A
        df['start_num'] = pd.to_numeric(df['start'], errors='coerce').fillna(0).astype(int)


        all_top_horses_dfs = []
        # Group by actual race identifiers
        for name, group in df.groupby(['date', 'track', 'start_num']):
            top_n = group.sort_values('normalized_win_probability', ascending=False).head(n)

            # Format probability and select columns
            top_n_formatted = top_n.copy() # Avoid SettingWithCopyWarning
            top_n_formatted['Prob (%)'] = (top_n_formatted['normalized_win_probability'] * 100).map('{:.2f}%'.format)
            top_n_formatted = top_n_formatted[['start', 'program_number', 'horse', 'driver', 'Prob (%)']]
            top_n_formatted.rename(columns={'program_number': 'Pgm', 'horse': 'Horse', 'driver': 'Driver'}, inplace=True)

            all_top_horses_dfs.append(top_n_formatted)

        if not all_top_horses_dfs:
            return "No groups found for table generation."

        # Combine all top N results
        top_horses_df = pd.concat(all_top_horses_dfs)

        # Create the string representation
        # Adjust max_rows if you expect many races
        pd.set_option('display.max_rows', 200)
        # pd.set_option('display.max_colwidth', None) # Ensure long names aren't truncated
        table_string = top_horses_df.to_string(index=False, justify='left')
        pd.reset_option('display.max_rows')
        # pd.reset_option('display.max_colwidth')

        header = f"\n--- Top {n} Horses Per Race ---\n"
        footer = "\n--- End Top N Table ---\n"
        return header + table_string + footer

    except Exception as e:
        logger.error(f"Error creating top N table: {e}", exc_info=True)
        return "Error generating table."


# --- Main Execution Logic (Modified) ---
def main():
    date = args.date
    try: datetime.strptime(date, "%Y-%m-%d")
    except ValueError: logger.error("Invalid date format. Use YYYY-MM-DD."); return

    track = args.track
    if not track:
        logger.info(f"No track specified, trying to find tracks for {date}...")
        available_tracks = get_track_code(date)
        if available_tracks:
            track = available_tracks[0]
            logger.info(f"Using first track found: {track}")
        else:
            logger.error(f"Cannot determine track for {date}. Please specify --track."); return

    # --- Load Scaler, Track Map, Model (Keep existing code) ---
    scaler = track_map = model = None
    try: # Scaler
        if not os.path.exists(args.scaler_path): logger.error(f"Scaler not found: {args.scaler_path}"); return
        with open(args.scaler_path, 'rb') as f: scaler = pickle.load(f)
        logger.info(f"Loaded scaler: {args.scaler_path}")
    except Exception as e: logger.error(f"Failed loading scaler: {e}", exc_info=True); return
    try: # Track Map
        if not os.path.exists(args.track_map_path): logger.error(f"Track map not found: {args.track_map_path}"); return
        with open(args.track_map_path, 'r') as f: track_map = json.load(f)
        logger.info(f"Loaded track map: {args.track_map_path} ({len(track_map)} tracks)")
    except Exception as e: logger.error(f"Failed loading track map: {e}", exc_info=True); return
    try: # Model
        if not os.path.exists(args.model_path): logger.error(f"Model not found: {args.model_path}"); return
        model = xgb.XGBClassifier(); model.load_model(args.model_path)
        logger.info(f"Loaded XGBoost model: {args.model_path}")
    except Exception as e: logger.error(f"Failed loading model: {e}", exc_info=True); return

    # --- Fetch Data (Keep existing code) ---
    logger.info(f"Fetching race data for {date}/{track}...")
    api_data = fetch_races(date, track)
    if not api_data: logger.error("No race data retrieved."); return

    # --- Process Features (Keep existing code) ---
    scaled_features, horse_info_list = process_prediction_data(api_data, scaler, track_map)
    if scaled_features is None or horse_info_list is None: logger.error("Feature processing failed."); return
    if scaled_features.shape[0] == 0: logger.error("No valid entries after feature processing."); return

    # --- Predict (Keep existing code) ---
    probabilities = predict_xgb(model, scaled_features)
    if probabilities is None: logger.error("Prediction failed."); return

    # --- Format and Normalize Results (Keep existing code) ---
    results = []
    if len(probabilities) != len(horse_info_list):
         logger.error(f"Prediction/Info mismatch ({len(probabilities)} vs {len(horse_info_list)})."); return
    for i, prob in enumerate(probabilities):
        info = horse_info_list[i]
        results.append({
            'date': info['date'], 'track': info['track'], 'start': info['start'],
            'program_number': info['program_number'], 'horse': info['horse'],
            'driver': info['driver'], 'lane': info['lane'],
            'raw_win_probability': float(prob)
        })

    logger.info("Normalizing probabilities within each race start...")
    start_groups = {}
    for result in results:
        key = (result['date'], result['track'], result['start'])
        if key not in start_groups: start_groups[key] = []
        start_groups[key].append(result)

    final_predictions = []
    for key, group in start_groups.items():
        total_prob = sum(r['raw_win_probability'] for r in group)
        logger.debug(f"Race {key}: Total raw probability sum = {total_prob}")
        if total_prob > 1e-6:
            for result in group:
                result['normalized_win_probability'] = result['raw_win_probability'] / total_prob
                final_predictions.append(result)
        else:
            logger.warning(f"Race {key}: Sum raw prob near zero. Using uniform prob.")
            num_horses = len(group)
            uniform_prob = 1.0 / num_horses if num_horses > 0 else 0
            for result in group:
                result['normalized_win_probability'] = uniform_prob
                final_predictions.append(result)

    # Sort final list (optional, based on previous script)
    final_predictions.sort(key=lambda x: (x['date'], x['track'], int(x.get('start', 0)), -x['normalized_win_probability']))

    # --- Output Results (Modified) ---
    logger.info("Predictions generated and normalized.")

    # Always print JSON (or control with another flag if needed)
    predictions_json = json.dumps(final_predictions, indent=2)
    print("\n--- Full Predictions (JSON) ---")
    print(predictions_json)
    print("--- End Full Predictions ---")

    # Save JSON to file if requested
    if args.output_file:
        try:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f: f.write(predictions_json)
            logger.info(f"Predictions saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write predictions to {args.output_file}: {e}", exc_info=True)

    # Print Top N Table if requested
    if args.top_n_table > 0:
        table_output = create_top_n_table(final_predictions, n=args.top_n_table)
        print(table_output) # Print the formatted table string


if __name__ == "__main__":
    main()

# --- END OF FILE predict_8feat_xgb.py ---
