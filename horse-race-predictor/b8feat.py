import json
import os
import pickle
import argparse
from datetime import datetime
from pathlib import Path
import time
import logging

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import requests
from tqdm import tqdm
import pandas as pd
from flask import Flask, request, jsonify, abort

# --- Argument Parsing for Artifact Paths ---
parser = argparse.ArgumentParser(description="Flask API for XGBoost Horse Race Predictions")
parser.add_argument('--model-dir', type=str, required=True,
                    help='Directory containing best_model.ubj, best_scaler.pkl, and best_track_map.json')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the API server to')
parser.add_argument('--port', type=int, default=5000, help='Port to run the API server on')
parser.add_argument('--debug-mode', action='store_true', help='Run Flask in debug mode (DO NOT use in production)')
cli_args = parser.parse_args()

# --- Basic Configuration ---
MODEL_DIR = Path(cli_args.model_dir)
MODEL_PATH = MODEL_DIR / "best_model.ubj"
SCALER_PATH = MODEL_DIR / "best_scaler.pkl"
TRACK_MAP_PATH = MODEL_DIR / "best_track_map.json"

# --- Flask App Setup ---
app = Flask(__name__)

# --- Logging Setup ---
# Use Flask's logger after app is created
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
gunicorn_logger = logging.getLogger('gunicorn.error') # Integrate with gunicorn if used
app.logger.handlers.extend(gunicorn_logger.handlers)
app.logger.setLevel(logging.INFO if not cli_args.debug_mode else logging.DEBUG)


# --- Global Variables for Loaded Artifacts ---
# Loaded once when the application starts
xgb_model = None
scaler = None
track_map = None
default_track_id = 0 # Will be set after loading track_map

# --- Load Artifacts Function ---
def load_artifacts():
    global xgb_model, scaler, track_map, default_track_id
    app.logger.info(f"Loading artifacts from directory: {MODEL_DIR}")

    # Load Scaler
    try:
        if not SCALER_PATH.exists(): raise FileNotFoundError("Scaler file not found")
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        app.logger.info(f"Scaler loaded successfully from {SCALER_PATH}")
    except Exception as e:
        app.logger.error(f"FATAL: Failed to load scaler: {e}", exc_info=True)
        raise RuntimeError(f"Could not load scaler from {SCALER_PATH}") from e

    # Load Track Map
    try:
        if not TRACK_MAP_PATH.exists(): raise FileNotFoundError("Track map file not found")
        with open(TRACK_MAP_PATH, 'r') as f: track_map = json.load(f)
        default_track_id = len(track_map) # Assign default ID for unknown tracks
        app.logger.info(f"Track map loaded successfully from {TRACK_MAP_PATH} ({len(track_map)} tracks)")
    except Exception as e:
        app.logger.error(f"FATAL: Failed to load track map: {e}", exc_info=True)
        raise RuntimeError(f"Could not load track map from {TRACK_MAP_PATH}") from e

    # Load XGBoost Model (Booster)
    try:
        if not MODEL_PATH.exists(): raise FileNotFoundError("XGBoost model file not found")
        # Load the booster directly
        xgb_model = xgb.Booster()
        xgb_model.load_model(MODEL_PATH)
        app.logger.info(f"XGBoost model booster loaded successfully from {MODEL_PATH}")
        # Optionally load best iteration if saved separately and needed
        # best_iter_path = MODEL_PATH.parent / "best_params.json" # Assuming path convention
        # if best_iter_path.exists():
        #     with open(best_iter_path, 'r') as f:
        #         params_info = json.load(f)
        #         xgb_model.best_iteration = params_info.get('best_iteration') - 1 # Store 0-based index if needed
        #         app.logger.info(f"Set model best_iteration to {xgb_model.best_iteration + 1}")

    except Exception as e:
        app.logger.error(f"FATAL: Failed to load XGBoost model: {e}", exc_info=True)
        raise RuntimeError(f"Could not load model from {MODEL_PATH}") from e

# --- API Constants & Helper Functions ---
BASE_URL = "https://heppa.hippos.fi/heppa2_backend"
HEADERS = {'Content-Type': 'application/json'}
CACHE_DIR = Path("./cache_predict_api") # Separate cache for API
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Include fetch_api_data, fetch_races, process_prediction_data, predict_xgb, get_track_code, create_top_n_table
# Ensure these functions use app.logger instead of logger

def fetch_api_data(endpoint, base_url=BASE_URL, headers=HEADERS, cache_dir=CACHE_DIR, retries=2, delay=5):
    """Fetch data from API with caching and retries."""
    safe_endpoint = "".join(c if c.isalnum() else "_" for c in endpoint)
    cache_key = f"api_{safe_endpoint}"
    if len(cache_key) > 150: import hashlib; cache_key = f"api_{hashlib.md5(endpoint.encode()).hexdigest()}"
    cache_file = Path(cache_dir) / f"{cache_key}.json"

    if cache_file.exists():
        try:
            app.logger.debug(f"Cache HIT for {endpoint} from {cache_file}")
            with open(cache_file, 'r') as f: return json.load(f)
        except Exception as e: app.logger.warning(f"Cache read error {cache_file}: {e}. Re-fetching."); cache_file.unlink()

    url = f"{base_url}{endpoint}"
    for attempt in range(retries + 1):
        try:
            app.logger.debug(f"Fetching (attempt {attempt+1}): {url}")
            response = requests.get(url, headers=headers, timeout=15) # Slightly shorter timeout for API context
            if '<html' in response.text[:100].lower():
                 app.logger.error(f"HTML response from {url}. Status: {response.status_code}.");
                 if attempt < retries: 
                     time.sleep(delay) 
                     continue
                 else: 
                     return None
            response.raise_for_status()
            data = response.json()
            app.logger.debug(f"Fetch SUCCESS: {endpoint}. Status: {response.status_code}")
            try:
                with open(cache_file, 'w') as f: json.dump(data, f)
                app.logger.debug(f"Cached data to {cache_file}")
            except Exception as e: app.logger.error(f"Cache write error {cache_file}: {e}")
            return data
        # Specific error handling for API context
        except requests.exceptions.Timeout:
             app.logger.warning(f"Timeout fetching {url} on attempt {attempt+1}")
             if attempt == retries: app.logger.error(f"Final timeout fetching {url}."); return None
        except requests.exceptions.HTTPError as e:
             app.logger.error(f"HTTP Error fetching {url}: {e.response.status_code} {e.response.reason}")
             return None # Don't retry client/server errors usually
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Request Exception fetching {url}: {e}")
            if attempt == retries: return None
        except json.JSONDecodeError as e:
             app.logger.error(f"JSON Decode Error from {url}. Status: {response.status_code}. Error: {e}.")
             return None # Bad response, don't retry
        # Wait before retry
        if attempt < retries:
             app.logger.info(f"Waiting {delay}s before retrying {url}...")
             time.sleep(delay)
    return None

def fetch_races(date, track):
    """Fetches and enriches race data for a given date and track from the API."""
    enriched_races = []
    app.logger.info(f"Fetching starts for {date}/{track} using /races endpoint...")
    starts_info = fetch_api_data(f"/race/{date}/{track}/races")
    if not starts_info or not isinstance(starts_info, list):
        app.logger.warning(f"No valid starts list found for {date}/{track} via /races")
        return None # Return None to indicate failure

    # Iterate through each race (start number) of the day
    for start_summary in starts_info: # No TQDM in API
        start_number = start_summary.get('race', {}).get('startNumber', '')
        if not start_number: continue
        app.logger.debug(f"Fetching horses for Start {start_number}")
        horses_in_start = fetch_api_data(f"/race/{date}/{track}/start/{start_number}")
        if not horses_in_start or not isinstance(horses_in_start, list):
            app.logger.warning(f"No horses found for {date}/{track}/start/{start_number}"); continue
        for horse_data in horses_in_start:
            if not isinstance(horse_data, dict): continue
            horse_id = horse_data.get('horseId', ''); driver_id = horse_data.get('driverId', '')
            horse_stats = fetch_api_data(f"/horse/{horse_id}/stats") if horse_id else {}
            driver_stats = fetch_api_data(f"/driver/{driver_id}/stats") if driver_id else {}
            enriched_races.append({ 'raceDate': date, 'trackCode': track, 'startNumber': start_number,
                                   'horse_data': horse_data, 'horse_stats': horse_stats or {},
                                   'driver_stats': driver_stats or {} })
    app.logger.info(f"Fetched {len(enriched_races)} horse entries for {date}/{track}.")
    return enriched_races

def process_prediction_data(enriched_api_data, loaded_scaler, loaded_track_map, loaded_default_track_id):
    """Processes API data into scaled features using loaded artifacts."""
    raw_features = []
    horse_info_list = []
    unknown_track_logged = set()
    app.logger.info(f"Processing {len(enriched_api_data)} fetched entries into features...")
    for entry in enriched_api_data: # No TQDM
        horse_data = entry.get('horse_data', {})
        horse_stats = entry.get('horse_stats', {}).get('total', {})
        driver_stats = entry.get('driver_stats', {}).get('allTotal', {})
        track_code = entry.get('trackCode', 'Unknown')
        if track_code in loaded_track_map: track_id = loaded_track_map[track_code]
        else:
            track_id = loaded_default_track_id
            if track_code not in unknown_track_logged:
                app.logger.warning(f"Track '{track_code}' not in map. Using default ID: {track_id}.")
                unknown_track_logged.add(track_code)
        try:
            feature_vector = [ float(horse_stats.get('winningPercent', 0.0) or 0.0), float(horse_stats.get('priceMoney', 0.0) or 0.0),
                               float(horse_stats.get('starts', 0.0) or 0.0), float(driver_stats.get('winPercentage', 0.0) or 0.0),
                               float(driver_stats.get('priceMoney', 0.0) or 0.0), float(driver_stats.get('starts', 0.0) or 0.0),
                               float(horse_data.get('lane', 0.0) or 0.0), float(track_id) ]
            raw_features.append(feature_vector)
            horse_info_list.append({ 'date': entry.get('raceDate', 'N/A'), 'track': track_code, 'start': entry.get('startNumber', 'N/A'),
                                     'program_number': horse_data.get('programNumber', 'N/A'), 'horse': horse_data.get('horseName', 'N/A'),
                                     'driver': horse_data.get('driverName', 'N/A'), 'lane': horse_data.get('lane', 'N/A') })
        except (ValueError, TypeError) as e:
            app.logger.warning(f"Feature conversion error: {e}. Skipping entry: {horse_data.get('horseName')}")
            continue
    if not raw_features: app.logger.error("No valid features extracted."); return None, None
    features_np = np.array(raw_features, dtype=np.float32)
    app.logger.info(f"Extracted features shape: {features_np.shape}")
    try:
        scaled_features = loaded_scaler.transform(features_np)
        app.logger.info("Applied scaler transform."); return scaled_features, horse_info_list
    except Exception as e: app.logger.error(f"Scaler transform error: {e}", exc_info=True); return None, None

def predict_xgb(booster_model, features):
    """Generates predictions using the loaded XGBoost Booster."""
    if features is None or features.shape[0] == 0: return None
    try:
        app.logger.info(f"Predicting probabilities for {features.shape[0]} entries...")
        # Use Booster's predict method. Need DMatrix.
        # Output is probability of the positive class (class 1).
        dmatrix = xgb.DMatrix(features)
        # Use best_iteration if available and set during load, otherwise predict using all trees
        iteration_args = {}
        # if hasattr(booster_model, 'best_iteration') and booster_model.best_iteration is not None:
        #     iteration_args['iteration_range'] = (0, booster_model.best_iteration + 1)
        #     app.logger.debug(f"Predicting using iteration range: {iteration_args['iteration_range']}")

        probabilities = booster_model.predict(dmatrix, **iteration_args)
        app.logger.info("Prediction complete."); return probabilities
    except Exception as e: app.logger.error(f"XGBoost prediction error: {e}", exc_info=True); return None

def get_track_code(date):
    """Fetches track codes for a given date from the API."""
    endpoint = f"/race/search/{date}/{date}/"
    data = fetch_api_data(endpoint)
    if data and isinstance(data, list) and len(data) > 0 and data[0].get('events'):
        tracks = [event['trackCode'] for event in data[0]['events'] if isinstance(event, dict)]
        if tracks: app.logger.debug(f"Found tracks for {date}: {tracks}"); return tracks
    app.logger.error(f"No tracks found for {date} via API."); return None

def create_top_n_table(predictions_list, n=5):
    """Creates a formatted string table of the top N horses per race using Pandas."""
    if not predictions_list or n <= 0: return ""
    try:
        df = pd.DataFrame(predictions_list)
        if df.empty: return "No data for table."
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
        return "\n".join(all_tables) + "\n--- End Top N Table ---"
    except Exception as e: app.logger.error(f"Error creating top N table: {e}"); return "Error generating table."

# --- Load Artifacts Before First Request ---
try:
    load_artifacts()
except RuntimeError as e:
     app.logger.critical(f"Application startup failed: Could not load critical artifacts. {e}")
     # Depending on deployment, might need to exit or handle differently
     exit(1)


# --- API Endpoints ---
@app.route('/')
def index():
    return jsonify({"message": "XGBoost Horse Race Prediction API", "status": "OK"})

@app.route('/predict/<string:date>', methods=['GET'])
def get_predictions(date):
    app.logger.info(f"Received prediction request for date: {date}")

    # Validate Date Format
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        app.logger.error(f"Invalid date format received: {date}")
        abort(400, description="Invalid date format. Please use YYYY-MM-DD.")

    # Get Track (optional query parameter)
    track = request.args.get('track', default=None, type=str)
    if not track:
        app.logger.info(f"Track not specified, attempting API lookup for date: {date}")
        available_tracks = get_track_code(date)
        if available_tracks:
            track = available_tracks[0] # Default to first track
            app.logger.info(f"Using auto-detected track: {track}")
        else:
            app.logger.error(f"Could not find track for date: {date}")
            abort(404, description=f"No track specified and could not find tracks for date {date}.")

    app.logger.info(f"Processing request for Date: {date}, Track: {track}")

    # 1. Fetch Data
    api_data = fetch_races(date, track)
    if api_data is None:
        app.logger.error(f"Failed to fetch race data from external API for {date}/{track}")
        abort(503, description="Failed to fetch data from upstream API.")
    if not api_data:
        app.logger.warning(f"No race data found for {date}/{track}")
        return jsonify({"message": f"No race data found for {date}/{track}", "predictions": []}) # Return empty list

    # 2. Process Features
    # Pass the globally loaded artifacts
    scaled_features, horse_info_list = process_prediction_data(api_data, scaler, track_map, default_track_id)
    if scaled_features is None or scaled_features.shape[0] == 0:
        app.logger.error("Feature processing failed or yielded no features.")
        abort(500, description="Internal error during feature processing.")

    # 3. Predict
    # Pass the globally loaded booster model
    probabilities = predict_xgb(xgb_model, scaled_features)
    if probabilities is None:
        app.logger.error("Model prediction failed.")
        abort(500, description="Internal error during model prediction.")

    # 4. Format and Normalize Results
    results = []
    if len(probabilities) != len(horse_info_list):
         app.logger.error(f"Mismatch pred/info count ({len(probabilities)} vs {len(horse_info_list)}).")
         abort(500, description="Internal error: Prediction count mismatch.")

    for i, prob in enumerate(probabilities):
        info = horse_info_list[i]
        results.append({ 'date': info['date'], 'track': info['track'], 'start': info['start'],
                         'program_number': info['program_number'], 'horse': info['horse'],
                         'driver': info['driver'], 'lane': info['lane'],
                         'raw_win_probability': float(prob) }) # Store raw prob

    # Normalize per start
    app.logger.info("Normalizing probabilities...")
    start_groups = {}
    for result in results:
        key = (result['date'], result['track'], result['start'])
        if key not in start_groups: start_groups[key] = []
        start_groups[key].append(result)

    final_predictions = []
    for key, group in start_groups.items():
        total_prob = sum(r['raw_win_probability'] for r in group)
        if total_prob > 1e-6:
            for result in group:
                result['normalized_win_probability'] = result['raw_win_probability'] / total_prob
                final_predictions.append(result)
        else:
            app.logger.warning(f"Race {key}: Sum raw prob near zero ({total_prob}). Using uniform.")
            num_horses = len(group)
            uniform_prob = 1.0 / num_horses if num_horses > 0 else 0
            for result in group:
                result['normalized_win_probability'] = uniform_prob
                final_predictions.append(result)

    # Sort results (optional)
    final_predictions.sort(key=lambda x: (x['date'], x['track'], int(x.get('start', 0)), -x['normalized_win_probability']))

    # 5. Prepare Response
    response_data = {"predictions": final_predictions}

    # Add Top-N Table if requested via query parameter
    top_n = request.args.get('top_n', default=0, type=int)
    if top_n > 0:
        app.logger.info(f"Generating Top-{top_n} table.")
        table_str = create_top_n_table(final_predictions, n=top_n)
        response_data["top_n_table"] = table_str # Add table to JSON response

    app.logger.info(f"Successfully generated {len(final_predictions)} predictions for {date}/{track}.")
    return jsonify(response_data)

# --- Run the App ---
if __name__ == '__main__':
    # This is for development only!
    # Use a production WSGI server like Gunicorn or Waitress in production.
    app.logger.info(f"Starting Flask development server on {cli_args.host}:{cli_args.port}")
    app.run(host=cli_args.host, port=cli_args.port, debug=cli_args.debug_mode)

# --- END OF FILE prediction_api.py ---
