import os
import json
import subprocess
import numpy as np
import logging
import argparse
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Parameters
POS_WEIGHTS = [5.0, 10.0, 20.0, 50.0, 100.0]
LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005, 0.01]
DEFAULT_DATES = ["2023-01-09", "2024-10-07"]
VALIDATION_DIR = "./validation"
CACHE_DIR = "./cache"
BASE_URL = "https://heppa.hippos.fi/heppa2_backend"
HEADERS = {'Content-Type': 'application/json'}

os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_api_data(endpoint, base_url=BASE_URL, headers=HEADERS, cache_dir=CACHE_DIR):
    """Fetch data from API with caching."""
    # Sanitize endpoint for filename (remove slashes, query params for simplicity)
    cache_key = endpoint.replace('/', '_').replace('?', '_').replace('&', '_')
    cache_file = Path(cache_dir) / f"{cache_key}.json"
    
    if cache_file.exists():
        logger.info(f"Loading cached data for {endpoint} from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    url = base_url + endpoint
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Fetched data from {endpoint} - Status: {response.status_code}")
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Cached data to {cache_file}")
        return data
    except requests.RequestException as e:
        logger.error(f"API request failed for {url}: {str(e)}")
        return None

def get_track_code(date):
    endpoint = f"/race/search/{date}/{date}/"
    data = fetch_api_data(endpoint)
    if data and isinstance(data, list) and data[0].get('events'):
        tracks = [event['trackCode'] for event in data[0]['events']]
        logger.debug(f"Found tracks for {date}: {tracks}")
        return tracks
    logger.error(f"No tracks found for {date}")
    return None

def fetch_start_results(start, date, track, endpoints):
    logger.info(f"Processing start number: {start} for {date}/{track}")
    start_results = {}
    for name, endpoint_fn in endpoints.items():
        endpoint = endpoint_fn(start)
        logger.info(f"Fetching {name} (Start {start})")
        data = fetch_api_data(endpoint)
        if data is not None:
            start_results[name] = data
        else:
            logger.error(f"Failed to fetch {name} for Start {start}")
    
    if start_results:
        filename = f"{VALIDATION_DIR}/actual_start_{start}_results_{date}_{track}.json"
        with open(filename, 'w') as f:
            json.dump(start_results, f, indent=2)
        logger.info(f"Saved results for start {start} to {filename}")
        return start_results
    return None

def get_start_numbers(date, track):
    endpoint = f"/race/{date}/{track}/races"
    starts_data = fetch_api_data(endpoint)
    if starts_data and isinstance(starts_data, list):
        start_numbers = [start.get('race', {}).get('startNumber', '') for start in starts_data if start.get('race', {}).get('startNumber')]
        logger.info(f"Found {len(start_numbers)} start numbers for {date}/{track}: {start_numbers}")
        return start_numbers
    logger.error(f"Failed to fetch starts for {date}/{track}")
    return []

def train_model(model_type, pos_weight, learning_rate):
    epochs = 20
    # 100 if model_type == "8feat" else 50
    howmanyfeatures = model_type.split('feat')[0]
    model_dir = f"./models/{model_type}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/horse_race_predictor_{pos_weight}pos_{learning_rate}learn.pth"
    scaler_path = f"{model_dir}/scaler_{pos_weight}pos_{learning_rate}learn.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        logger.info(f"Existing model found for {model_type} with POS_WEIGHT={pos_weight}, LEARNING_RATE={learning_rate}")
        return model_path, scaler_path
    
    cmd = (
        f"./venv/bin/python3 train_{howmanyfeatures}feat.py "
        f"--pos-weight {pos_weight} --learning-rate {learning_rate} --epochs {epochs} "
        f"--model-path {model_path} --scaler-path {scaler_path}"
    )
    logger.info(f"Training {model_type} with command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"{model_type} training completed. Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed for {model_type}: {e.stderr}")
        return None, None
    
    return model_path, scaler_path

def predict_races(model_type, pos_weight, learning_rate, model_path, scaler_path, date, track):
    howmanyfeatures = model_type.split('feat')[0]
    track_map_path = f"./horse-race-predictor/racedata/scaler_track_map.json"
    output_file = f"{VALIDATION_DIR}/model_{model_type}_{pos_weight}pos_{learning_rate}learn_{date}_{track}.json"
    
    cmd = (
        f"./venv/bin/python3 predict_{howmanyfeatures}feat.py {date} "
        f"--track {track} --model-path {model_path} --scaler-path {scaler_path} "
        f"--track-map-path {track_map_path} --output-file {output_file}"
    )
    logger.info(f"Predicting for {model_type} on {date}/{track} with command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Prediction completed for {date}/{track}. Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Prediction failed for {model_type} on {date}/{track}: {e.stderr}")
        return []
    
    with open(output_file, 'r') as f:
        predictions = json.load(f)
    return predictions

def get_top_3_predictions(predictions):
    start_predictions = {}
    for pred in predictions:
        start = pred['start']
        if start not in start_predictions:
            start_predictions[start] = []
        start_predictions[start].append(pred)
    
    top_3_per_start = {}
    for start, preds in start_predictions.items():
        probs = [float(p['win_probability']) for p in preds]
        # Check if probabilities are uniform (within 1e-6 tolerance)
        if max(probs) - min(probs) < 1e-6:  # All probs are essentially equal
            # Sort alphabetically by horse name and take first 3
            sorted_preds = sorted(preds, key=lambda x: x['horse'])[:3]
        else:
            # Sort by probability (descending) and take top 3
            sorted_preds = sorted(preds, key=lambda x: float(x['win_probability']), reverse=True)[:3]
        top_3_per_start[start] = sorted_preds
    
    return top_3_per_start

def evaluate_model(predictions, actual_results):
    # Hit Rate
    top_3_preds = get_top_3_predictions(predictions)
    actual_winners = {}
    for start, data in actual_results.items():
        start_horses = data.get('getStartHorses', [])
        winners = [h['horseName'] for h in start_horses if h.get('placing') == "1"]
        actual_winners[start] = winners
    
    hits = 0
    total = 0
    for start in top_3_preds:
        if start in actual_winners:
            pred_horses = {p['horse'] for p in top_3_preds[start]}
            actual_horses = set(actual_winners[start])
            hits += len(pred_horses & actual_horses)
            total += min(3, len(actual_horses))
    
    hit_rate = hits / total if total > 0 else 0.0

    # Variance Score
    start_predictions = {}
    for pred in predictions:
        start = pred['start']
        if start not in start_predictions:
            start_predictions[start] = []
        start_predictions[start].append(float(pred['win_probability']))

    variance_scores = []
    for start, probs in start_predictions.items():
        if len(probs) > 1:
            variance = np.var(probs)
            #variance_score = 1 / (1 + np.exp(-10 * (variance - 0.01)))
            #variance_score = variance / 0.05 if variance < 0.05 else 1.0
            variance_score = variance / 0.01 if variance < 0.01 else 1.0
            variance_scores.append(variance_score)
        else:
            variance_scores.append(1.0)
    avg_variance_score = np.mean(variance_scores) if variance_scores else 1.0

    # Brier Score
    brier_scores = []
    for pred in predictions:
        start = pred['start']
        horse = pred['horse']
        prob = float(pred['win_probability'])
        if start in actual_winners:
            actual = 1.0 if horse in actual_winners[start] else 0.0
            brier = (prob - actual) ** 2
            brier_scores.append(brier)
    
    avg_brier_score = np.mean(brier_scores) if brier_scores else 0.0
    brier_adjusted = 1 - avg_brier_score

    # Combined Score
    final_score = (hit_rate * 0.3 + avg_variance_score * 0.4 + brier_adjusted * 0.3)
    logger.info(f"Hit Rate: {hit_rate:.4f}, Variance Score: {avg_variance_score:.4f}, Brier Adjusted: {brier_adjusted:.4f}, Final Score: {final_score:.4f}")

    return final_score

def train_all_models():
    logger.info("Starting training phase")
    models = []
    for model_type in ["8feat", "14feat"]:
        for pos_weight in POS_WEIGHTS:
            for learning_rate in LEARNING_RATES:
                model_path, scaler_path = train_model(model_type, pos_weight, learning_rate)
                if model_path and scaler_path:
                    models.append((model_type, pos_weight, learning_rate, model_path, scaler_path))
    return models

def predict_all_dates(models, dates):
    logger.info("Starting prediction phase")
    predictions_dict = {}
    date_track_pairs = []
    for date in dates:
        tracks = get_track_code(date)
        if tracks and tracks[0]:
            track = tracks[0]
            date_track_pairs.append((date, track))
        else:
            logger.warning(f"Skipping predictions for {date} due to no track code")
    
    for model_type, pos_weight, learning_rate, model_path, scaler_path in models:
        model_key = f"{model_type}_{pos_weight}pos_{learning_rate}learn"
        predictions_dict[model_key] = {}
        for date, track in date_track_pairs:
            predictions = predict_races(model_type, pos_weight, learning_rate, model_path, scaler_path, date, track)
            predictions_dict[model_key][date] = predictions
            filename = f"{VALIDATION_DIR}/model_{model_key}_{date}_{track}.json"
            logger.info(f"Saved predictions to {filename}")
            
            top_3 = get_top_3_predictions(predictions)
            logger.info(f"Top 3 predictions for {model_key} on {date}/{track}:")
            for start, preds in top_3.items():
                logger.info(f"Start {start}:")
                for p in preds:
                    logger.info(f"  {p['horse']} - {p['win_probability']}")
    return predictions_dict, date_track_pairs

def evaluate_all_models(predictions_dict, dates, actual_results_file=None):
    logger.info("Starting evaluation phase")
    actual_results = {}
    date_track_pairs = []
    
    if actual_results_file:
        with open(actual_results_file, 'r') as f:
            actual_results = json.load(f)
    else:
        for date in dates:
            tracks = get_track_code(date)
            if tracks and tracks[0]:
                track = tracks[0]
                date_track_pairs.append((date, track))
                start_numbers = get_start_numbers(date, track)
                if not start_numbers:
                    logger.warning(f"No starts found for {date}/{track}")
                    continue
                endpoints = {
                    "getTotoResults": lambda start: f"/race/{date}/{track}/totoresults?startNumber={start}",
                    "getStartHorses": lambda start: f"/race/{date}/{track}/start/{start}",
                    "getInterMediateTimes": lambda start: f"/race/{date}/{track}/intermediatetimes?startNumber={start}",
                    "getRaceStatus": lambda start: f"/race/{date}/{track}/{start}/status"
                }
                actual_results[date] = {}
                for start in start_numbers:
                    results = fetch_start_results(start, date, track, endpoints)
                    if results:
                        actual_results[date][start] = results
    
    scores = {}
    for model_key, predictions_by_date in predictions_dict.items():
        scores[model_key] = []
        for date in dates:
            if date in predictions_by_date:
                track = next((t for d, t in date_track_pairs if d == date), None)
                if track:
                    predictions = predictions_by_date[date]
                    score = evaluate_model(predictions, actual_results.get(date, {}))
                    scores[model_key].append(score)
                    logger.info(f"{model_key} - {date}/{track}: Combined Score = {score:.4f}")
        
        avg_score = np.mean(scores[model_key]) if scores[model_key] else 0.0
        scores[model_key] = avg_score
        logger.info(f"{model_key} - Average Combined Score = {avg_score:.4f}")
    
    best_model = max(scores.items(), key=lambda x: x[1]) if scores else ("None", 0.0)
    logger.info(f"\nBest Model: {best_model[0]} with Average Combined Score = {best_model[1]:.4f}")
    return scores, best_model

def main():
    parser = argparse.ArgumentParser(description="Horse race prediction model trainer, predictor, and evaluator.")
    parser.add_argument('--train', action='store_true', help='Train models with various parameters.')
    parser.add_argument('--predict', action='store_true', help='Generate predictions for specified dates.')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate predictions and select best model.')
    parser.add_argument('--all', action='store_true', help='Run full workflow (train, predict, evaluate).')
    parser.add_argument('--dates', nargs='+', default=DEFAULT_DATES, help='List of dates.')
    parser.add_argument('--actual-results-file', type=str, help='JSON file with actual results.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    #parser.add_argument('--debug', type=int, default=20, choices=[0, 10, 20, 30, 40], help='Debug level: 0=silent, 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR (default: 20)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='%(levelname)s: %(message)s')

#    if args.debug == 0:
#        logging.disable(logging.CRITICAL)
#    else:
#        logging.basicConfig(level=args.debug, format='%(levelname)s: %(message)s')
   

    if not (args.train or args.predict or args.evaluate):
        args.all = True

    dates = args.dates
    logger.info(f"Running with dates: {dates}")

    if args.all or args.train:
        models = train_all_models()
    else:
        models = []
        for model_type in ["8feat", "14feat"]:
            for pos_weight in POS_WEIGHTS:
                for learning_rate in LEARNING_RATES:
                    model_dir = f"./models/{model_type}"
                    model_path = f"{model_dir}/horse_race_predictor_{pos_weight}pos_{learning_rate}learn.pth"
                    scaler_path = f"{model_dir}/scaler_{pos_weight}pos_{learning_rate}learn.pkl"
                    if os.path.exists(model_path) and os.path.exists(scaler_path):
                        models.append((model_type, pos_weight, learning_rate, model_path, scaler_path))
        if not models:
            logger.error("No existing models found. Please run with --train first.")
            return

    if args.all or args.predict:
        predictions_dict, date_track_pairs = predict_all_dates(models, dates)
    else:
        predictions_dict, date_track_pairs = {}, []

    if args.all or args.evaluate:
        if not predictions_dict and (args.all or args.predict):
            predictions_dict, date_track_pairs = predict_all_dates(models, dates)
        elif not predictions_dict:
            predictions_dict = {}
            for model_type, pos_weight, learning_rate, _, _ in models:
                model_key = f"{model_type}_{pos_weight}pos_{learning_rate}learn"
                predictions_dict[model_key] = {}
                for date in dates:
                    tracks = get_track_code(date)
                    if tracks and tracks[0]:
                        track = tracks[0]
                        filename = f"{VALIDATION_DIR}/model_{model_key}_{date}_{track}.json"
                        if os.path.exists(filename):
                            with open(filename, 'r') as f:
                                predictions_dict[model_key][date] = json.load(f)
        if predictions_dict:
            scores, best_model = evaluate_all_models(predictions_dict, dates, args.actual_results_file)
        else:
            logger.error("No predictions available for evaluation.")

if __name__ == "__main__":
    main()
