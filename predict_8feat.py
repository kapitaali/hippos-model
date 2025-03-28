import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Predict horse race outcomes with an 8-feature model")
parser.add_argument('date', type=str, help='Date in YYYY-MM-DD format')
parser.add_argument('--track', type=str, default=None, help='Track code (e.g., "H"); if not provided, fetched from API')
parser.add_argument('--pos-weight', type=float, default=5.0, help='Positive weight used in training (default: 5.0)')
parser.add_argument('--learning-rate', type=float, default=0.005, help='Learning rate used in training (default: 0.005)')
parser.add_argument('--model-path', type=str, default=None, help='Path to trained model (overrides pos-weight and learning-rate if provided)')
parser.add_argument('--scaler-path', type=str, default=None, help='Path to scaler file (overrides pos-weight and learning-rate if provided)')
parser.add_argument('--output-file', type=str, default=None, help='Path to save predictions as JSON (optional)')
parser.add_argument('--debug', type=int, default=20, choices=[10, 20, 30, 40, 50],
                    help='Debug level: 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL (default: 20)')
parser.add_argument('--track-map-path', type=str, default='./horse-race-predictor/racedata/14feat/scaler_track_map.json',
                    help='Path to track map file (default: ./horse-race-predictor/racedata/14feat/scaler_track_map.json)')
args = parser.parse_args()

# Setup logging
logging.basicConfig(level=args.debug, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

BASE_URL = "https://heppa.hippos.fi/heppa2_backend"
HEADERS = {'Content-Type': 'application/json'}
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Dynamically set model and scaler paths based on pos-weight and learning-rate unless overridden
MODEL_DIR = './models/8feat'
if args.model_path is None:
    args.model_path = f"{MODEL_DIR}/horse_race_predictor_{args.pos_weight}pos_{args.learning_rate}learn.pth"
if args.scaler_path is None:
    args.scaler_path = f"{MODEL_DIR}/scaler_{args.pos_weight}pos_{args.learning_rate}learn.pkl"

class HorseRaceDataset(Dataset):
    def __init__(self, data, scaler=None):
        self.scaler = scaler if scaler else StandardScaler()
        self.features = []
        self.labels = []
        self.horse_info = []
        
        self.process_api_data(data)

        logger.info(f"Processed {len(self.features)} valid entries from {len(data)} items")
        if not self.features:
            logger.warning("No valid data processed")
            return
            
        if scaler:
            self.features = self.scaler.transform(np.array(self.features))
            logger.info("Applied precomputed scaler to features")
        else:
            self.features = self.scaler.fit_transform(np.array(self.features))
            logger.info("Fitted new scaler to features (no precomputed scaler provided)")
        self.features = torch.FloatTensor(self.features).to(device)
        self.labels = torch.FloatTensor(self.labels).to(device)

    def process_api_data(self, api_data):
        for race in tqdm(api_data, desc="Processing API data"):
            if not isinstance(race, dict):
                logger.error(f"Invalid race data: {race}")
                continue
            
            date = race.get('raceDate', 'Unknown')
            track = race.get('trackCode', 'Unknown')
            start_number = race.get('startNumber', 'Unknown')
            horse_data = race.get('horse_data', {})
            horse_stats = race.get('horse_stats', {}).get('total', {})  # Use total
            driver_stats = race.get('driver_stats', {}).get('allTotal', {})  # Use allTotal
            
            feature_vector = [
                float(horse_stats.get('winningPercent', 0)),
                float(horse_stats.get('priceMoney', 0)),
                float(horse_stats.get('starts', 0)),
                float(driver_stats.get('winPercentage', 0)),
                float(driver_stats.get('priceMoney', 0)),
                float(driver_stats.get('starts', 0)),
                float(horse_data.get('lane', 0)),
                0  # Placeholder for track_id (not used in prediction here)
            ]
            
            self.features.append(feature_vector)
            self.labels.append(0)  # No labels for prediction
            self.horse_info.append({
                'date': date,
                'track': track,
                'start': start_number,
                'horse': horse_data.get('horseName', 'Unknown'),
                'driver': horse_data.get('driverName', 'Unknown')
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class RacePredictor(nn.Module):
    def __init__(self, input_size=8):
        super(RacePredictor, self).__init__()
        layers = []
        prev_size = input_size
        
        hidden_configs = [
            (128, 0.4),
            (96, 0.4),
            (64, 0.3),
            (32, 0.3),
            (16, 0.0)
        ]
        
        for size, dropout in hidden_configs:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

def fetch_api_data(endpoint, base_url=BASE_URL, headers=HEADERS, cache_dir=CACHE_DIR):
    """Fetch data from API with caching."""
    cache_key = f"predict_8feat_{endpoint.replace('/', '_').replace('?', '_').replace('&', '_')}"
    cache_file = Path(cache_dir) / f"{cache_key}.json"
    
    if cache_file.exists():
        logger.info(f"Loading cached data for {endpoint} from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    url = f"{base_url}{endpoint}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        raw_text = response.text
        logger.debug(f"Raw response from {endpoint}: {raw_text[:500]}...")
        response.raise_for_status()
        data = response.json()
        logger.info(f"Endpoint: {endpoint} - Status: {response.status_code}")
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Cached data to {cache_file}")
        return data
    except requests.RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

def fetch_races(date, track):
    enriched_races = []
    is_past_date = datetime.strptime(date, "%Y-%m-%d") <= datetime.now()
    
    if is_past_date:
        logger.info(f"Attempting data for past date {date}/{track}")
        starts = fetch_api_data(f"/race/{date}/{track}/races")
        if not starts or not isinstance(starts, list):
            logger.warning(f"No valid starts for {date}/{track}")
            return enriched_races
    else:
        logger.info(f"Fetching races for future date {date}/{track} using /races")
        starts = fetch_api_data(f"/race/{date}/{track}/races")
        if not starts or not isinstance(starts, list):
            logger.warning(f"No valid starts for {date}/{track}")
            return enriched_races

    for start in starts:
        start_number = start.get('race', {}).get('startNumber', '')
        if not start_number:
            logger.debug(f"Skipping start with no startNumber: {start}")
            continue
        
        horses = fetch_api_data(f"/race/{date}/{track}/start/{start_number}")
        if not horses or not isinstance(horses, list):
            logger.warning(f"No horses for {date}/{track}/start/{start_number}")
            continue
        
        for horse in horses:
            horse_id = horse.get('horseId', '')
            driver_id = horse.get('driverId', '')
            horse_stats = fetch_api_data(f"/horse/{horse_id}/stats") if horse_id else {}
            driver_stats = fetch_api_data(f"/driver/{driver_id}/stats") if driver_id else {}
            
            enriched_races.append({
                'raceDate': date,
                'trackCode': track,
                'startNumber': start_number,
                'horse_data': horse,
                'horse_stats': horse_stats or {},
                'driver_stats': driver_stats or {}
            })
    
    logger.info(f"Enriched {len(enriched_races)} races")
    if enriched_races and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Sample race data: {json.dumps(enriched_races[0], indent=2)}")
    return enriched_races

def predict_races(model, predict_loader, dataset):
    model.eval()
    predictions = []

    with torch.no_grad():
        for features, _ in predict_loader:
            features = features.to(device)
            outputs = model(features)
            probabilities = outputs.cpu().numpy().tolist()
            predictions.extend(probabilities)

    results = []
    for i, prob in enumerate(predictions):
        horse_info = dataset.horse_info[i]
        results.append({
            'date': horse_info['date'],
            'track': horse_info['track'],
            'start': horse_info['start'],
            'horse': horse_info['horse'],
            'driver': horse_info['driver'],
            'win_probability': prob
        })

    # Normalize per start
    start_groups = {}
    for result in results:
        key = (result['date'], result['track'], result['start'])
        if key not in start_groups:
            start_groups[key] = []
        start_groups[key].append(result)

    for key, group in start_groups.items():
        total_prob = sum(r['win_probability'] for r in group)
        if total_prob > 0:
            for result in group:
                result['win_probability'] /= total_prob

    return results

def get_track_code(date):
    endpoint = f"/race/search/{date}/{date}/"
    data = fetch_api_data(endpoint)
    if data and isinstance(data, list) and data[0].get('events'):
        tracks = [event['trackCode'] for event in data[0]['events']]
        logger.debug(f"Found tracks for {date}: {tracks}")
        return tracks
    logger.error(f"No tracks found for {date}")
    return None

def main():
    date = args.date
    track = args.track if args.track else get_track_code(date)[0] if get_track_code(date) else 'H'  # Default to 'H' if no track found
    
    model_path = args.model_path
    scaler_path = args.scaler_path
    
    model = RacePredictor()
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    logger.info("Model loaded successfully")

    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Loaded precomputed scaler")
    else:
        logger.warning(f"Scaler file not found at {scaler_path}, proceeding without scaler")

    logger.info(f"Fetching race data for {date}/{track}...")
    api_data = fetch_races(date, track)
    if not api_data:
        logger.error("No race data retrieved")
        return
    
    predict_dataset = HorseRaceDataset(api_data, scaler=scaler)
    if len(predict_dataset) == 0:
        logger.error("No valid data to predict")
        return
    
    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False, 
                               num_workers=0, pin_memory=False)
    
    logger.info("Predicting win probabilities...")
    predictions = predict_races(model, predict_loader, predict_dataset)
    
    predictions_json = json.dumps(predictions, indent=2)
    logger.info("Predictions generated")
    print("Predictions JSON:")
    print(predictions_json)

    if args.output_file:
        try:
            with open(args.output_file, 'w') as f:
                f.write(predictions_json)
            logger.info(f"Predictions saved to {args.output_file}")
        except Exception as e:
            logger.error(f"Failed to write predictions to {args.output_file}: {e}")

if __name__ == "__main__":
    main()
