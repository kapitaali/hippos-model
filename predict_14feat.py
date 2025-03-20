import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import requests
import json
import os
import pickle
from datetime import datetime
import argparse
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_URL = "https://heppa.hippos.fi/heppa2_backend"
HEADERS = {'Content-Type': 'application/json'}

logger = logging.getLogger(__name__)
logger.info(f"Using device: {device}")


class HorseRaceDataset(Dataset):
    def __init__(self, data, scaler=None, track_map=None):
        self.scaler = scaler
        self.track_map = track_map if track_map else {}
        self.features = []
        self.labels = []
        self.horse_info = []
        
        self.process_api_data(data)

        logger.info(f"Processed {len(self.features)} valid entries from {len(data)} items")
        if not self.features:
            logger.warning("No valid data processed")
            return
        
        raw_features = np.array(self.features)
        logger.debug(f"First raw feature vector: {raw_features[0].tolist()}")
        
        if self.scaler:
            logger.debug(f"Scaler expected features: {self.scaler.n_features_in_}")
            if self.scaler.n_features_in_ != raw_features.shape[1]:
                logger.error(f"Scaler mismatch: expected {self.scaler.n_features_in_} features, got {raw_features.shape[1]}")
                raise ValueError("Feature count mismatch between scaler and data")
            self.features = self.scaler.transform(raw_features)
            logger.info("Applied precomputed scaler to features")
            logger.debug(f"First scaled feature vector: {self.features[0].tolist()}")
            logger.debug(f"Scaler mean: {self.scaler.mean_.tolist()}")
            logger.debug(f"Scaler scale: {self.scaler.scale_.tolist()}")
        else:
            logger.warning("No scaler provided; features unscaled")
        
        self.features = torch.FloatTensor(self.features).to(device)
        self.labels = torch.FloatTensor(self.labels).to(device)

    def process_api_data(self, data):
        for race in data:
            if not isinstance(race, dict):
                logger.error(f"Invalid race data: {race}")
                continue
            
            track_code = race.get('trackCode', 'Unknown')
            track_id = self.track_map.get(track_code, len(self.track_map))
            date = race.get('raceDate', 'Unknown')
            start_number = race.get('startNumber', 'Unknown')
            horse_data = race.get('horse_data', {})
            horse_stats = race.get('horse_stats', {}).get('total', {})
            driver_stats = race.get('driver_stats', {}).get('allTotal', {})
            
            feature_vector = [
                float(horse_stats.get('winningPercent', 0)),
                float(horse_stats.get('priceMoney', 0)),
                float(horse_stats.get('starts', 0)),
                float(driver_stats.get('winPercentage', 0)),
                float(driver_stats.get('priceMoney', 0)),
                float(driver_stats.get('starts', 0)),
                float(horse_data.get('lane', 0)),
                float(track_id),
                float(horse_stats.get('gallopPercentage', 0)),
                float(horse_stats.get('disqualificationPercentage', 0)),
                float(horse_stats.get('improperGaitPercentage', 0) or 0),
                float(driver_stats.get('gallop', 0)) / max(float(driver_stats.get('starts', 1)), 1) * 100,
                float(driver_stats.get('disqualified', 0)) / max(float(driver_stats.get('starts', 1)), 1) * 100,
                float(driver_stats.get('improperGait', 0)) / max(float(driver_stats.get('starts', 1)), 1) * 100
            ]
            
            self.features.append(feature_vector)
            self.labels.append(0)
            self.horse_info.append({
                'date': date,
                'track': track_code,
                'start': start_number,
                'horse': horse_data.get('horseName', 'Unknown'),
                'driver': horse_data.get('driverName', 'Unknown')
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class RacePredictor(nn.Module):
    def __init__(self, input_size=14):
        super(RacePredictor, self).__init__()
        layers = []
        prev_size = input_size
        hidden_configs = [(128, 0.4), (96, 0.4), (64, 0.3), (32, 0.3), (16, 0.0)]
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

def fetch_api_data(endpoint):
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Fetched data from {endpoint} - Status: {response.status_code}")
        return data
    except requests.RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

def fetch_races(date, track):
    enriched_races = []
    logger.info(f"Fetching races for {date}/{track}")
    
    starts = fetch_api_data(f"/race/{date}/{track}/races")
    if not starts or not isinstance(starts, list):
        logger.warning(f"No valid starts for {date}/{track}, falling back to default")
        fallback_date = "2024-06-01"
        fallback_track = "KJ"
        starts = fetch_api_data(f"/race/{fallback_date}/{fallback_track}/races")
        if not starts or not isinstance(starts, list):
            logger.error(f"No valid starts for fallback {fallback_date}/{fallback_track}")
            return enriched_races
        date, track = fallback_date, fallback_track

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
            if len(predictions) < 5:
                formatted_probs = [f"{p:.10f}" for p in probabilities[:5]]
                logger.debug(f"Raw model outputs (first 5, 10 decimals): {formatted_probs}")
            predictions.extend(probabilities)

    # Group by race (start number)
    race_probs = {}
    for i, prob in enumerate(predictions):
        start = dataset.horse_info[i]['start']
        race_probs.setdefault(start, []).append((i, prob))
    
    # Normalize per race
    for start, probs in race_probs.items():
        total = sum(p[1] for p in probs)
        if total > 0:
            for idx, _ in probs:
                predictions[idx] /= total

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
    parser = argparse.ArgumentParser(description="Predict horse race outcomes")
    parser.add_argument('date', type=str, help='YYYY-MM-DD')
    parser.add_argument('-d', '--debug', type=int, default=20, 
                        choices=[0, 10, 20, 30, 40], 
                        help="Debug level: 0=silent, 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR (default: 20)")
    args = parser.parse_args()

    if args.debug == 0:
        logging.disable(logging.CRITICAL)
    else:
        logging.basicConfig(level=args.debug, format='%(levelname)s: %(message)s')

    date = args.date
    data_dir = './horse-race-predictor/racedata/'
    model_path = os.path.join(data_dir, 'horse_race_predictor.pth')
    scaler_path = os.path.join(data_dir, 'scaler.pkl')
    track_map_path = os.path.join(data_dir, 'scaler_track_map.json')

    model = RacePredictor()
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    logger.debug(f"Model weights loaded: {any(p.nonzero().size(0) > 0 for p in model.parameters())}")
    logger.info("Model loaded successfully")

    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Loaded precomputed scaler")
    track_map = {}
    if os.path.exists(track_map_path):
        with open(track_map_path, 'r') as f:
            track_map = json.load(f)
        logger.info(f"Loaded track map with {len(track_map)} entries")

    tracks = get_track_code(date)
    if not tracks:
        logger.error("No tracks retrieved; exiting")
        return
    track = tracks[0]
    api_data = fetch_races(date, track)
    if not api_data:
        logger.error("No race data retrieved; exiting")
        return

    predict_dataset = HorseRaceDataset(api_data, scaler=scaler, track_map=track_map)
    if len(predict_dataset) == 0:
        logger.error("No valid data to predict; exiting")
        return
    
    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    logger.info("Predicting win probabilities...")
    predictions = predict_races(model, predict_loader, predict_dataset)

    formatted_predictions = [
        {k: f"{v:.10f}" if k == 'win_probability' else v for k, v in pred.items()}
        for pred in predictions
    ]
    logger.info("Predictions JSON:")
    print(json.dumps(formatted_predictions, indent=2))

if __name__ == "__main__":
    main()
