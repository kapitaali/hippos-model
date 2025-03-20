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

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_URL = "https://heppa.hippos.fi/heppa2_backend"
HEADERS = {'Content-Type': 'application/json'}

class HorseRaceDataset(Dataset):
    def __init__(self, data, scaler=None):
        self.scaler = scaler if scaler else StandardScaler()
        self.features = []
        self.labels = []
        self.horse_info = []
        
        self.process_api_data(data)

        print(f"Processed {len(self.features)} valid entries from {len(data)} items")
        if not self.features:
            print("No valid data processed")
            return
            
        if scaler:
            self.features = self.scaler.transform(np.array(self.features))
            print("Applied precomputed scaler to features")
        else:
            self.features = self.scaler.fit_transform(np.array(self.features))
            print("Fitted new scaler to features (no precomputed scaler provided)")
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
                0  # Placeholder
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

def fetch_api_data(endpoint):
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        raw_text = response.text
        try:
            print(f"Raw response from {endpoint}: {raw_text[:500]}...")  # Truncate for readability
        except BrokenPipeError:
            logger.debug(f"Pipe broken, logging raw response from {endpoint}: {raw_text[:500]}...")
        logger.debug(f"Raw response from {endpoint}: {raw_text}")
        response.raise_for_status()
        data = response.json()
        print(f"Endpoint: {endpoint} - Status: {response.status_code}")
        return data
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        logger.error(f"Error fetching {url}: {e}")
        try:
            print(f"Raw response: {raw_text}")
        except BrokenPipeError:
            logger.debug(f"Pipe broken, logging raw response: {raw_text}")
        return None

def fetch_races(date, track):
    enriched_races = []
    is_past_date = datetime.strptime(date, "%Y-%m-%d") <= datetime.now()
    
    if is_past_date:
        print(f"Attempting scraped data for past date {date}/{track}")
        try:
            starts = fetch_api_data(f"/race/{date}/{track}/races")
            if not starts or not isinstance(starts, list):
                print(f"No valid starts for {date}/{track}")
                return enriched_races
#            with open('./horse-race-predictor/racedata/scraped_race_data_2018-2024.json', 'r') as f:
#                scraped = json.load(f)
#            starts = scraped.get(date, {}).get(track, [])
#            if not starts or not isinstance(starts, list):
#                print(f"No scraped starts found for {date}/{track}")
#            else:
#                print(f"Found {len(starts)} scraped starts for {date}/{track}")
        except Exception as e:
            print(f"Scrape failed: {e}")
            starts = []
    else:
        print(f"Fetching races for future date {date}/{track} using /races")
        starts = fetch_api_data(f"/race/{date}/{track}/races")
        if not starts or not isinstance(starts, list):
            print(f"No valid starts for {date}/{track}")
            return enriched_races

    if not starts or not isinstance(starts, list) or len(starts) == 0:
        #fallback_date = "2024-06-01"
        #fallback_track = "KJ"
        #print(f"Falling back to {fallback_date}/{fallback_track} using /races")
        #starts = fetch_api_data(f"/race/{fallback_date}/{fallback_track}/races")
        if not starts or not isinstance(starts, list):
            print(f"No valid starts for {date}")
        #    return enriched_races
        #date, track = fallback_date, fallback_track

    for start in starts:
        start_number = start.get('race', {}).get('startNumber', '')
        if not start_number:
            print(f"Skipping start with no startNumber: {start}")
            continue
        
        horses = fetch_api_data(f"/race/{date}/{track}/start/{start_number}")
        if not horses or not isinstance(horses, list):
            print(f"No horses for {date}/{track}/start/{start_number}")
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
    
    print(f"Enriched {len(enriched_races)} races")
    if enriched_races:
        print(f"Sample race data: {json.dumps(enriched_races[0], indent=2)}")
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
    parser = argparse.ArgumentParser(description="Predict horse race outcomes")
    parser.add_argument('date', type=str, help='YYYY-MM-DD')
    args = parser.parse_args()
    date = args.date

    data_dir = './horse-race-predictor/racedata/8feat/'
    model_path = os.path.join(data_dir, 'horse_race_predictor.pth')
    scaler_path = os.path.join(data_dir, 'scaler.pkl')
    
    model = RacePredictor()
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    print("Model loaded successfully")

    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Loaded precomputed scaler")

    print("Fetching race data...")
    api_data = fetch_races(date, get_track_code(date)[0])
    if not api_data:
        print("No race data retrieved")
        return
    
    predict_dataset = HorseRaceDataset(api_data, scaler=scaler)
    if len(predict_dataset) == 0:
        print("No valid data to predict")
        return
    
    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False, 
                               num_workers=0, pin_memory=False)
    
    print("Predicting win probabilities...")
    predictions = predict_races(model, predict_loader, predict_dataset)
    
    predictions_json = json.dumps(predictions, indent=2)
    print("Predictions JSON:")
    print(predictions_json)

if __name__ == "__main__":
    main()
