import json
import os
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from tqdm import tqdm
import logging
import ijson
import multiprocessing
import argparse
from torch.onnx import export

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a 14-feature horse race predictor model")
parser.add_argument('-d', '--debug', type=int, default=20, choices=[10, 20, 30, 40, 50],
                    help="Debug level: 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL (default: 20)")
parser.add_argument('--pos-weight', type=float, default=50.0, help='Positive class weight for loss function (default: 50.0)')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for optimizer (default: 0.001)')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
parser.add_argument('--data-path', type=str, default='./horse-race-predictor/racedata/scraped_race_data_2018-2024.json',
                    help='Path to combined race data JSON (default: ./horse-race-predictor/racedata/scraped_race_data_2018-2024.json)')
parser.add_argument('--model-path', type=str, default='./horse-race-predictor/racedata/14feat/horse_race_predictor_14feat.pth',
                    help='Path to save trained model (default: ./horse-race-predictor/racedata/14feat/horse_race_predictor.pth)')
parser.add_argument('--scaler-path', type=str, default='./horse-race-predictor/racedata/14feat/scaler_14feat.pkl',
                    help='Path to save scaler data (default: ./horse-race-predictor/racedata/14feat/scaler_14feat.pkl)')
parser.add_argument('--onnx-path', type=str, default='./horse-race-predictor/racedata/14feat/horse_race_predictor_14feat.onnx',
                    help='Path to save ONNX model (default: ./horse-race-predictor/racedata/14feat/horse_race_predictor_14feat.onnx)')
args = parser.parse_args()

# Setup logging
logging.basicConfig(level=args.debug, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- Model and Training Parameters ---
BATCH_SIZE = 10000
SAMPLE_SIZE = 100000
MAX_RAM_MB = 50000

INPUT_SIZE = 14
HIDDEN_LAYERS = [
    (128, 0.4),
    (96, 0.4),
    (64, 0.3),
    (32, 0.3),
    (16, 0.0),
]
OUTPUT_SIZE = 1

NUM_EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
POS_WEIGHT = args.pos_weight
onnx_path = args.onnx_path

# --- End of Parameters ---

class HorseRaceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)  # Keep on CPU until training
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class RacePredictor(nn.Module):
    def __init__(self):
        super(RacePredictor, self).__init__()
        layers = []
        prev_size = INPUT_SIZE
        for neurons, dropout in HIDDEN_LAYERS:
            layers.append(nn.Linear(prev_size, neurons))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = neurons
        layers.append(nn.Linear(prev_size, OUTPUT_SIZE))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

def combine_chunks(input_dir, output_file):
    all_data = []
    chunk_files = glob(os.path.join(input_dir, 'scraped_race_data_*.json'))
    logger.info(f"Found {len(chunk_files)} chunk files")
    
    for file in chunk_files:
        with open(file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
        logger.info(f"Loaded {file} - Total entries: {len(all_data)}")
    
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    logger.info(f"Combined data saved to {output_file} with {len(all_data)} entries")

def process_batch(races, scaler=None, track_map=None):
    features = []
    labels = []
    win_count = 0
    
    if track_map is None:
        track_map = {}
    
    for race in races:
        race_info = race.get('race_info', {})
        starts = race.get('starts', [])
        race_data = race_info.get('race', {})
        toto_results = race_info.get('totoResults', [])
        
        track_code = race_data.get('trackCode', 'Unknown')
        logger.debug(f"Race trackCode: {track_code}, Starts: {len(starts)}, TotoResults: {len(toto_results)}")
        
        if not starts:
            logger.debug(f"No starts found for {track_code}")
            continue
        
        if track_code not in track_map:
            track_map[track_code] = len(track_map)
        track_id = track_map[track_code]
        
        result_map = {}
        for result in toto_results:
            if result.get('placing') == '1' and result.get('gameType') == 'VOITTAJA':
                linenumbers = result.get('linenumbers')
                result_map[linenumbers] = 1
            else:
                linenumbers = result.get('linenumbers')
                if linenumbers and linenumbers not in result_map:
                    result_map[linenumbers] = 0
        
        for start in starts:
            horse_data = start.get('horse_data', {})
            #horse_stats = next((s for s in start.get('horse_stats', {}).get('stats', []) if s.get('year') == '2024'), {})
            horse_stats = start.get('horse_stats', {}).get('total', {})  # Use allTotal for consistency            
            driver_stats = start.get('driver_stats', {}).get('allTotal', {})  # Use allTotal for consistency
            
            program_number = horse_data.get('programNumber', '0')
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
            
            label = result_map.get(program_number, 0)
            if label == 1:
                win_count += 1
                logger.debug(f"Found a win for programNumber: {program_number}")
            
            if not features and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Sample horse_data: {json.dumps(horse_data, indent=2)}")
                logger.debug(f"Sample driver_stats (allTotal): {json.dumps(driver_stats, indent=2)}")
            
            features.append(feature_vector)
            labels.append(label)
    
    if not features:
        logger.warning("No valid features in batch")
        return None, None, track_map
    
    features = np.array(features)
    logger.info(f"Total wins in batch: {win_count}/{len(labels)}")
    logger.debug(f"Batch features shape: {features.shape}, Wins: {win_count}/{len(labels)}")
    if scaler:
        features = scaler.transform(features)
        logger.debug(f"First scaled feature vector: {features[0].tolist()}")
    else:
        logger.debug(f"First raw feature vector: {features[0].tolist()}")
    return features, labels, track_map

def stream_json_data(file_path, batch_size=BATCH_SIZE):
    with open(file_path, 'rb') as f:
        parser = ijson.items(f, 'item')
        races = []
        total_races = 0
        
        for race_dict in parser:
            for date, tracks in race_dict.items():
                for track_code, starts_dict in tracks.items():
                    for start_number, data in starts_dict.items():
                        race_info = data.get('race_info', {})
                        starts = data.get('starts', [])
                        races.append({'race_info': race_info, 'starts': starts})
                        total_races += 1
                        if len(races) >= batch_size:
                            logger.debug(f"Yielding batch of {len(races)} races, Total so far: {total_races}")
                            yield races
                            races = []
        
        if races:
            logger.debug(f"Yielding final batch of {len(races)} races, Total: {total_races}")
            yield races
    logger.info(f"Total races processed: {total_races}")

def train_model_incrementally(data_path, model_path, scaler_path, onnx_path, val_split=0.2):
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Detected {num_cores} CPU cores")

    model = RacePredictor()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss(weight=torch.tensor([POS_WEIGHT]).to(device))
    
    scaler = StandardScaler()
    track_map = {}
    sample_features = []
    sampled = 0
    
    logger.info("Fitting scaler on sample data...")
    batch_gen = stream_json_data(data_path)
    for batch_idx, batch in enumerate(batch_gen):
        features, _, track_map = process_batch(batch, track_map=track_map)
        if features is None:
            logger.warning(f"Batch {batch_idx} has no valid features")
            continue
        sample_features.append(features)
        sampled += len(features)
        logger.info(f"Sampled {sampled}/{SAMPLE_SIZE} entries")
        if sampled >= SAMPLE_SIZE:
            break
    
    if not sample_features:
        raise ValueError("No valid data found in sample for scaler fitting")
    
    sample_features = np.vstack(sample_features)
    scaler.fit(sample_features)
    logger.info(f"Scaler fitted on {sampled} samples")
    logger.debug(f"Scaler mean: {scaler.mean_.tolist()}")
    logger.debug(f"Scaler scale: {scaler.scale_.tolist()}")
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    scaler_data = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}
    with open(scaler_path.replace('.pkl', '_data.json'), 'w') as f:
        json.dump(scaler_data, f, indent=2)
    with open(scaler_path.replace('.pkl', '_track_map.json'), 'w') as f:
        json.dump(track_map, f, indent=2)
    
    logger.info("Collecting data for train/validation split...")
    all_features = []
    all_labels = []
    total_wins = 0
    for batch in stream_json_data(data_path):
        features, labels, _ = process_batch(batch, scaler=scaler, track_map=track_map)
        if features is None:
            continue
        all_features.append(features)
        all_labels.append(labels)
        total_wins += sum(labels)
    all_features = np.vstack(all_features)
    all_labels = np.hstack(all_labels)
    logger.info(f"Total samples: {len(all_labels)}, Total wins: {total_wins}")
    
    if total_wins == 0:
        raise ValueError("No winning horses found in the dataset!")
    
    from sklearn.model_selection import train_test_split
    train_features, val_features, train_labels, val_labels = train_test_split(
        all_features, all_labels, test_size=val_split, random_state=42
    )
    train_dataset = HorseRaceDataset(train_features, train_labels)
    val_dataset = HorseRaceDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10, pin_memory=True)  # Reduced workers
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=10, pin_memory=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        train_batches = 0
        for feat, lab in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            feat, lab = feat.to(device), lab.to(device)  # Move to device here
            optimizer.zero_grad()
            outputs = model(feat)
            loss = criterion(outputs, lab)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_batches += 1
        
        model.eval()
        total_val_loss = 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for feat, lab in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                feat, lab = feat.to(device), lab.to(device)
                outputs = model(feat)
                loss = criterion(outputs, lab)
                total_val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(lab.cpu().numpy())
        
        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_preds = np.array(val_preds)
        val_true = np.array(val_true)
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(val_preds, bins) - 1
        for i in range(len(bins)-1):
            bin_mask = bin_indices == i
            if bin_mask.sum() > 0:
                bin_true = val_true[bin_mask].mean()
                bin_pred = val_preds[bin_mask].mean()
                logger.info(f"  Prob {bins[i]:.1f}-{bins[i+1]:.1f}: Pred {bin_pred:.3f}, True {bin_true:.3f}, Count {bin_mask.sum()}")

        torch.save(model.state_dict(), model_path + f".epoch{epoch+1}")
    
    torch.save(model.state_dict(), model_path)
    logger.info(f"Final model saved to {model_path}")

    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, INPUT_SIZE).to(device)
    export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'], opset_version=11)
    logger.info(f"Model exported to ONNX at {onnx_path}")

def main():
    data_dir = './horse-race-predictor/racedata/'
    combined_data_path = args.data_path
    model_path = args.model_path
    scaler_path = args.scaler_path
    onnx_path = args.onnx_path
    
    if not os.path.exists(combined_data_path):
        logger.info("Combining monthly data chunks...")
        combine_chunks(data_dir, combined_data_path)
    
    logger.info(f"Training with POS_WEIGHT={POS_WEIGHT}, LEARNING_RATE={LEARNING_RATE}, EPOCHS={NUM_EPOCHS}")
    train_model_incrementally(combined_data_path, model_path, scaler_path, onnx_path)

if __name__ == "__main__":
    main()
