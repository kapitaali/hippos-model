import torch
import torch.nn as nn
                               
# --- Model and Training Parameters ---
BATCH_SIZE = 10000
SAMPLE_SIZE = 100000
MAX_RAM_MB = 50000

INPUT_SIZE = 8
HIDDEN_LAYERS = [
    (128, 0.4),
    (96, 0.4),
    (64, 0.3),
    (32, 0.3),
    (16, 0.0),
]
OUTPUT_SIZE = 1

NUM_EPOCHS = 20
LEARNING_RATE = 0.001
POS_WEIGHT = 10.0

# --- End of Parameters ---

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


model = RacePredictor()
model.load_state_dict(torch.load('./racedata/8feat/horse_race_predictor.pth'))
model.eval()

dummy_input = torch.randn(1, 8)  # 8 features, change this and the model definition if you're doing 14feats
torch.onnx.export(model, dummy_input, './racedata/8feat/horse_race_predictor.onnx', 
                  input_names=['input'], output_names=['output'], opset_version=11)

