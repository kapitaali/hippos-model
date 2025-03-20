/* v1.6 */
const express = require('express');
const ort = require('onnxruntime-node');
const fs = require('fs').promises;
const path = require('path');
const axios = require('axios');

const app = express();
app.use(express.json());

const MODEL_PATH = path.join(__dirname, 'racedata/8feat', 'horse_race_predictor.onnx');
const SCALER_DATA_PATH = path.join(__dirname, 'racedata/8feat', 'scaler_data.json');
const TRACK_MAP_PATH = path.join(__dirname, 'racedata/8feat', 'scaler_track_map.json');
const RACING_API_URL = 'https://heppa.hippos.fi/heppa2_backend';

let session, scalerData, trackMap;

async function loadModelAndScaler() {
    try {
        session = await ort.InferenceSession.create(MODEL_PATH);
        console.log('Model loaded successfully');

        const scalerJson = await fs.readFile(SCALER_DATA_PATH, 'utf8');
        scalerData = JSON.parse(scalerJson);
        console.log('Scaler data loaded:', scalerData);

        const trackMapJson = await fs.readFile(TRACK_MAP_PATH, 'utf8');
        trackMap = JSON.parse(trackMapJson);
        console.log('Track map loaded:', Object.keys(trackMap).length, 'tracks');
    } catch (err) {
        console.error('Error loading model or scaler:', err);
        process.exit(1);
    }
}

function standardize(features) {
    return features.map((val, i) => 
        (val - scalerData.mean[i]) / scalerData.scale[i]
    );
}

async function predictHorse(horseData, trackCode) {
    const horseName = horseData.horse_data.horseName || 'Unknown';
    console.log(`Predicting for horse: ${horseName}`);
    const horseStats = horseData.horse_stats?.total || {};
    const driverStats = horseData.driver_stats?.allTotal || {};
    const features = [
        parseFloat(horseStats.winningPercent || 0),
        parseFloat(horseStats.priceMoney || 0),
        parseFloat(horseStats.starts || 0),
        parseFloat(driverStats.winPercentage || 0),
        parseFloat(driverStats.priceMoney || 0),
        parseFloat(driverStats.starts || 0),
        parseFloat(horseData.lane || 0),
        parseFloat(trackMap[trackCode] || 0),
        parseFloat(horseStats.gallopPercentage || 0),
        parseFloat(horseStats.disqualificationPercentage || 0),
        parseFloat(horseStats.improperGaitPercentage || 0),
        parseFloat((driverStats.gallop || 0) / Math.max(driverStats.starts || 1, 1) * 100),
        parseFloat((driverStats.disqualified || 0) / Math.max(driverStats.starts || 1, 1) * 100),
        parseFloat((driverStats.improperGait || 0) / Math.max(driverStats.starts || 1, 1) * 100)
    ];

    console.log('Raw features:', features);
    const standardizedFeatures = standardize(features);
    console.log('Standardized features:', standardizedFeatures);

    const tensor = new ort.Tensor('float32', standardizedFeatures, [1, 14]);
    const feeds = { input: tensor };

    try {
        const results = await session.run(feeds);
        console.log('Model output:', results);
        if (!results.output || !results.output.data || results.output.data.length === 0) {
            throw new Error('Model returned invalid output');
        }
        const rawProb = results.output.data[0];
        return Math.max(rawProb, 0.001); // Minimum probability of 0.001
    } catch (err) {
        console.error('Inference error:', err);
        return 0.001; // Default to 0.001 on error
    }
}

async function fetchApiData(endpoint) {
    const url = `${RACING_API_URL}${endpoint}`;
    try {
        const response = await axios.get(url);
        return response.data;
    } catch (err) {
        console.error(`Error fetching ${endpoint}: ${err.message}`);
        return null;
    }
}

async function fetchAndPredictRaces(date, track) {
    let predictions = {};
    const isPastDate = new Date(date) < new Date(); // Today’s date

    console.log(`Processing races for ${date}/${track} - Past date: ${isPastDate}`);

    let starts;
    /*
    if (isPastDate) {
        console.log('Past dates not supported yet—use future dates for predictions');
        return predictions; // Empty for past dates
    } else {
    */
        starts = await fetchApiData(`/race/${date}/${track}/races`);
    //}

    if (!starts || !Array.isArray(starts) || starts.length === 0) {
        console.log(`No valid starts for ${date}/${track}`);
        return predictions;
    }

    predictions = {};
    for (const start of starts) {
        const startNumber = start.race?.startNumber;
        if (!startNumber) {
            console.log(`Skipping start with no startNumber: ${JSON.stringify(start)}`);
            continue;
        }

        const horses = await fetchApiData(`/race/${date}/${track}/start/${startNumber}`);
        if (!horses || !Array.isArray(horses)) {
            console.log(`No horses for ${date}/${track}/start/${startNumber}`);
            continue;
        }

        predictions[startNumber] = {};
        for (const horse of horses) {
            const horseId = horse.horseId || '';
            const driverId = horse.driverId || '';
            const horseStats = horseId ? await fetchApiData(`/horse/${horseId}/stats`) : {};
            const driverStats = driverId ? await fetchApiData(`/driver/${driverId}/stats`) : {};

            const enrichedHorse = {
                horse_data: horse,
                horse_stats: horseStats,
                driver_stats: driverStats,
                lane: horse.lane || 0
            };
            const probability = await predictHorse(enrichedHorse, track);
            const programNumber = horse.programNumber || 'unknown';
            
            const horseName = horse.horseName || horse.name || 'Unknown Horse';
            const riderName = horse.driverName || 'Unknown Driver';

            predictions[startNumber][programNumber] = {
                horseName,
                riderName,
                winProbability: probability
            };
        }
    }

    for (const startNumber in predictions) {
        const horses = predictions[startNumber];
        const totalProb = Object.values(horses).reduce((sum, h) => sum + (h.winProbability || 0), 0);
        if (totalProb > 0) {
            // Step 1: Initial normalization
            for (const programNumber in horses) {
                horses[programNumber].winProbability = horses[programNumber].winProbability / totalProb;
            }
            // Step 2: Apply minimum probability floor (e.g., 0.01)
            for (const programNumber in horses) {
                horses[programNumber].winProbability = Math.max(horses[programNumber].winProbability, 0.03);
            }
            // Step 3: Renormalize to sum to 1.0
            const newTotal = Object.values(horses).reduce((sum, h) => sum + h.winProbability, 0);
            for (const programNumber in horses) {
                horses[programNumber].winProbability = (horses[programNumber].winProbability / newTotal).toFixed(4);
            }
        } else {
            // If totalProb is 0, assign uniform probabilities
            const numHorses = Object.keys(horses).length;
            for (const programNumber in horses) {
                horses[programNumber].winProbability = (1 / numHorses).toFixed(4);
            }
        }
    }

    console.log(`Generated predictions for ${Object.keys(predictions).length} races`);
    console.log('Final predictions:', JSON.stringify(predictions, null, 2));
    return predictions;


/*
    for (const startNumber in predictions) {
        const horses = predictions[startNumber];
        const totalProb = Object.values(horses).reduce((sum, h) => sum + (h.winProbability || 0), 0);
        if (totalProb > 0) {
            for (const programNumber in horses) {
                horses[programNumber].winProbability = (horses[programNumber].winProbability / totalProb).toFixed(4);
            }
        }
    }

    console.log(`Generated predictions for ${Object.keys(predictions).length} races`);
    console.log('Final predictions:', JSON.stringify(predictions, null, 2));
    return predictions;
*/    
}

app.post('/predict/:date/:track', async (req, res) => {
    const { date, track } = req.params;
    if (!date || !track) {
        return res.status(400).json({ error: 'Missing date or track' });
    }

    try {
        const predictions = await fetchAndPredictRaces(date, track);
        if (Object.keys(predictions).length === 0) {
            return res.status(404).json({ error: 'No valid starts found for prediction' });
        }
        res.json(predictions);
    } catch (err) {
        console.error('Prediction error:', err.message);
        res.status(500).json({ error: 'Failed to process prediction', details: err.message });
    }
});

const PORT = 3001;
loadModelAndScaler().then(() => {
    app.listen(PORT, () => {
        console.log(`Server running on port ${PORT}`);
    });
});
