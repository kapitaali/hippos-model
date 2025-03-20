# hippos-model
Horse racing prediction model predicting Finnish TOTO race outcomes. Trained a couple of models, one with 8 features and other with 14 features. More to come.

# Setting up the env

```
git clone https://github.com/kapitaali/hippos-model
cd hippos-model
python3 -m venv ./venv
./venv/bin/pip install -r ./requirements.txt
cd horse-race-predictor
npm install
```

and you should have python + npm set up.

# Usage

## Gather data

First we scrape the API. You should edit the scraping.py to check which year you want to download:

    data_files = scrape_periods(2010, 2016, chunk_size='month')

Running `./venv/bin/python3 scraping.py` will create 12 .json files to ./horse-race-predictor/racedata for each year. When you have downloaded all years since 2010, you have all the data.

## Build the model

Model building scripts will first look into the ../../racedata dir to see if the full data is in one .json file `scraped_race_data_2018-2024.json`. If this does not exist, the script will create it. After it you can delete the individual monthly .json files if you want.

Running `./venv/bin/python3 train_14feat.py` will train the 14 feature model. Similarly `./venv/bin/python3 train_8feat.py` will train the 8 feature model. You can suppress debug output by giving the debug level with `-d`, `-d 50` will print only critical errors.

## Predict 

Running `./venv/bin/python3 predict_14feat.py <YYYY-MM-DD>` will use the 14 feature model to predict a race outcome for a given date. 

There is a node backend included in the `./horse-race-predictor` directory: 
```
cd horse-race-predictor
node backend_14feat.js
```

will run the 14 feature backend. And then you can do `curl -X POST http://localhost:3001/predict/<date>/<track>` eg. `curl -X POST http://localhost:3001/predict/2025-03-19/H`. Same with the 8 feature backend.

Be sure to run the model exporter if you don't have the ONNX file. Earlier versions didn't export it to ONNX but it should be ok now.

# Roadmap

* Add more command line switches to the scripts to remove file editing. 
* Rewrite backend so that you can select model with API endpoint.
* Do model crossvalidation, train the model with different parameters and pick the ones with highest overall prediction rate.
* Add more models: XGBoost, LightGBM, CatBoost, Logistic Regression, LSTM, GRU, Temporal Fusion Transformers (TFT), BERT, ViTs, Multi-Modal Neural Networks, DQN, PPO

