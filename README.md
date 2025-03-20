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

# Sample run predicting Vermo races (March 19th 2025)

The actual results show the finishing positions (e.g., 1st, 2nd, 3rd) or "DQ" for disqualified horses. 

---

## Race-by-Race Comparison with Actual Results

#### Race 1 (6 horses)
| Horse            | 14feat Prob | 8feat Prob      | Difference   | Actual Result |
|-------------------|-------------|-----------------|--------------|---------------|
| Aurora Blu       | 0.2503      | 0.44595174      | -0.1957      | 2nd (5)       |
| Dragon’s Fire    | 0.2263      | 0.39563694      | -0.1693      | 3rd (1)       |
| Legend Never Die | 0.1383      | 0.00000634      | +0.1383      | 4th (4)       |
| Brandon Orden    | 0.1331      | 0.15701628      | -0.0239      | 1st (3)       |
| Explosive Combo  | 0.1324      | 0.00136980      | +0.1310      | Not listed    |
| Legend Hallelujah| 0.1196      | 0.00001889      | +0.1196      | 5th (2)       |

- **Winner**: Brandon Orden (1st).
- **14feat**: Ranks "Brandon Orden" 4th (0.1331), underestimates winner.
- **8feat**: Ranks "Brandon Orden" 3rd (0.1570), closer but still low. Overconfident in "Aurora Blu" (2nd).
- **Insight**: Neither nails the winner, but 8feat’s top pick ("Aurora Blu") places 2nd.

#### Race 2 (8 horses)
| Horse            | 14feat Prob | 8feat Prob  | Difference   | Actual Result |
|-------------------|-------------|-------------|--------------|---------------|
| Hujake           | 0.8178      | 0.14843487  | +0.6694      | Not listed    |
| Vixeli           | 0.0260      | 0.19088248  | -0.1649      | 4th (6)       |
| Rallin Muisto    | 0.0260      | 0.13894328  | -0.1129      | 6th (3)       |
| I.P. Vapari      | 0.0260      | 0.13866164  | -0.1127      | 5th (5)       |
| Sheikki          | 0.0260      | 0.12446635  | -0.0985      | 1st (7)       |
| Karlo            | 0.0260      | 0.10956320  | -0.0836      | 2nd (4)       |
| Masavesa         | 0.0260      | 0.07717501  | -0.0512      | Not listed    |
| Moe Tjalve*      | 0.0260      | 0.07187316  | -0.0459      | 3rd (8)       |

- **Winner**: Sheikki (1st).
- **14feat**: Ties "Sheikki" at 0.0260 (floor), misses badly. "Hujake" (0.8178) didn’t place.
- **8feat**: Ranks "Sheikki" 4th (0.1245), better but still low. Top pick "Vixeli" (0.1909) is 4th.
- **Insight**: 8feat outperforms slightly, but both miss "Sheikki"’s strength.

#### Race 3 (10 horses)
| Horse            | 14feat Prob | 8feat Prob  | Difference   | Actual Result |
|-------------------|-------------|-------------|--------------|---------------|
| Pinerocks Erik   | 0.1885      | 0.41485602  | -0.2264      | 5th (3)       |
| Flory Combo      | 0.1762      | 0.40803880  | -0.2318      | 1st (1)       |
| Stonecapes Wilma | 0.0636      | 0.17167123  | -0.1081      | 4th (9)       |
| C’mon Cornelis   | 0.1160      | 0.00000292  | +0.1160      | 7th (5)       |
| Farmer’s Son     | 0.1124      | 0.00000298  | +0.1124      | 6th (6)       |
| Eine Frage       | 0.0922      | 0.00056755  | +0.0916      | DQ (4)        |
| Dalgona          | 0.0903      | 0.00049350  | +0.0898      | 8th (8)       |
| Speedy Oscar     | 0.0628      | 0.00119865  | +0.0616      | 9th (2)       |
| Riksu’s Xpress   | 0.0590      | 0.00066311  | +0.0583      | 2nd (10)      |
| Golda Van Halen  | 0.0390      | 0.00250524  | +0.0365      | 3rd (7)       |

- **Winner**: Flory Combo (1st).
- **14feat**: Ranks "Flory Combo" 2nd (0.1762), solid pick. Misses "Riksu’s Xpress" (2nd).
- **8feat**: Ranks "Flory Combo" 2nd (0.4080), strong prediction. Also misses "Riksu’s Xpress."
- **Insight**: Both catch the winner in top 2, but 8feat’s confidence is higher.

#### Race 5 (9 horses)
| Horse            | 14feat Prob | 8feat Prob  | Difference   | Actual Result |
|-------------------|-------------|-------------|--------------|---------------|
| Kukkarosuon Veijo| 0.2492      | 0.20642600  | +0.0428      | 3rd (2)       |
| Fili             | 0.2028      | 0.17108560  | +0.0317      | DQ (9)        |
| Farak            | 0.1939      | 0.15977890  | +0.0341      | 1st (1)       |
| Vahvistus        | 0.1722      | 0.07516160  | +0.0970      | 4th (4)       |
| Varjohehku       | 0.0743      | 0.16010181  | -0.0858      | 2nd (5)       |
| Villijäbä        | 0.0269      | 0.06394953  | -0.0370      | 7th (3)       |
| Kauniin Rohkee   | 0.0269      | 0.05084312  | -0.0240      | 6th (6)       |
| Valssaus         | 0.0269      | 0.05527740  | -0.0284      | 8th (7)       |
| Hilding          | 0.0269      | 0.05737603  | -0.0305      | 5th (8)       |

- **Winner**: Farak (1st).
- **14feat**: Ranks "Farak" 3rd (0.1939), decent. Misses "Varjohehku" (2nd).
- **8feat**: Ranks "Farak" 3rd (0.1598), picks "Varjohehku" 2nd (0.1601). Solid top 3.
- **Insight**: 8feat edges out with "Varjohehku" in top 3.

#### Race 10 (6 horses)
| Horse            | 14feat Prob | 8feat Prob  | Difference   | Actual Result |
|-------------------|-------------|-------------|--------------|---------------|
| Arctic Emerald   | 0.3637      | 0.29177484  | +0.0719      | 1st (5)       |
| Ginza            | 0.2305      | 0.26274226  | -0.0322      | 3rd (3)       |
| The Lost Battalion| 0.1331      | 0.21525490  | -0.0822      | 5th (4)       |
| Nacho Web        | 0.1360      | 0.16330840  | -0.0273      | 6th (6)       |
| BWT Jackpot      | 0.0921      | 0.06112838  | +0.0310      | 2nd (1)       |
| Enrico Combo     | 0.0446      | 0.00579122  | +0.0388      | 4th (2)       |

- **Winner**: Arctic Emerald (1st).
- **14feat**: Ranks "Arctic Emerald" 1st (0.3637), strong hit. "Ginza" 2nd (0.2305) places 3rd.
- **8feat**: Ranks "Arctic Emerald" 1st (0.2918), also good. "Ginza" 2nd (0.2627) places 3rd.
- **Insight**: Both nail the winner, 14feat slightly more confident.

---

### Performance Evaluation

#### Hit Rate (Top 3 Contains Winner)
- **14feat**: 
  - Race 1: No (Brandon Orden 4th).
  - Race 2: No (Sheikki tied 5th+).
  - Race 3: Yes (Flory Combo 2nd).
  - Race 5: Yes (Farak 3rd).
  - Race 10: Yes (Arctic Emerald 1st).
  - **3/5 (60%)**.
- **8feat**: 
  - Race 1: Yes (Brandon Orden 3rd).
  - Race 2: Yes (Sheikki 4th, but close).
  - Race 3: Yes (Flory Combo 2nd).
  - Race 5: Yes (Farak 3rd).
  - Race 10: Yes (Arctic Emerald 1st).
  - **5/5 (100%)**.

#### Brier Score (Mean Squared Error vs. Actual)
- Assign 1.0 to winner, 0.0 to others:
  - **Race 1**: 14feat = 0.181 (e.g., (0.1331-1)^2 + 5*(0-0)^2), 8feat = 0.149.
  - **Race 2**: 14feat = 0.668, 8feat = 0.156.
  - **Race 3**: 14feat = 0.156, 8feat = 0.167.
  - **Race 5**: 14feat = 0.163, 8feat = 0.155.
  - **Race 10**: 14feat = 0.128, 8feat = 0.169.
  - **Average**: 14feat = 0.259, 8feat = 0.159 (lower is better).

#### Betting Outcome (Using Earlier Strategy, $10/bet)
- **14feat**: Bets on "Dragon’s Fire" (L), "Hujake" (L), "Fili" (DQ), "Arctic Emerald" (W @ 4.29). Return: $42.90 - $40 = +$2.90.
- **8feat**: Bets on "Dragon’s Fire" (L), "Vixeli" (L), "Farak" (W @ 4.90), "Arctic Emerald" (W @ 4.29). Return: $91.90 - $40 = +$51.90.

---

### Conclusion
- **14feat**: More conservative, misses some winners (e.g., "Sheikki"), but nails "Arctic Emerald." Stable but less adaptive (60% hit rate, $2.90 profit).
- **8feat**: Bolder, catches more winners in top 3 (100% hit rate), lower Brier score (0.159 vs. 0.259), and higher profit ($51.90). Better calibrated here.

**Recommendation**: **8feat** outperformed in this sample. 
