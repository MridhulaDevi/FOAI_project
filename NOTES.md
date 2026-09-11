# SpotWise AI — Progress Notes

## Day 1 (done)
- Created project structure: data/, src/, src/utils/
- Set up virtual environment (venv) and installed: streamlit, pandas, numpy, plotly, scikit-learn
- Extracted Kaggle dataset zip (11 region CSVs) into data/
- Working with data/us-east-1.csv (3,721,999 rows)
- Fixed FileNotFoundError by running scripts from project root, not src/
- Created src/explore_data.py to verify dataset loads
- Created src/app.py (Streamlit shell) — confirmed app runs at localhost:8501
- Discovered CSV has NO header row — first row was being misread as column names
- Fixed by using header=None, names=[...] with correct column order:
  Timestamp, InstanceType, OperatingSystem, AvailabilityZone, SpotPrice

## Day 2 (done)
- Created src/clean_data.py — load_and_clean() function
- Applied header=None fix to load raw CSV correctly
- Converted Timestamp to datetime, SpotPrice to numeric
- Dropped rows with parsing failures, removed duplicates
- Sorted data chronologically
- Added feature engineering: Hour, DayOfWeek extracted from Timestamp
- Verified output:
  - Cleaned shape: 3,261,000 rows x 7 columns
  - Date range: 2017-04-26 to 2017-05-08
  - 68 unique InstanceTypes, 5 AvailabilityZones, 3 OperatingSystems
  - SpotPrice range: $0.0041 - $192.26
  - No missing values after cleaning
- Updated app.py to import load_and_clean() instead of raw pd.read_csv()

## Day 3 (done)
- Created src/train_model.py
- Filtered dataset to 3 common instance types for faster iteration
- One-hot encoded categorical columns: InstanceType, AvailabilityZone, OperatingSystem
- Chronological 80/20 train/test split (NOT random shuffle — time-series data)
- Trained Linear Regression model to predict SpotPrice from:
  Hour, DayOfWeek, InstanceType (encoded), AvailabilityZone (encoded), OperatingSystem (encoded)
- Evaluated model: printed MAE and RMSE
- Calculated residual_std (prediction error spread) — needed later for
  estimating uncertainty in Expectiminimax chance nodes
- Saved model + feature columns + residual_std to src/utils/price_model.pkl
  using joblib

## Day 4 (next)
- Build src/search.py — Expectiminimax core algorithm
- Decision nodes = candidate bid choices
- Chance nodes = price outcome probabilities, built from:
  price_model.pkl predictions + residual_std (uncertainty band)
- Utility function = score based on cost, interruption risk, deadline satisfaction
- Pick bid with highest expected utility

## Day 5 (planned)
- Connect ML model output -> chance-node probabilities -> Expectiminimax search
- Test end-to-end: input -> prediction -> search -> recommended bid

## Day 6 (planned)
- Build src/explain.py — forward-chaining rule engine
- IF-THEN rules over volatility, deadline slack, budget headroom, risk profile
- Generate plain-language explanation for recommended bid

## Day 7 (planned)
- Build dashboard input form in app.py
- Inputs: cloud provider, instance type, runtime, budget, deadline, risk profile

## Day 8 (planned)
- Wire dashboard inputs -> backend pipeline (ML + search + explain) -> outputs
- Display: recommended bid, cost, completion probability, savings, risk, explanation

## Day 9 (planned)
- Add Plotly charts: historical price trend + predicted trend overlay
- Strategy comparison table (recommended vs. fixed-bid baselines)

## Day 10 (planned)
- "What-if" counterfactual comparison panel

## Day 11 (planned)
- Testing, edge cases, styling, final polish