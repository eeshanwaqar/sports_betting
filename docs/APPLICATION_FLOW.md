# EPL Betting App - Complete Application Flow

## 🎯 Simplified Project Goal

Build a **Sports Betting Odds Prediction App** that:
1. Loads 20 years of EPL data
2. Trains ML models to predict match outcomes
3. Provides betting odds predictions
4. Exposes predictions via a simple API
5. Shows results in a web interface

---

## 📋 Simplified Directory Structure

```
Project/
├── data/
│   ├── raw/              # Original CSV files copied here
│   ├── processed/        # Cleaned, merged data
│   ├── features/         # Feature-engineered datasets
│   └── models/           # Trained models
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py           # Load CSV files
│   │   ├── cleaner.py          # Clean data
│   │   └── processor.py        # Process data
│   │
│   ├── features/
│   │   ├── team_stats.py       # Team statistics
│   │   ├── form.py             # Recent form features
│   │   └── h2h.py              # Head-to-head features
│   │
│   ├── models/
│   │   ├── trainer.py          # Train models
│   │   ├── predictor.py        # Make predictions
│   │   └── evaluator.py        # Evaluate performance
│   │
│   ├── api/
│   │   ├── app.py              # FastAPI application
│   │   └── routes.py           # API endpoints
│   │
│   └── utils/
│       ├── config.py           # Configuration
│       └── helpers.py          # Helper functions
│
├── web/
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── templates/
│       └── index.html
│
├── config.yaml           # Single config file
├── requirements.txt      # Dependencies
├── train.py             # Script to train models
├── predict.py           # Script to make predictions
└── run_api.py           # Script to start API
```

---

## 🔄 Complete Application Flow

### **PHASE 1: Data Pipeline** 📊

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Load Historical Data                           │
│  File: src/data/loader.py                               │
├─────────────────────────────────────────────────────────┤
│  Input:  archive/Datasets/*.csv (2000-2020)            │
│  Action: Read all CSV files, combine into one dataset   │
│  Output: data/raw/all_matches.csv                       │
│                                                          │
│  Code:                                                   │
│  - Read each season CSV                                  │
│  - Add 'season' column                                   │
│  - Concatenate all seasons                               │
│  - Save to data/raw/                                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2: Clean Data                                     │
│  File: src/data/cleaner.py                              │
├─────────────────────────────────────────────────────────┤
│  Input:  data/raw/all_matches.csv                       │
│  Action: Clean missing values, fix data types           │
│  Output: data/processed/clean_matches.csv               │
│                                                          │
│  Tasks:                                                  │
│  - Handle missing values                                 │
│  - Standardize team names                                │
│  - Convert dates to datetime                             │
│  - Remove duplicates                                     │
│  - Fix data types (int, float, str)                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 3: Feature Engineering                            │
│  Files: src/features/*.py                               │
├─────────────────────────────────────────────────────────┤
│  Input:  data/processed/clean_matches.csv               │
│  Action: Create ML features                             │
│  Output: data/features/model_ready.csv                  │
│                                                          │
│  Features to Create:                                     │
│  ✓ Team form (last 5 matches: W/D/L)                   │
│  ✓ Goals scored/conceded (rolling average)             │
│  ✓ Home/away performance                                │
│  ✓ Head-to-head history                                 │
│  ✓ League position                                      │
│  ✓ Days since last match                                │
└─────────────────────────────────────────────────────────┘
```

### **PHASE 2: Model Training** 🤖

```
┌─────────────────────────────────────────────────────────┐
│  STEP 4: Train ML Models                                │
│  File: train.py (uses src/models/trainer.py)            │
├─────────────────────────────────────────────────────────┤
│  Input:  data/features/model_ready.csv                  │
│  Action: Train and save models                          │
│  Output: data/models/*.pkl                              │
│                                                          │
│  Models to Train:                                        │
│                                                          │
│  1️⃣ Match Outcome Classifier                           │
│     Target: FTR (H/D/A)                                 │
│     Algorithm: Random Forest or XGBoost                  │
│     Output: Probability of Home/Draw/Away               │
│                                                          │
│  2️⃣ Goals Predictor (Optional)                         │
│     Target: FTHG, FTAG                                  │
│     Algorithm: Gradient Boosting                         │
│     Output: Expected goals for each team                 │
│                                                          │
│  Split:                                                  │
│  - Training: 2000-2017 (70%)                            │
│  - Validation: 2018 (15%)                               │
│  - Test: 2019-2020 (15%)                                │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 5: Evaluate Models                                │
│  File: src/models/evaluator.py                          │
├─────────────────────────────────────────────────────────┤
│  Metrics:                                                │
│  - Accuracy                                              │
│  - Precision/Recall per class (H/D/A)                   │
│  - F1-Score                                              │
│  - Confusion Matrix                                      │
│  - ROC-AUC                                               │
│                                                          │
│  Save:                                                   │
│  - Best model to data/models/best_model.pkl             │
│  - Metrics to data/models/metrics.json                  │
└─────────────────────────────────────────────────────────┘
```

### **PHASE 3: Prediction System** 🔮

```
┌─────────────────────────────────────────────────────────┐
│  STEP 6: Make Predictions                               │
│  File: predict.py (uses src/models/predictor.py)        │
├─────────────────────────────────────────────────────────┤
│  Input:  Match details (home_team, away_team)           │
│  Process:                                                │
│    1. Load trained model (data/models/best_model.pkl)   │
│    2. Extract features for the match                     │
│       - Get team recent form                             │
│       - Calculate statistics                             │
│       - Get H2H history                                  │
│    3. Run prediction                                     │
│    4. Convert probabilities to odds                      │
│                                                          │
│  Output:                                                 │
│  {                                                       │
│    "home_team": "Arsenal",                              │
│    "away_team": "Chelsea",                              │
│    "probabilities": {                                    │
│      "home_win": 0.45,                                  │
│      "draw": 0.30,                                      │
│      "away_win": 0.25                                   │
│    },                                                    │
│    "odds": {                                            │
│      "home_win": 2.22,                                  │
│      "draw": 3.33,                                      │
│      "away_win": 4.00                                   │
│    }                                                     │
│  }                                                       │
└─────────────────────────────────────────────────────────┘
```

### **PHASE 4: API Layer** 🌐

```
┌─────────────────────────────────────────────────────────┐
│  STEP 7: Create REST API                                │
│  File: run_api.py (uses src/api/app.py)                 │
├─────────────────────────────────────────────────────────┤
│  Framework: FastAPI                                      │
│  Port: 8000                                              │
│                                                          │
│  Endpoints:                                              │
│                                                          │
│  📍 GET /                                               │
│     Home page                                            │
│                                                          │
│  📍 POST /predict                                       │
│     Body: {"home_team": "X", "away_team": "Y"}         │
│     Returns: Predictions + Odds                          │
│                                                          │
│  📍 GET /teams                                          │
│     Returns: List of all teams                          │
│                                                          │
│  📍 GET /history?team=Arsenal                           │
│     Returns: Historical matches for team                 │
│                                                          │
│  Start with:                                             │
│  python run_api.py                                       │
│  or                                                      │
│  uvicorn src.api.app:app --reload                       │
└─────────────────────────────────────────────────────────┘
```

### **PHASE 5: Web Interface** 💻

```
┌─────────────────────────────────────────────────────────┐
│  STEP 8: Build Simple Web UI                            │
│  File: web/templates/index.html                          │
├─────────────────────────────────────────────────────────┤
│  Interface Components:                                   │
│                                                          │
│  ┌─────────────────────────────────────┐               │
│  │  EPL Match Odds Predictor            │               │
│  │─────────────────────────────────────│               │
│  │  Home Team:  [Dropdown ▼]           │               │
│  │  Away Team:  [Dropdown ▼]           │               │
│  │              [Predict Button]        │               │
│  │─────────────────────────────────────│               │
│  │  Results:                            │               │
│  │  🏠 Home Win: 45%  (Odds: 2.22)    │               │
│  │  🤝 Draw:     30%  (Odds: 3.33)    │               │
│  │  ✈️  Away Win: 25%  (Odds: 4.00)    │               │
│  └─────────────────────────────────────┘               │
│                                                          │
│  Technology:                                             │
│  - HTML + Bootstrap (styling)                           │
│  - JavaScript (AJAX calls to API)                       │
│  - Chart.js (visualizations)                            │
└─────────────────────────────────────────────────────────┘
```

---

## 🎬 Step-by-Step Execution Flow

### **1. Development Phase**

```bash
# Step 1: Explore data
jupyter notebook notebooks/01_data_exploration.ipynb

# Step 2: Clean data
python -c "from src.data import loader, cleaner; 
           data = loader.load_all_matches(); 
           clean = cleaner.clean_data(data); 
           clean.to_csv('data/processed/clean_matches.csv')"

# Step 3: Engineer features
python -c "from src.features import team_stats, form; 
           # Create features"

# Step 4: Train model
python train.py

# Step 5: Test prediction
python predict.py --home "Arsenal" --away "Chelsea"
```

### **2. Production Phase**

```bash
# Start API server
python run_api.py

# Access application
# Browser: http://localhost:8000
# API: http://localhost:8000/docs (Swagger UI)
```

---

## 🎯 Detailed Feature Engineering Flow

```
Input: Clean match data

For each match, calculate:

1. HOME TEAM FEATURES:
   ├── recent_form_5 (points from last 5 matches)
   ├── goals_scored_avg (last 10 matches)
   ├── goals_conceded_avg (last 10 matches)
   ├── home_win_rate (season)
   ├── shots_per_match_avg
   └── league_position

2. AWAY TEAM FEATURES:
   ├── recent_form_5
   ├── goals_scored_avg
   ├── goals_conceded_avg
   ├── away_win_rate (season)
   ├── shots_per_match_avg
   └── league_position

3. HEAD-TO-HEAD FEATURES:
   ├── h2h_home_wins (last 5 meetings)
   ├── h2h_away_wins (last 5 meetings)
   ├── h2h_draws (last 5 meetings)
   └── h2h_avg_goals

4. TEMPORAL FEATURES:
   ├── month
   ├── day_of_week
   └── days_since_last_match

Output: Feature matrix ready for ML
```

---

## 📊 End-to-End Example

```python
# Complete flow in one script

# 1. LOAD DATA
from src.data.loader import load_all_matches
df = load_all_matches('archive/Datasets/')

# 2. CLEAN DATA
from src.data.cleaner import clean_data
df_clean = clean_data(df)

# 3. CREATE FEATURES
from src.features.team_stats import create_team_features
from src.features.form import create_form_features
df_features = create_team_features(df_clean)
df_features = create_form_features(df_features)

# 4. TRAIN MODEL
from src.models.trainer import train_model
model, metrics = train_model(df_features, target='FTR')

# 5. MAKE PREDICTION
from src.models.predictor import predict_match
prediction = predict_match(
    model, 
    home_team='Arsenal', 
    away_team='Chelsea',
    data=df_clean
)

# 6. CONVERT TO ODDS
odds = {
    'home': 1 / prediction['home_win'],
    'draw': 1 / prediction['draw'],
    'away': 1 / prediction['away_win']
}

print(f"Match: Arsenal vs Chelsea")
print(f"Home Win: {prediction['home_win']:.1%} (Odds: {odds['home']:.2f})")
print(f"Draw:     {prediction['draw']:.1%} (Odds: {odds['draw']:.2f})")
print(f"Away Win: {prediction['away_win']:.1%} (Odds: {odds['away']:.2f})")
```

---

## 🚀 Quick Start Commands

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Explore data
jupyter notebook notebooks/01_data_exploration.ipynb

# 3. Process data and train model
python train.py

# 4. Make a prediction
python predict.py --home "Manchester United" --away "Liverpool"

# 5. Start web app
python run_api.py
# Visit: http://localhost:8000
```

---

## 📋 Implementation Checklist

### Week 1: Data Pipeline
- [ ] Create `src/data/loader.py` - Load CSV files
- [ ] Create `src/data/cleaner.py` - Clean data
- [ ] Create `notebooks/01_data_exploration.ipynb` - Explore data
- [ ] Create `notebooks/02_data_cleaning.ipynb` - Test cleaning

### Week 2: Features
- [ ] Create `src/features/team_stats.py` - Team statistics
- [ ] Create `src/features/form.py` - Form features
- [ ] Create `src/features/h2h.py` - Head-to-head features
- [ ] Create `notebooks/03_feature_engineering.ipynb` - Test features

### Week 3: Modeling
- [ ] Create `src/models/trainer.py` - Train models
- [ ] Create `src/models/predictor.py` - Make predictions
- [ ] Create `src/models/evaluator.py` - Evaluate models
- [ ] Create `notebooks/04_model_training.ipynb` - Experiment
- [ ] Create `train.py` - Training script
- [ ] Create `predict.py` - Prediction script

### Week 4: API
- [ ] Create `src/api/app.py` - FastAPI app
- [ ] Create `src/api/routes.py` - API routes
- [ ] Create `run_api.py` - Start server script
- [ ] Test API endpoints

### Week 5: Web Interface
- [ ] Create `web/templates/index.html` - Homepage
- [ ] Create `web/static/css/style.css` - Styling
- [ ] Create `web/static/js/app.js` - Frontend logic
- [ ] Connect UI to API

### Week 6: Polish
- [ ] Add error handling
- [ ] Improve UI/UX
- [ ] Add visualizations
- [ ] Write documentation
- [ ] Deploy (optional)

---

## 🎯 Key Decisions Made Simple

| Decision | Choice | Why |
|----------|--------|-----|
| Data Storage | CSV files | Simple, no database needed initially |
| ML Framework | Scikit-learn | Easy to use, sufficient for this task |
| Model Type | Random Forest / XGBoost | Good for tabular data, interpretable |
| API Framework | FastAPI | Modern, fast, auto-documentation |
| Frontend | HTML + Bootstrap | Simple, no React/Vue complexity |
| Deployment | Local first | Can deploy to Heroku/AWS later |

---

## 💡 This Flow is Better Because:

1. ✅ **Linear & Clear** - Each step leads to the next
2. ✅ **Notebook-First** - Experiment before coding
3. ✅ **Incremental** - Build and test each part
4. ✅ **Practical** - Focus on what's needed, not enterprise features
5. ✅ **Simple** - No over-engineering

---

## 🎉 Final Result

You'll have:
- 📊 A trained ML model predicting match outcomes
- 🌐 A REST API serving predictions
- 💻 A web interface to query predictions
- 📈 Betting odds for any EPL matchup

All in ~6 weeks of focused development! 🚀
