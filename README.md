# House Price Prediction System

A production-ready machine learning pipeline for predicting house prices using Random Forest with hyperparameter tuning and cross-validation.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset in data/
# Download from Kaggle: data/House_price_prediction.csv

# 3. Train model
python train.py

# 4. Evaluate model
python evaluate.py

# 5. Explore data (optional)
jupyter notebook src/notebook/EDA.ipynb
```

**Expected time**: ~40 minutes for first run (training + evaluation)

---

## Features

✅ **IQR Outlier Removal**: Removes extreme price values using statistical method  
✅ **Hyperparameter Tuning**: GridSearchCV tests 32 combinations automatically  
✅ **5-Fold Cross-Validation**: Robust model evaluation across data splits  
✅ **Enhanced Diagnostics**: Detects overfitting, underfitting, model stability  
✅ **Production Ready**: Version control friendly Python scripts  
✅ **Reproducible**: All results saved (model, preprocessor, tuning details)  
✅ **Minimal Documentation**: Just README.md + RESULTS.md

## Dataset

The dataset used is `House_price_prediction.csv`. Place it in the `data/` directory at the project root:
```
data/
├── House_price_prediction.csv
└── .gitkeep
```

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets). It contains features like location, property type, area, bedrooms, baths, etc., with the target variable being `price`.

## Project Structure

```
House_price_prediction_system/
├── train.py                      # Training entry point
├── evaluate.py                   # Evaluation entry point
├── README.md                     # Getting started guide
├── RESULTS.md                    # Example outputs & results
├── requirements.txt              # Dependencies
├── .gitignore                    # Git ignore rules
├── data/
│   ├── House_price_prediction.csv
│   └── .gitkeep
├── models/
│   ├── random_forest.pkl         # Trained model
│   ├── preprocessor.pkl          # Data preprocessor
│   ├── tuning_results.pkl        # Hyperparameter results
│   └── .gitkeep
└── src/
    ├── train.py                  # Training implementation
    ├── evaluate.py               # Evaluation implementation
    ├── preprocess.py             # Data preprocessing
    └── notebook/
        └── EDA.ipynb             # Exploratory analysis
```

## Installation         

1. Clone repository:
   ```bash
   git clone <repository-url>
   cd House_price_prediction_system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare dataset:
   ```bash
   # Download from Kaggle and place in data/
   # File: data/House_price_prediction.csv
   ```

## Usage

### Train Model
```bash
python train.py
```

**Does:**
1. Load and preprocess data (IQR outlier removal)
2. Hyperparameter tuning (GridSearchCV, 32 combinations)
3. 5-fold cross-validation evaluation
4. Save best model, preprocessor, tuning results

**Saves:**
- `models/random_forest.pkl` - Best trained model
- `models/preprocessor.pkl` - Data preprocessor
- `models/tuning_results.pkl` - Tuning details

### Evaluate Model
```bash
python evaluate.py
```

**Does:**
1. Load trained model
2. Perform 5-fold cross-validation
3. Display metrics and diagnostics
4. Provide recommendations

**Output:** CV metrics (mean ± std), train/test metrics, model status

### Exploratory Data Analysis
```bash
jupyter notebook src/notebook/EDA.ipynb
```

**Contains:** Visualizations, statistical analysis, data exploration

## See Also

- **RESULTS.md** - Example outputs, metrics, before/after comparisons
- **src/notebook/EDA.ipynb** - Data exploration & visualization
- **src/preprocess.py** - Preprocessing details
- **src/train.py** - Training details
- **src/evaluate.py** - Evaluation details

## Troubleshooting

**Q: Training is slow?**  
A: Normal! GridSearchCV + 5-fold CV = 160 model trainings.

**Q: Modify hyperparameters?**  
A: Edit `PARAM_GRID` in `src/train.py`.

**Q: Interpret results?**  
A: See RESULTS.md for detailed explanations.

## Contributing

Open issues or submit pull requests.

## Credits

- **ChatGPT**: Initial guidance
- **GitHub Copilot**: Code generation

## License

Open-source. Use at your own risk.

---

**Status:** Production Ready ✅