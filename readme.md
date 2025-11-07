
## Solar power production prediction

The project predicts solar power generation with the use of weather forecasts trained on both forecasted and observed weather with the use of machine learning models, specifically LightGBM. The project includes:
- **Data Preprocessing**: Cleaning and feature engineering.
- **Exploratory Data Analysis (EDA)**: Visualizing trends and correlations.
- **Model Training**: Using LightGBM for regression tasks.
- **Hyperparameter Optimization**: Leveraging Optuna for tuning.
- **Kaggle Submission**: Generating predictions for competition.

### Key Directories
- `data/`: Contains raw, processed, and submission data.
- `notebooks/`: Jupyter notebooks for EDA, legacy experiments, and model training.
- `src/`: Core Python modules and scripts for data processing, modeling, and submission generation.
- `models/`: Stores trained models.

---

## Developer Workflows

### Environment Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the `data/` directory structure is in place:
   - `raw/`: Raw input data files.
   - `processed/`: Processed data files.
   - `submissions/`: Directory for storing submission files.

### Running Notebooks
- Use Jupyter to run notebooks in the `notebooks/` directory.
- Key notebooks:
  - `eda.ipynb`: For exploratory data analysis.
  - `model.ipynb`: For hyperparameter tuning.

### Model Training
- Train models using LightGBM in `notebooks/model.ipynb` or `src/model.py`.
- The best parameters from optuna studies can be loaded as is shown in 'notebooks/model.ipynb'

```python
base_params = {} #fill with parameters that were constant in the study

optuna_study = optuna.load_study(
    study_name="name", 
    storage="sqlite:///notebooks/lgbm_pv_optimization.db") #replace name


model_params = base_params | small_model_study.best_trial.params

```

### Hyperparameter Optimization
- Use Optuna for tuning LightGBM parameters.
- Optuna studies can be viewed and analysed with the optuna dashboard.
```bash
optuna-dashboard sqlite:///lgbm_pv_optimization.db
```

- Example workflow in `notebooks/model.ipynb`.

### Kaggle Submission
- Generate predictions using `src/kaggle_submission.py`:
  ```bash
  python src/kaggle_submission.py
  ```
- Submission files are saved in `data/submissions/`.

---

## Project-Specific Conventions

### Model Evaluation
- Use Mean Absolute Error (MAE) as the primary evaluation metric.
- Cross-validation is implemented using `KFold` from `sklearn`.

### File Naming
- Use descriptive names for notebooks and scripts (e.g., `eda.ipynb`, `model.py`).
- Keep submission files in `data/submissions/` with clear versioning (e.g., `submission_v1.csv`).

---

## Integration Points

### External Dependencies
- **LightGBM**: For regression modeling.
- **Optuna**: For hyperparameter optimization.
- **Pandas/NumPy**: For data manipulation.
- **Matplotlib/Seaborn**: For visualization.


