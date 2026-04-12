import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Load data ──────────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

TARGET = 'rental_price'
ID_COL = 'id'

X = train.drop(columns=[ID_COL, TARGET])
y = train[TARGET]
X_test = test.drop(columns=[ID_COL])

# ── Feature engineering ────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    df['speed_kmh'] = df['distance_km'] / (df['trip_duration_min'] / 60 + 1e-6)
    df['price_per_km_est'] = df['avg_price_last_week'] / (df['distance_km'] + 1e-6)
    df['battery_distance_ratio'] = df['battery_level_start'] / (df['distance_km'] + 1e-6)
    df['demand_weekend'] = df['demand_index'] * df['is_weekend']
    return df

X      = add_features(X)
X_test = add_features(X_test)

# ── Column groups ──────────────────────────────────────────────────────────────
cat_cols = ['city_zone', 'scooter_model']
num_cols = [c for c in X.columns if c not in cat_cols]

# ── Preprocessor ──────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

X_proc      = preprocessor.fit_transform(X)
X_test_proc = preprocessor.transform(X_test)

# ── Optuna: XGBoost ────────────────────────────────────────────────────────────
def xgb_objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int('n_estimators', 300, 2000),
        learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        max_depth         = trial.suggest_int('max_depth', 3, 10),
        min_child_weight  = trial.suggest_int('min_child_weight', 1, 10),
        subsample         = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X_proc, y, cv=kf, scoring='r2', n_jobs=-1)
    return scores.mean()

print('Tuning XGBoost...')
xgb_study = optuna.create_study(direction='maximize')
xgb_study.optimize(xgb_objective, n_trials=100, show_progress_bar=True)
print(f'XGBoost best R² = {xgb_study.best_value:.4f}')
print(f'Best params: {xgb_study.best_params}\n')

# ── Optuna: LightGBM ───────────────────────────────────────────────────────────
def lgb_objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int('n_estimators', 300, 2000),
        learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        num_leaves        = trial.suggest_int('num_leaves', 20, 200),
        min_child_samples = trial.suggest_int('min_child_samples', 5, 50),
        subsample         = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model = lgb.LGBMRegressor(**params)
    scores = cross_val_score(model, X_proc, y, cv=kf, scoring='r2', n_jobs=-1)
    return scores.mean()

print('Tuning LightGBM...')
lgb_study = optuna.create_study(direction='maximize')
lgb_study.optimize(lgb_objective, n_trials=100, show_progress_bar=True)
print(f'LightGBM best R² = {lgb_study.best_value:.4f}')
print(f'Best params: {lgb_study.best_params}\n')

# ── Train best models on full data ─────────────────────────────────────────────
best_xgb = xgb.XGBRegressor(**{**xgb_study.best_params, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0})
best_lgb = lgb.LGBMRegressor(**{**lgb_study.best_params, 'random_state': 42, 'n_jobs': -1, 'verbose': -1})

best_xgb.fit(X_proc, y)
best_lgb.fit(X_proc, y)

xgb_pred  = best_xgb.predict(X_test_proc)
lgb_pred  = best_lgb.predict(X_test_proc)
final_pred = (xgb_pred + lgb_pred) / 2

# ── Submission ─────────────────────────────────────────────────────────────────
submission = pd.DataFrame({'id': test[ID_COL], 'rental_price': final_pred})
submission.to_csv('submission.csv', index=False)
print(f'Submission saved: {len(submission)} rows')
print(submission.head())
