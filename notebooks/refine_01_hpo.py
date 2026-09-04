import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_absolute_error
import pickle, json, time

optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_parquet('../data/features.parquet')

with open('../data/split_config.pkl', 'rb') as f:
    cfg = pickle.load(f)
feature_cols = cfg['feature_cols']
cat_features = cfg['cat_features']
target_col = cfg['target_col']
SPLIT_WEEK = cfg['split_week']  # 135 -> gercek test = 136-145, HIC dokunulmayacak

# Sizinti onlemek icin: HPO icin train setinin kendi icinde ayri bir ic-validasyon kuruyoruz
# inner-train: hafta <= 125, inner-valid: hafta 126-135 (baseline'in test'iyle AYNI uzunlukta: 10 hafta)
INNER_SPLIT = SPLIT_WEEK - 10  # 125

full_train_df = df[df['week'] <= SPLIT_WEEK].copy()
real_test_df  = df[df['week'] > SPLIT_WEEK].copy()   # bu Optuna sirasinda HICBIR sekilde kullanilmiyor

inner_train_df = full_train_df[full_train_df['week'] <= INNER_SPLIT].copy()
inner_valid_df = full_train_df[full_train_df['week'] > INNER_SPLIT].copy()

print(f'inner_train: {inner_train_df.shape}  (hafta <= {INNER_SPLIT})')
print(f'inner_valid: {inner_valid_df.shape}  (hafta {INNER_SPLIT+1}-{SPLIT_WEEK})')
print(f'real_test  : {real_test_df.shape}  (hafta {SPLIT_WEEK+1}-145, HPO sirasinda GORULMEDI)')

lgb_inner_train = lgb.Dataset(inner_train_df[feature_cols], label=inner_train_df[target_col],
                               categorical_feature=cat_features)
lgb_inner_valid = lgb.Dataset(inner_valid_df[feature_cols], label=inner_valid_df[target_col],
                               categorical_feature=cat_features, reference=lgb_inner_train)


def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
        'seed': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.12, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 200),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 150),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 5.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 5.0, log=True),
        'feature_pre_filter': False,
    }
    model = lgb.train(
        params, lgb_inner_train, num_boost_round=3000,
        valid_sets=[lgb_inner_valid],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    return model.best_score['valid_0']['rmse']


t0 = time.time()
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=False)
print(f'HPO suresi: {time.time()-t0:.0f}s, en iyi ic-validasyon RMSE: {study.best_value:.5f}')
print('En iyi parametreler:', study.best_params)

# Baseline (default) parametrelerle ayni ic-validasyon karsilastirmasi (adil kiyas icin)
baseline_params = {
    'objective': 'regression', 'metric': 'rmse',
    'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 30,
    'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'bagging_freq': 1,
    'verbose': -1, 'seed': 42,
}
baseline_inner_model = lgb.train(
    baseline_params, lgb_inner_train, num_boost_round=3000,
    valid_sets=[lgb_inner_valid],
    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
)
print(f'Baseline parametrelerle ic-validasyon RMSE: {baseline_inner_model.best_score["valid_0"]["rmse"]:.5f}')

with open('../data/optuna_best_params.json', 'w') as f:
    json.dump({
        'best_params': study.best_params,
        'best_inner_valid_rmse': study.best_value,
        'baseline_inner_valid_rmse': baseline_inner_model.best_score['valid_0']['rmse'],
        'n_trials': len(study.trials),
        'inner_split_week': INNER_SPLIT,
    }, f, indent=2)
print('Kaydedildi: optuna_best_params.json')
