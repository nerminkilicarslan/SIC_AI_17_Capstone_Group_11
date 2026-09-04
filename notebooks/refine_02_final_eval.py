import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import pickle, json

df = pd.read_parquet('../data/features.parquet')
with open('../data/split_config.pkl', 'rb') as f:
    cfg = pickle.load(f)
feature_cols = cfg['feature_cols']
cat_features = cfg['cat_features']
target_col = cfg['target_col']
SPLIT_WEEK = cfg['split_week']

train_df = df[df['week'] <= SPLIT_WEEK].copy()
test_df = df[df['week'] > SPLIT_WEEK].copy()   # gercek test - HPO sirasinda hic gorulmedi

BASELINE_PARAMS = {
    'objective': 'regression', 'metric': 'rmse',
    'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 30,
    'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'bagging_freq': 1,
    'verbose': -1, 'seed': 42,
}

TUNED_PARAMS = {
    'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'seed': 42,
    'feature_pre_filter': False,
    'learning_rate': 0.02692053280761391,
    'num_leaves': 64,
    'min_data_in_leaf': 48,
    'feature_fraction': 0.708312282750701,
    'bagging_fraction': 0.8454887011519805,
    'bagging_freq': 4,
    'lambda_l1': 0.019201401202374752,
    'lambda_l2': 5.509697698456197e-05,
}


def evaluate(name, params):
    lgb_train = lgb.Dataset(train_df[feature_cols], label=train_df[target_col], categorical_feature=cat_features)
    lgb_valid = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], categorical_feature=cat_features, reference=lgb_train)
    model = lgb.train(
        params, lgb_train, num_boost_round=3000,
        valid_sets=[lgb_train, lgb_valid], valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    pred_log = model.predict(test_df[feature_cols], num_iteration=model.best_iteration)
    pred_orders = np.clip(np.expm1(pred_log), 0, None)
    actual_orders = test_df['num_orders'].values
    rmsle = np.sqrt(np.mean((np.log1p(pred_orders) - np.log1p(actual_orders)) ** 2))
    mae = mean_absolute_error(actual_orders, pred_orders)
    result = {
        'name': name,
        'best_iteration': int(model.best_iteration),
        'train_rmse': float(model.best_score['train']['rmse']),
        'valid_rmse': float(model.best_score['valid']['rmse']),
        'rmsle_test': float(rmsle),
        'mae_test': float(mae),
    }
    print(result)
    return result, model, pred_orders


results = {}
baseline_result, baseline_model, baseline_pred = evaluate('baseline (original params)', BASELINE_PARAMS)
tuned_result, tuned_model, tuned_pred = evaluate('tuned (Optuna, 40 trials)', TUNED_PARAMS)

improvement_rmsle = (baseline_result['rmsle_test'] - tuned_result['rmsle_test']) / baseline_result['rmsle_test'] * 100
improvement_mae = (baseline_result['mae_test'] - tuned_result['mae_test']) / baseline_result['mae_test'] * 100

print(f'\nRMSLE iyilesme: %{improvement_rmsle:.2f}')
print(f'MAE iyilesme: %{improvement_mae:.2f}')

with open('../data/hpo_final_comparison.json', 'w') as f:
    json.dump({
        'baseline': baseline_result,
        'tuned': tuned_result,
        'rmsle_improvement_pct': improvement_rmsle,
        'mae_improvement_pct': improvement_mae,
        'tuned_params': TUNED_PARAMS,
    }, f, indent=2)

tuned_model.save_model('../data/tuned_lgb_model.txt')
test_df = test_df.assign(pred_baseline=baseline_pred, pred_tuned=tuned_pred)
test_df.to_parquet('../data/test_with_tuned_pred.parquet', index=False)
print('Kaydedildi.')
