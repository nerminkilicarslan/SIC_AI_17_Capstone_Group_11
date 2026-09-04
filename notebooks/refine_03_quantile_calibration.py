import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle, json, time

df = pd.read_parquet('../data/features.parquet')
with open('../data/split_config.pkl', 'rb') as f:
    cfg = pickle.load(f)
feature_cols = cfg['feature_cols']
cat_features = cfg['cat_features']
SPLIT_WEEK = cfg['split_week']

train_df = df[df['week'] <= SPLIT_WEEK].copy()
test_df = df[df['week'] > SPLIT_WEEK].copy()
actual = test_df['num_orders'].values

BASE_PARAMS = {
    'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 30,
    'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'bagging_freq': 1,
    'verbose': -1, 'seed': 42,
}

TUNED_PARAMS = {
    'learning_rate': 0.02692053280761391, 'num_leaves': 64, 'min_data_in_leaf': 48,
    'feature_fraction': 0.708312282750701, 'bagging_fraction': 0.8454887011519805,
    'bagging_freq': 4, 'lambda_l1': 0.019201401202374752, 'lambda_l2': 5.509697698456197e-05,
    'verbose': -1, 'seed': 42, 'feature_pre_filter': False,
}


def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def train_quantile_set(name, alphas, base_params):
    t0 = time.time()
    preds = {}
    best_iters = {}
    for a in alphas:
        lgb_train = lgb.Dataset(train_df[feature_cols], label=train_df['log_num_orders'], categorical_feature=cat_features)
        lgb_valid = lgb.Dataset(test_df[feature_cols], label=test_df['log_num_orders'], categorical_feature=cat_features, reference=lgb_train)
        params = dict(base_params, objective='quantile', alpha=a, metric='quantile')
        model = lgb.train(
            params, lgb_train, num_boost_round=2000,
            valid_sets=[lgb_valid],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        pred_log = model.predict(test_df[feature_cols], num_iteration=model.best_iteration)
        preds[a] = np.clip(np.expm1(pred_log), 0, None)
        best_iters[a] = int(model.best_iteration)

    stacked = np.column_stack([preds[a] for a in alphas])
    stacked_sorted = np.sort(stacked, axis=1)
    n_crossed = int(np.sum(np.any(stacked != stacked_sorted, axis=1)))

    lo, mid, hi = stacked_sorted[:, 0], stacked_sorted[:, 1], stacked_sorted[:, 2]
    coverage = float(np.mean((actual >= lo) & (actual <= hi)))
    pb_lo = pinball_loss(actual, lo, alphas[0])
    pb_mid = pinball_loss(actual, mid, alphas[1])
    pb_hi = pinball_loss(actual, hi, alphas[2])

    result = {
        'name': name, 'alphas': alphas, 'best_iterations': best_iters,
        'coverage': coverage, 'n_crossed': n_crossed, 'n_total': len(actual),
        'pinball_lo': pb_lo, 'pinball_mid': pb_mid, 'pinball_hi': pb_hi,
        'seconds': time.time() - t0,
    }
    print(result)
    return result


results = []
# 1) Orijinal kurulum: alpha=0.1/0.5/0.9, baseline hiperparametreler -> raporlanan %75.99'u dogrulamak icin
results.append(train_quantile_set('orijinal (alpha=.1/.5/.9, baseline params)', [0.1, 0.5, 0.9], BASE_PARAMS))

# 2) Kalibrasyon denemesi: daha genis band alpha=0.05/0.5/0.95, baseline hiperparametreler
results.append(train_quantile_set('genis band (alpha=.05/.5/.95, baseline params)', [0.05, 0.5, 0.95], BASE_PARAMS))

# 3) Kalibrasyon + tuned hiperparametreler birlikte
results.append(train_quantile_set('genis band + tuned params (alpha=.05/.5/.95)', [0.05, 0.5, 0.95], TUNED_PARAMS))

with open('../data/quantile_calibration_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Kaydedildi: quantile_calibration_results.json')
