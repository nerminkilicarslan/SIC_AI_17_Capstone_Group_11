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
        model.save_model(f'../data/quantile_final_p{int(a*100)}.txt')

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
    return result, (lo, mid, hi)


result, preds = train_quantile_set('kalibre edilmis final (alpha=.08/.5/.92, tuned params)', [0.08, 0.5, 0.92], TUNED_PARAMS)

with open('../data/quantile_final_calibrated.json', 'w') as f:
    json.dump(result, f, indent=2)

test_out = test_df.assign(pred_p_lo=preds[0], pred_p50=preds[1], pred_p_hi=preds[2])
test_out.to_parquet('../data/test_with_final_quantile_pred.parquet', index=False)
print('Kaydedildi.')
