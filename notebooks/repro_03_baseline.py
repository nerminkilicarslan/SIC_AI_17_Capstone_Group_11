import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import pickle

df = pd.read_parquet('../data/features.parquet')

SPLIT_WEEK = df['week'].max() - 10  # 135

feature_cols = [
    'checkout_price', 'base_price', 'discount_ratio',
    'emailer_for_promotion', 'homepage_featured',
    'num_orders_lag_1', 'num_orders_lag_4', 'num_orders_roll_mean_4',
    'weekofyear', 'week_sin', 'week_cos',
    'center_type_enc', 'category_enc', 'cuisine_enc',
    'city_code', 'region_code', 'op_area',
    'center_id', 'meal_id',
]
cat_features = ['center_type_enc', 'category_enc', 'cuisine_enc', 'center_id', 'meal_id']
target_col = 'log_num_orders'

train_df = df[df['week'] <= SPLIT_WEEK].copy()
test_df = df[df['week'] > SPLIT_WEEK].copy()
print('train:', train_df.shape, 'test:', test_df.shape)

lgb_train = lgb.Dataset(train_df[feature_cols], label=train_df[target_col], categorical_feature=cat_features)
lgb_valid = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], categorical_feature=cat_features, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_data_in_leaf': 30,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'verbose': -1,
    'seed': 42,
}

baseline_model = lgb.train(
    params, lgb_train, num_boost_round=2000,
    valid_sets=[lgb_train, lgb_valid], valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)

pred_log = baseline_model.predict(test_df[feature_cols], num_iteration=baseline_model.best_iteration)
pred_orders = np.clip(np.expm1(pred_log), 0, None)
actual_orders = test_df['num_orders'].values

rmsle = np.sqrt(np.mean((np.log1p(pred_orders) - np.log1p(actual_orders)) ** 2))
mae = mean_absolute_error(actual_orders, pred_orders)

print(f'RMSLE: {rmsle:.4f}')
print(f'MAE  : {mae:.2f}')
print(f'Best iteration: {baseline_model.best_iteration}')

baseline_model.save_model('../data/baseline_lgb_model.txt')
with open('../data/split_config.pkl', 'wb') as f:
    pickle.dump({'feature_cols': feature_cols, 'cat_features': cat_features,
                 'target_col': target_col, 'split_week': SPLIT_WEEK}, f)
test_df = test_df.assign(pred_num_orders=pred_orders)
test_df.to_parquet('../data/test_with_baseline_pred.parquet', index=False)
print('Kaydedildi.')
