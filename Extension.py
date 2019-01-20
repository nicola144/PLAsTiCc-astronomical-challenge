
import sys, os
import argparse
import time
from datetime import datetime as dt
import gc
from functools import partial, wraps
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
from sklearn.model_selection import StratifiedKFold
from tsfresh.feature_extraction import extract_features
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from numba import jit
from utils import *
from hyperopt import STATUS_OK
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
np.warnings.filterwarnings('ignore')
#
# using ideas from https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135
# https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data
# https://www.kaggle.com/meaninglesslives/simple-neural-net-for-time-series-classification

def get_classifiers(param_list , full_train, y ,classes, class_weights):

    # Compute weights
    w = y.value_counts()
    weights = {i : np.sum(w) / w[i] for i in w.index}

    classifiers = []

    for i in range(len(param_list)):
        classifier = LGBMClassifier(**param_list[i])
        classifier.fit(full_train, y, sample_weight= y.map(weights))
        classifiers.append(classifier)

    return classifiers

def predict_chunk(df_, classifiers_, meta_, features, featurize_configs, train_mean, isonce):

    # process all features
    full_test = featurize(df_, meta_, featurize_configs['aggs'], featurize_configs['fcp'])
    full_test.fillna(0, inplace=True)

    if not(isonce):
        isonce = True
        features_val = list(features.values)
        augmented_feat = ['object_id'] + features_val
        full_test[augmented_feat].to_csv('./data/test_features.csv', header=True, index=False, float_format='%.6f')
    else:
        features_val = list(features.values)
        augmented_feat = ['object_id'] + features_val
        full_test[augmented_feat].to_csv('./data/test_features.csv', header=False, mode='a', index=False, float_format='%.6f')

    # Make predictions
    preds_ = None
    for classifier in classifiers_:
        if preds_ is None:
            preds_ = classifier.predict_proba(full_test[features])
        else:
            preds_ += classifier.predict_proba(full_test[features])

    preds_ = preds_ / len(classifiers_)

    # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds_.shape[0])
    for i in range(preds_.shape[1]):
        preds_99 *= (1 - preds_[:, i])

    # Create DataFrame from predictions
    preds_df_ = pd.DataFrame(preds_, columns=['class_{}'.format(s) for s in classifiers_[0].classes_])
    preds_df_['object_id'] = full_test['object_id']
    preds_df_['class_99'] = preds_99
    return preds_df_

def process_test(classifiers, features, featurize_configs,  train_mean, filename, chunks=5000000):

    meta_test = process_meta('./data/test_set_metadata.csv')
    # meta_test.set_index('object_id',inplace=True)

    remain_df = None
    isonce = False

    for i_c, df in enumerate(pd.read_csv('./data/test_set.csv', chunksize=chunks, iterator=True)):

        unique_ids = the_unique(df['object_id'].tolist())
        new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()

        if remain_df is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
        # Create remaining samples df
        remain_df = new_remain_df

        preds_df = predict_chunk(df_=df, classifiers_=classifiers, meta_=meta_test, features=features, featurize_configs=featurize_configs, train_mean=train_mean, isonce=isonce)

        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=False)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=False)

        del preds_df
        gc.collect()
        # print('{:15d} done in {:5.1f} minutes' .format(
        #         chunks * (i_c + 1), (time.time() - start) / 60), flush=True)

    # Compute last object in remain_df
    preds_df = predict_chunk(df_=remain_df, classifiers_=classifiers, meta_=meta_test, features=features, featurize_configs=featurize_configs, train_mean=train_mean, isonce=isonce)

    preds_df.to_csv(filename, header=False, mode='a', index=False)

    return

def objective_wrapper(full_train, y, classes, class_weights):

    def objective(params):

        params = {
            'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
            'n_estimators': int(params['n_estimators']),
            'num_leaves': int(params['num_leaves']),
            'cat_l2': '{:.3f}'.format(params['cat_l2']),
            'cat_smooth': int(params['cat_smooth']),
            'max_depth': int(params['max_depth'])
        }

        classifier = LGBMClassifier(
            learning_rate=0.001,
            **params
        )
        custom_metric = get_metric(classes, class_weights)
        print('do cross val')
        score = cross_val_score(classifier, full_train , y , scoring=custom_metric, cv=StratifiedKFold(n_splits=15, shuffle=True, random_state=1)).mean()
        print("Loss {:.3f} params {}".format(score, params))
        return score

    return objective

def main():

    # agg features
    aggs = {
        'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'detected': ['mean'],
        'flux_ratio_sq':['sum', 'skew'],
        'flux_by_flux_ratio_sq':['sum','skew'],
    }

    # tsfresh features
    fcp = {
        'flux': {
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,
            'mean_change': None,
            'mean_abs_change': None,
            'length': None,
        },

        'flux_by_flux_ratio_sq': {
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,
        },

        'flux_passband': {
            'fft_coefficient': [
                    {'coeff': 0, 'attr': 'abs'},
                    {'coeff': 1, 'attr': 'abs'}
                ],
            'kurtosis' : None,
            'skewness' : None,
        },

        'mjd': {
            'maximum': None,
            'minimum': None,
            'mean_change': None,
            'mean_abs_change': None,
        },
    }

    meta_train = process_meta('./data/training_set_metadata.csv')

    train = pd.read_csv('./data/training_set.csv')

    full_train = featurize(train, meta_train, aggs, fcp)

    if 'target' in full_train:
        y = full_train['target']
        del full_train['target']

    classes = sorted(y.unique())
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    class_weights = {c: 1 for c in classes}
    class_weights.update({c:2 for c in [64, 15]})

    if 'object_id' in full_train:
        oof_df = full_train[['object_id']]
        del full_train['object_id'], full_train['hostgal_specz'], full_train['ra'], full_train['decl'], full_train['gal_l'], full_train['gal_b'], full_train['ddf']

    train_mean = full_train.mean(axis=0)

    full_train.fillna(train_mean, inplace=True)

    param_list = []

    p1 = {
    'objective': 'multiclass',
    'num_class': 14,
    'n_jobs': -1,
    'max_depth': 7,
    'importance_type': 'gain',
    'boosting_type': 'dart',
    'n_jobs': -1,
    'n_estimators': 10000,
    'subsample_freq': 2,
    'subsample_for_bin': 300000,
    'min_data_per_group': 100,
    'max_cat_to_onehot': 4,
    'cat_l2': 0.01,
    'cat_smooth': 41,
    'max_cat_threshold': 32,
    'metric_freq': 10,
    'verbosity': -1,
    'metric': 'multi_logloss',
    'xgboost_dart_mode': True,
    'uniform_drop': False,
    'colsample_bytree': 0.5,
    'drop_rate': 0.173,
    'learning_rate': 0.001,
    'max_drop': 5,
    'min_child_samples': 10,
    'min_child_weight': 100.0,
    'min_split_gain': 0.1,
    'num_leaves': 10,
    'reg_alpha': 0.05,
    'reg_lambda': 0.00023,
    'skip_drop': 0.55,
    'subsample': 0.75
    }

    param_list.append(p1)

    p2 = {
    'objective': 'multiclass',
    'num_class': 14,
    'n_jobs': -1,
    'max_depth': 20,
    'importance_type': 'split',
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'n_estimators': 15000,
    'subsample_freq': 2,
    'subsample_for_bin': 300000,
    'min_data_per_group': 100,
    'max_cat_to_onehot': 4,
    'cat_l2': 0.015,
    'cat_smooth': 41,
    'max_cat_threshold': 27,
    'metric_freq': 10,
    'verbosity': -1,
    'metric': 'multi_logloss',
    'xgboost_dart_mode': False,
    'uniform_drop': False,
    'colsample_bytree': 0.5,
    'drop_rate': 0.173,
    'learning_rate': 0.001,
    'max_drop': 5,
    'min_child_samples': 10,
    'min_child_weight': 100.0,
    'min_split_gain': 0.1,
    'num_leaves': 10,
    'reg_alpha': 0.3,
    'reg_lambda': 0.001,
    'skip_drop': 0.55,
    'subsample': 0.8
    }

    param_list.append(p2)

    p3 = {
     'objective': 'multiclass',
    'num_class': 14,
    'n_jobs': -1,
    'max_depth': 7,
    'importance_type': 'split',
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'n_estimators': 8000,
    'subsample_freq': 2,
    'subsample_for_bin': 200000,
    'min_data_per_group': 100,
    'max_cat_to_onehot': 4,
    'cat_l2': 0.015,
    'cat_smooth': 41,
    'max_cat_threshold': 27,
    'metric_freq': 10,
    'verbosity': -1,
    'metric': 'multi_logloss',
    'xgboost_dart_mode': False,
    'uniform_drop': False,
    'colsample_bytree': 0.5,
    'drop_rate': 0.173,
    'learning_rate': 0.001,
    'max_drop': 5,
    'min_child_samples': 10,
    'min_child_weight': 100.0,
    'min_split_gain': 0.111,
    'num_leaves': 10,
    'reg_alpha': 0.3,
    'reg_lambda': 0.001,
    'skip_drop': 0.55,
    'subsample': 0.8
    }

    param_list.append(p3)

    p4 = {
    'objective': 'multiclass',
    'num_class': 14,
    'n_jobs': -1,
    'max_depth': 11,
    'importance_type': 'split',
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'n_estimators': 15000,
    'subsample_freq': 2,
    'subsample_for_bin': 200000,
    'min_data_per_group': 100,
    'max_cat_to_onehot': 4,
    'cat_l2': 0.015,
    'cat_smooth': 98,
    'max_cat_threshold': 27,
    'metric_freq': 10,
    'verbosity': -1,
    'metric': 'multi_logloss',
    'xgboost_dart_mode': False,
    'uniform_drop': False,
    'colsample_bytree': 0.5,
    'drop_rate': 0.173,
    'learning_rate': 0.001,
    'max_drop': 4,
    'min_child_samples': 10,
    'min_child_weight': 100.0,
    'min_split_gain': 0.111,
    'num_leaves': 10,
    'reg_alpha': 0.3,
    'reg_lambda': 0.001,
    'skip_drop': 0.55,
    'subsample': 0.9
    }

    param_list.append(p4)

    p5 = {
    'objective': 'multiclass',
    'num_class': 14,
    'n_jobs': -1,
    'max_depth': 11,
    'importance_type': 'split',
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'n_estimators': 10000,
    'subsample_freq': 2,
    'subsample_for_bin': 200000,
    'min_data_per_group': 100,
    'max_cat_to_onehot': 4,
    'cat_l2': 1.0,
    'cat_smooth': 59,
    'max_cat_threshold': 32,
    'metric_freq': 10,
    'verbosity': -1,
    'metric': 'multi_logloss',
    'xgboost_dart_mode': False,
    'uniform_drop': False,
    'colsample_bytree': 0.5,
    'drop_rate': 0.15,
    'learning_rate': 0.001,
    'max_drop': 4,
    'min_child_samples': 10,
    'min_child_weight': 100.0,
    'min_split_gain': 0.12,
    'num_leaves': 14,
    'reg_alpha': 0.1,
    'reg_lambda': 0.00555,
    'skip_drop': 0.55,
    'subsample': 1
    }

    param_list.append(p5)

    p6 = {
    'objective': 'multiclass',
    'num_class': 14,
    'n_jobs': -1,
    'max_depth': 15,
    'importance_type': 'split',
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'n_estimators': 7000,
    'subsample_freq': 2,
    'subsample_for_bin': 200000,
    'min_data_per_group': 100,
    'max_cat_to_onehot': 4,
    'cat_l2': 0.015,
    'cat_smooth': 98,
    'max_cat_threshold': 33,
    'metric_freq': 10,
    'verbosity': -1,
    'metric': 'multi_logloss',
    'xgboost_dart_mode': False,
    'uniform_drop': False,
    'colsample_bytree': 0.5,
    'drop_rate': 0.15,
    'learning_rate': 0.001,
    'max_drop': 4,
    'min_child_samples': 10,
    'min_child_weight': 100.0,
    'min_split_gain': 0.12,
    'num_leaves': 14,
    'reg_alpha': 0.15,
    'reg_lambda': 0.00555,
    'skip_drop': 0.55,
    'subsample': 1
    }

    param_list.append(p6)

    p7 = {
    'objective': 'multiclass',
    'num_class': 14,
    'n_jobs': -1,
    'max_depth': 5,
    'importance_type': 'split',
    'boosting_type': 'dart',
    'n_jobs': -1,
    'n_estimators': 1000,
    'subsample_freq': 2,
    'subsample_for_bin': 400000,
    'min_data_per_group': 300,
    'max_cat_to_onehot': 4,
    'cat_l2': 1.0,
    'cat_smooth': 40,
    'max_cat_threshold': 33,
    'metric_freq': 10,
    'verbosity': -1,
    'metric': 'multi_logloss',
    'xgboost_dart_mode': True,
    'uniform_drop': False,
    'colsample_bytree': 0.5,
    'drop_rate': 0.15,
    'learning_rate': 0.001,
    'max_drop': 4,
    'min_child_samples': 10,
    'min_child_weight': 100.0,
    'min_split_gain': 0.12,
    'num_leaves': 14,
    'reg_alpha': 0.25,
    'reg_lambda': 0.00555,
    'skip_drop': 0.55,
    'subsample':  0.75
    }

    param_list.append(p7)

    # -----------------------------------------------------------------------------------------------
    # HYPERPARAMETER OPTIMIZATION

    # space = {
    #     'n_estimators' : hp.quniform('n_estimators', 1000, 10000),
    #     'num_leaves': hp.quniform('num_leaves', 1, 15, 2),
    #     'cat_l2': hp.uniform('cat_l2', 0.1, 2.0),
    #     'cat_smooth': hp.uniform('cat_smooth', 10, 100),
    #     'max_depth' : hp.uniform('max_depth', 2, 25),
    #     'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    #     'max_cat_threshold': hp.uniform('colsample_bytree', 5, 60),
    #     'min_child_samples': hp.uniform('colsample_bytree', 5, 20),
    #     'min_child_weight': hp.uniform('colsample_bytree', 10, 300),
    #     'max_drop': hp.quniform('colsample_bytree', 1, 10),
    #     'reg_alpha': hp.uniform('colsample_bytree', 0.0001, 0.8),
    #     'reg_lambda': hp.uniform('colsample_bytree', 0.0001, 0.8)
    # }

    # callback = partial(objective, full_train, y)

    # best = fmin(fn=objective_wrapper(full_train, y, classes, class_weights), space=space,algo=tpe.suggest,max_evals=10)
    #
    # print("Hyperopt estimated optimum {}".format(best))
    # --------------------------------------------------------------------------------------------------------
    #
    # eval_func = partial(lgbm_modeling_cross_validation, full_train=full_train, y=y, classes=classes, class_weights=class_weights, nr_fold=15,random_state=1)
    #
    # # modeling from CV
    # classifiers, score = eval_func(best_params)
    # -----------------------------------------------------------------------------------------------------------
    classifiers = get_classifiers(param_list, full_train, y, classes, class_weights)

    # -----------------------------------------------
    filename = 'submission.csv'

    # TEST
    process_test(classifiers, features=full_train.columns, featurize_configs={'aggs': aggs, 'fcp': fcp},  train_mean=train_mean,filename=filename, chunks=5000000)
    z = pd.read_csv(filename)
    print("Shape BEFORE grouping: {}".format(z.shape))
    z = z.groupby('object_id').mean()
    z.to_csv('single_pred.csv', index=True)

if __name__ == '__main__':
    gc.enable()
    try:
        main()
    except Exception:
        raise
