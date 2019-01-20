
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils import to_categorical
from collections import Counter
import itertools
import pickle, gzip
import logging
import glob
from sklearn.preprocessing import StandardScaler
import warnings
import sys
import time
import tensorflow as tf
from sklearn.metrics import make_scorer
from tsfresh.feature_extraction import extract_features
from collections import deque

def haversine_plus(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) from
    #https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    #Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    #Implementing Haversine Formula:
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),
                          np.multiply(np.cos(lat1),
                                      np.multiply(np.cos(lat2),
                                                  np.power(np.sin(np.divide(dlon, 2)), 2))))

    haversine = np.multiply(2, np.arcsin(np.sqrt(a)))
    return {
        'haversine': haversine,
        'latlon1': np.subtract(np.multiply(lon1, lat1), np.multiply(lon2, lat2)),
   }

def process_meta(filename):
    meta_df = pd.read_csv(filename)

    meta_dict = dict()
    # distance
    meta_dict.update(haversine_plus(meta_df['ra'].values, meta_df['decl'].values,
                   meta_df['gal_l'].values, meta_df['gal_b'].values))
    #
    meta_dict['hostgal_photoz_certain'] = np.multiply(meta_df['hostgal_photoz'].values, np.exp(meta_df['hostgal_photoz_err'].values))

    meta_df = pd.concat([meta_df, pd.DataFrame(meta_dict, index=meta_df.index)], axis=1)
    return meta_df

def mwll_wrapper(classes, class_weights):
    def multi_weighted_logloss(y_true, y_preds):
        """
        refactor from
        @author olivier https://www.kaggle.com/ogrellier
        multi logloss for PLAsTiCC challenge
        """
        y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
        # Trasform y_true in dummies
        y_ohe = pd.get_dummies(y_true)
        # Normalize rows and limit y_preds to 1e-15, 1-1e-15
        y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
        # Transform to log
        y_p_log = np.log(y_p)
        # Get the log for ones, .values is used to drop the index of DataFrames
        # Exclude class 99 for now, since there is no class99 in the training set
        # we gave a special process for that class
        y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
        # Get the number of positives for each class
        nb_pos = y_ohe.sum(axis=0).values.astype(float)
        # Weight average and divide by the number of positives
        class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
        y_w = y_log_ones * class_arr / nb_pos

        loss = - np.sum(y_w) / np.sum(class_arr)
        return loss
    return multi_weighted_logloss

def lgbm_multi_weighted_logloss(y_true, y_preds):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weights = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}

    loss = multi_weighted_logloss(y_true, y_preds, classes, class_weights)
    return 'wloss', loss, False

# https://www.kaggle.com/gregbehm/plasticc-data-reader-get-objects-by-id
def get_objects_by_id(path, chunksize):
    """
    Generator that iterates over chunks of PLAsTiCC Astronomical Classification challenge
    data contained in the CVS file at path.

    Yields subsequent (object_id, pd.DataFrame) tuples, where each DataFrame contains
    all observations for the associated object_id.

    Inputs:
        path: CSV file path name
        chunksize: iteration chunk size in rows

    Output:
        Generator that yields (object_id, pd.DataFrame) tuples
    """

    # set initial state
    last_id = None
    last_df = pd.DataFrame()

    for df in pd.read_csv(path, chunksize=chunksize):

        # Group by object_id; store grouped dataframes into dict for fast access
        grouper = {
            object_id: pd.DataFrame(group)
            for object_id, group in df.groupby('object_id')
        }

        # queue unique object_ids, in order, for processing
        object_ids = df['object_id'].unique()
        queue = deque(object_ids)

        # if the object carried over from previous chunk matches
        # the first object in this chunk, stitch them together
        first_id = queue[0]
        if first_id == last_id:
            first_df = grouper[first_id]
            last_df = pd.concat([last_df, first_df])
            grouper[first_id] = last_df
        elif last_id is not None:
            # save last_df and return as first result
            grouper[last_id] = last_df
            queue.appendleft(last_id)

        # save last object in chunk
        last_id = queue[-1]
        last_df = grouper[last_id]

        # check for edge case with only one object_id in this chunk
        if first_id == last_id:
            # yield nothing for now...
            continue

        # yield all but last object, which may be incomplete in this chunk
        while len(queue) > 1:
            object_id = queue.popleft()
            object_df = grouper.pop(object_id)
            yield (object_id, object_df)

    # yield remaining object
    yield (last_id, last_df)

def loss_wrapper(wtable):
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
    def mywloss(y_true,y_pred):
        yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
        loss= -(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
        return loss

    return mywloss

def galactic_multi_weighted_logloss(y_true, y_preds):
    """
    Adapted from olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone

    classes = [6, 16, 53, 65, 92]
    class_weight = {6: 1, 16: 1, 53: 1, 65: 1, 92: 1}

    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

def extragalactic_multi_weighted_logloss(y_true, y_preds):
    """
    Adapted from olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone

    classes = [15, 42, 52, 62, 64, 67, 88, 90, 95]
    class_weight = { 15: 2,  42: 1, 52: 1, 62: 1, 64: 2, 67: 1, 88: 1, 90: 1, 95: 1}

    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

def get_aggregations():

    aggs = {
        'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'abs_magn': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'detected': ['mean'],
        'flux_ratio_sq':['sum', 'skew'],
        'flux_by_flux_ratio_sq':['sum','skew'],
        }

    return aggs

# adapted from https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135
#  Features from olivier's kernel: https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data
#  The mjd difference feature on detected==1 given here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
# fft features added to capture periodicity https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70346#415506
def featurize(df, df_meta, aggs, fcp, n_jobs=4):

    # getting better distmod
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    del df_meta['distmod']

    df_meta['distmod'] = df_meta['hostgal_photoz'].apply(lambda x: cosmo.distmod(x).value)

    df_meta.replace(-np.Inf, np.nan)

    df = add_abs_magn(df, df_meta)

    flux_ratio_sq = np.power(df['flux'].values / df['flux_err'].values, 2.0)

    df_flux = pd.DataFrame({'flux_ratio_sq': flux_ratio_sq,'flux_by_flux_ratio_sq': df['flux'].values * flux_ratio_sq,}, index=df.index)

    df = pd.concat([df, df_flux], axis=1)
    # calculating feature for each passband
    for i in range(6):
        flux_w_mean = df['flux_by_flux_ratio_sq_sum_'+str(i)].values / df['flux_ratio_sq_sum_'+str(i)].values
        flux_diff = df['flux_max_'+str(i)].values - df['flux_min_'+str(i)].values
        df_flux_agg = pd.DataFrame({'flux_w_mean_'+str(i): flux_w_mean,'flux_diff1_'+str(i): flux_diff,'flux_diff2_'+str(i): flux_diff / df['flux_mean_'+str(i)].values,'flux_diff3_'+str(i): flux_diff /flux_w_mean}, index=df.index)
        df =  pd.concat([df, df_flux_agg], axis=1)

    agg_df = df.groupby(['object_id', 'passband']).agg(aggs)

    agg_df.columns = [ '{}_{}'.format(k, agg) for k in aggs.keys() for agg in aggs[k]]

    agg_df.reset_index(level='object_id', inplace=True)
    agg_df.reset_index(level='passband', inplace=True)

    to_keep = [col for col in agg_df.columns.values]
    to_keep.remove("object_id")
    to_keep.remove("passband")

    # pivot
    agg_df = agg_df.pivot(index='object_id', columns='passband', values=to_keep)

    agg_df.columns = [ '{}_{}'.format(tuple[0], int(tuple[1])) for tuple in agg_df.columns.values]

    agg_df = process_flux_agg(agg_df) # new feature to play with tsfresh

    agg_df_ts_flux_passband = extract_features(df,column_id='object_id', column_sort='mjd', column_kind='passband', column_value='flux', default_fc_parameters=fcp['flux_passband'], n_jobs=n_jobs, disable_progressbar=True)

    agg_df_ts_flux = extract_features(df,  column_id='object_id',  column_value='flux', default_fc_parameters=fcp['flux'], n_jobs=n_jobs, disable_progressbar=True)

    agg_df_ts_flux_by_flux_ratio_sq = extract_features(df,  column_id='object_id',  column_value='flux_by_flux_ratio_sq',  default_fc_parameters=fcp['flux_by_flux_ratio_sq'], n_jobs=n_jobs, disable_progressbar=True)

    df_det = df[df['detected']==1].copy()
    agg_df_mjd = extract_features(df_det,column_id='object_id', column_value='mjd', default_fc_parameters=fcp['mjd'], n_jobs=n_jobs, disable_progressbar=True)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'].values - agg_df_mjd['mjd__minimum'].values
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']

    agg_df_ts_flux_passband.index.rename('object_id', inplace=True)
    agg_df_ts_flux.index.rename('object_id', inplace=True)
    agg_df_ts_flux_by_flux_ratio_sq.index.rename('object_id', inplace=True)
    agg_df_mjd.index.rename('object_id', inplace=True)
    agg_df_ts = pd.concat([agg_df, agg_df_ts_flux_passband, agg_df_ts_flux, agg_df_ts_flux_by_flux_ratio_sq, agg_df_mjd], axis=1).reset_index()
    result = agg_df_ts.merge(right=df_meta, how='left', on='object_id')

    return result

def add_features_to_agg(df, old_df):

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

    # see https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696
    # https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135

    # Add more features with
    agg_df_ts = extract_features(old_df, column_id='object_id', column_sort='mjd', column_kind='passband', column_value = 'flux', default_fc_parameters = fcp, n_jobs=4, disable_progressbar=True)
    df_det = old_df[old_df['detected']==1].copy()
    agg_df_mjd = extract_features(df_det, column_id='object_id', column_value = 'mjd', default_fc_parameters = {'maximum':None, 'minimum':None}, n_jobs=4, disable_progressbar=True)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'] - agg_df_mjd['mjd__minimum']
    agg_df_ts = pd.merge(agg_df_ts, agg_df_mjd, on = 'id')

    # tsfresh returns a dataframe with an index name='id'
    agg_df_ts.index.rename('object_id',inplace=True)

    df.reset_index(level='object_id', inplace=True)
    df.reset_index(level='passband', inplace=True)

    to_keep = [col for col in df.columns.values]
    to_keep.remove("object_id")
    to_keep.remove("passband")

    # pivot
    df = df.pivot(index='object_id', columns='passband', values=to_keep)

    df.columns = [str(col[0]) + "_" + str(col[1]) for col in df.columns]

    final_df = pd.merge(df, agg_df_ts, on='object_id')

    return final_df

def aggregate(data, aggs):
    df = data
    df = df.groupby(['object_id', 'passband']).agg(aggs)
    return df

def merge(data, right,mode):
    df = data
    df = data.reset_index().merge(
        right=right,
        how=mode,
        on='object_id'
    )
    return df

def get_metric(classes, class_weights):
    custom_metric = make_scorer(mwll_wrapper(classes,class_weights), greater_is_better=False, needs_proba=True)
    return custom_metric

def get_metric_gal():
    custom_metric = make_scorer(galactic_multi_weighted_logloss, greater_is_better=False, needs_proba=True)
    return custom_metric

def get_metric_extragal():
    custom_metric = make_scorer(extragalactic_multi_weighted_logloss, greater_is_better=False, needs_proba=True)
    return custom_metric

# calculate and add absolute magnitude feature
def add_abs_magn(train, meta):

    # start = time.time()

    original_train = train.as_matrix()

    original_cols = list(train.columns.values)

    train = pd.merge(train, meta, on='object_id')

    train = train[['object_id', 'flux', 'distmod']]

    train.fillna(0, inplace=True)

    data = train.as_matrix()

    absolute_magn_column = []

    for row in train.itertuples():

        row = np.array(row)

        flux = row[2] if row[2] > 0 else -row[2]

        distmod = row[3]

        abs_magn = np.log(flux) * (-2.5) - distmod if flux != 0 else 0

        absolute_magn_column.append(abs_magn)

    absolute_magn_column = np.array(absolute_magn_column)
    absolute_magn_column.reshape((absolute_magn_column.shape[0],1))
    result = np.column_stack(( original_train, absolute_magn_column ))

    result = pd.DataFrame(data=result, columns=original_cols+['abs_magn'])

    # finish = time.time()
    # print('Took ', finish - start)

    return result

# data preprocessing
def preprocess(df_, meta_, is_test):

    aggs = get_aggregations()

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

    merged = featurize(df_, meta_, aggs, fcp)

    merged.fillna(0, inplace=True)

    return merged

# create table of weight frequencies
def create_wtable(target):

    from sklearn.preprocessing import LabelEncoder
    label = LabelEncoder()
    unique_target = np.unique(target)
    label.fit_transform(unique_target)
    target_map = label.transform(target)
    target_categorical = to_categorical(target_map)
    target_count = Counter(target_map)
    wtable = np.zeros((len(unique_target),))
    for i in range(len(unique_target)):
        wtable[i] = target_count[i]/target_map.shape[0]

    return wtable

# to get unique ids
def the_unique(x):
    return [x[i] for i in range(len(x)) if x[i] != x[i-1]]

# data augmentation
def augment_data(data):

    cols = list(data.columns.values)

    data_m = data.as_matrix()

    start = time.time()

    toadd_mat = []

    for row in data.itertuples():

        flux_err = row.flux_err

        # new_flux = row.flux + np.random.normal(loc=0.0, scale= 2.0 * flux_err, size=None)

        new_flux = row.flux + np.random.uniform(low= -2.0 * flux_err, high= 2.0 * flux_err)

        new_row = np.array( [row.object_id, row.mjd, row.passband, new_flux, row.flux_err, row.detected] )

        toadd_mat.append(new_row)

    np.vstack([data_m, toadd_mat])

    data = pd.DataFrame(data=data_m, columns=cols)

    finish = time.time()

    print('Took', finish - start)

    data.to_csv('./data/training_set_aug.csv', header=False, mode='a', index=False, float_format='%.6f')

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
