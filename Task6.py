
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import matplotlib.pyplot as plt
import seaborn as sns
# import lightgbm as lgb
# from catboost import Pool, CatBoostClassifier
import itertools
import pickle, gzip
import logging
import glob
from sklearn.preprocessing import StandardScaler
import warnings
import sys
import time
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Conv2D,BatchNormalization
from keras.layers import Dropout, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
from keras.regularizers import l1_l2

from functools import partial
import pickle
import multiprocessing
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input

import os
from os import environ
import datetime
from collections import deque

# Any results you write to the current directory are saved as output.
from sklearn.externals import joblib

from utils import *
from progress.bar import Bar

warnings.simplefilter('ignore', FutureWarning)

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
tf.Session(config=config)

BATCH_SIZE = 200
EPOCHS = 300
LEARNING_RATE = 0.01
DECAY_RATE = 0.0 #5e-6
MOMENTUM = 0.9
DROPOUT = 0.25

BETA_1=0.9
BETA_2=0.999

length_of_time_series = 100

def prepare_for_cnn(train, is_gal, is_test):

    features_for_cnn = ['flux', 'flux_err', 'detected']

    if not(is_test):
        # standardize
        flux_mean = train['flux'].mean()
        flux_err_mean = train['flux_err'].mean()
        detected_mean = train['detected'].mean()

        flux_std = train['flux'].std()
        flux_err_std = train['flux_err'].std()
        detected_std = train['detected'].std()

        train['flux'].apply(lambda x: (x - flux_mean) / flux_std)
        train['flux_err'].apply(lambda x: (x - flux_err_mean) / flux_err_std)
        train['detected'].apply(lambda x: (x - detected_mean) / detected_std)

        standardization_data = [flux_mean, flux_err_mean, detected_mean, flux_std, flux_err_std, detected_std]

        if(is_gal):
            pickle.dump(standardization_data, open('stand_data_gal.txt', 'wb'))
        else:
            pickle.dump(standardization_data, open('stand_data_extragal.txt', 'wb'))
    else:
        if(is_gal):
            standardization_data = pickle.load(open('stand_data_gal.txt', 'rb'))
        else:
            standardization_data = pickle.load(open('stand_data_extragal.txt', 'rb'))

        flux_mean = standardization_data[0]
        flux_err_mean = standardization_data[1]
        detected_mean = standardization_data[2]
        flux_std = standardization_data[3]
        flux_err_std = standardization_data[4]
        detected_std = standardization_data[5]

        train.iloc[:,0].apply(lambda x: (x - flux_mean) / flux_std)
        train.iloc[:,1].apply(lambda x: (x - flux_err_mean) / flux_err_std)
        train.iloc[:,2].apply(lambda x: (x - detected_mean) / detected_std)

    num_passbands = 6

    all_objects = dict()

    single_timeseries = np.zeros( (length_of_time_series, len(features_for_cnn),  6  ) )

    ids = np.array(train.object_id.unique(), dtype=np.int32)

    for obj_id in ids:

        df = train[train['object_id'] == obj_id]

        # list of 6 matrices , one for each passband
        matrices_for_this_object = dict()

        for passband in range(num_passbands):

            df_passband = df[ df['passband'] == passband ][features_for_cnn]

            if(df_passband.shape[0] > length_of_time_series ): #should not happen
                print('use longer timeseries')
                sys.exit()

            if(df_passband.shape[0] < length_of_time_series):

                # padding
                number_of_rows_to_append = length_of_time_series - df_passband.shape[0]

                data = [ np.zeros(len(features_for_cnn)) ] * number_of_rows_to_append

                mat = np.matrix(df_passband.values.tolist())

                for row in data:
                    mat = np.vstack([mat, row])

            matrices_for_this_object[passband] = mat
            # prog_bar.next()

        all_objects[obj_id] = matrices_for_this_object

    all_things = dict()

    for obj in ids:

        matrices_for_this_object = all_objects[obj]

        list_of_matrices = []

        for idx,matrix in matrices_for_this_object.items():
            matrix_as_list = matrix.tolist()
            list_of_matrices.append(matrix_as_list)

        one_thing = np.array(list_of_matrices)

        one_thing.reshape((one_thing.shape[1], one_thing.shape[2], one_thing.shape[0]))

        all_things[obj] = one_thing

    all_things = list(all_things.values())

    all_things = np.array(all_things)

    return all_things

# https://datascience.stackexchange.com/questions/20469/keras-visualizing-the-output-of-an-intermediate-layer
def layer_to_visualize(layer):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)

    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12,8))
    for i in range(len(convolutions)):
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(convolutions[i], cmap='gray')

def build_cnn( num_features, length_of_time_series, n_passbands, output_dim, wtable):

    K.clear_session()

    data_format="channels_last"

    adam = optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=None, decay=DECAY_RATE, amsgrad=False)

    model = Sequential()
    model.add(BatchNormalization())

    model.add(Conv2D(64, (1,3), strides=(1,3), data_format=data_format, padding='same', kernel_initializer='uniform', input_shape=( num_features, length_of_time_series,n_passbands ), use_bias=True ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(64, activation='relu', use_bias=True))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))

    model.add(Dense(16, activation='relu', use_bias=True))
    model.add(BatchNormalization())
    # model.add(Dropout(DROPOUT))

    model.add(Dense(output_dim, activation="softmax", use_bias=True))

    model.compile(loss=loss_wrapper(wtable), optimizer=adam,metrics=['accuracy'])

    return model

def train_cnn(train, target, wtable, is_gal):

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(target)
    encoded_target = encoder.transform(target)
    # convert integers to dummy variables (i.e. one hot encoded)
    target = np_utils.to_categorical(encoded_target)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    length_of_time_series = train.shape[2]
    num_features = train.shape[1]
    n_passbands = train.shape[3]

    if(is_gal):
        output_dim = 5
        filepath_to_save_models = './models/cnn/cnn_gal.hdf5'
    else:
        output_dim = 9
        filepath_to_save_models = './models/cnn/cnn_extragal.hdf5'

    callback = partial(build_cnn, num_features, length_of_time_series, n_passbands, output_dim, wtable)

    classifier = KerasClassifier(build_fn=callback, validation_split=0.3, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    # callbacks
    tensorboard = TensorBoard(log_dir='./log1', histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True, \
    write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, \
    embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
    model_checkpoint = ModelCheckpoint(filepath_to_save_models, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    # train nn
    history = classifier.fit(train,target, callbacks=[tensorboard,model_checkpoint,reduce_lr_on_plateau, early_stopping])

    print(np.average(history.history['acc']))
    print(np.average(history.history['val_acc']))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./figs/accuracy_at_"+str(datetime.datetime.now())+".png")
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./figs/loss_at_"+str(datetime.datetime.now())+".png" )
    plt.close()

    return classifier

def load_nn(wtable_gal,wtable_extragal):

    classifier_gal = load_model("./models/cnn/cnn_gal.hdf5", custom_objects={'mywloss': loss_wrapper(wtable_gal)})
    classifier_extragal = load_model("./models/cnn/cnn_extragal.hdf5", custom_objects={'mywloss': loss_wrapper(wtable_extragal)})

    global graph
    graph = tf.get_default_graph()

    return classifier_gal, classifier_extragal

def predict(classifier, currents, classes, other_classes, isfirst, ids):

    preds = classifier.predict_proba(currents)

    preds_df = pd.DataFrame(data=preds, columns=classes)
    preds_df['class_99'] = np.ones(preds.shape[0])

    for i in range(preds.shape[1]):
        preds_df['class_99'] *= (1 - preds[:, i])

    for class_ in other_classes:
        preds_df[class_] = 0.0

    preds_df['object_id'] = ids

    preds_df = preds_df.reindex(columns=cols)

    if isfirst:
        preds_df.to_csv('./predictions/predictions_cnn.csv', header=True, index=False, float_format='%.6f')
        isfirst = False
    else:
        preds_df.to_csv('./predictions/predictions_cnn.csv', header=False, mode='a', index=False, float_format='%.6f')

def main():

    train = pd.read_csv('./data/training_set.csv')

    meta_train = pd.read_csv('./data/training_set_metadata.csv')

    train = pd.merge(train, meta_train, on='object_id')

    galactic_merged_train = train[train["hostgal_photoz"] == 0.0000]
    # gal_mean = galactic_merged_train.mean(axis=0)
    galactic_merged_train.fillna(0, inplace=True)

    extragalactic_merged_train = train[train["hostgal_photoz"] != 0.0000]
    # galactic_merged_train.replace([np.inf, -np.inf], np.nan)
    # extragal_mean = extragalactic_merged_train.mean(axis=0)
    extragalactic_merged_train.fillna(0, axis=0)

    gal_ids = galactic_merged_train['object_id']
    extragal_ids = extragalactic_merged_train['object_id']

    original_galactic_target = galactic_merged_train[['object_id', 'target']]
    original_extragalactic_target = extragalactic_merged_train[['object_id', 'target']]
    galactic_target = original_galactic_target.drop_duplicates()
    extragalactic_target = original_extragalactic_target.drop_duplicates()
    galactic_target = galactic_target['target']
    extragalactic_target = extragalactic_target['target']

    assert np.all(np.isfinite(galactic_merged_train.values))
    assert not np.any(np.isnan(galactic_merged_train.values))
    assert np.all(np.isfinite(extragalactic_merged_train.values))
    assert not np.any(np.isnan(extragalactic_merged_train.values))

    # create wtable
    wtable_gal = create_wtable(galactic_target)
    wtable_extragal = create_wtable(extragalactic_target)

    if(int(sys.argv[1]) == 1):
        # train
        galactic_train = prepare_for_cnn(galactic_merged_train, is_gal=True, is_test=False)
        extragalactic_train = prepare_for_cnn(extragalactic_merged_train, is_gal=False, is_test=False)

        # number of samples,  number of cols, number of rows, number of channels
        galactic_train = galactic_train.reshape( galactic_train.shape[0], galactic_train.shape[3], galactic_train.shape[2], galactic_train.shape[1] )
        extragalactic_train = extragalactic_train.reshape( extragalactic_train.shape[0], extragalactic_train.shape[3], extragalactic_train.shape[2], extragalactic_train.shape[1] )

        classifier_gal = train_cnn(galactic_train, galactic_target, wtable_gal, is_gal=True)
        classifier_extragal = train_cnn(extragalactic_train, extragalactic_target, wtable_extragal, is_gal=False)

        print("You need to re run and just load the models after training.")
        sys.exit(0)

    elif(int(sys.argv[1]) == 0):
        # just load
        print('Loading models')
        classifier_gal, classifier_extragal = load_nn(wtable_gal,wtable_extragal)
    else:
        print("must give an arg")
        sys.exit(0)


    # import meta test data
    meta_test = pd.read_csv('./data/test_set_metadata.csv')

    # Number of rows for each chunk
    start = time.time()
    # chunks = 5000000
    remain_test_raw = None

    chunksize = 10000
    currents_gal = []
    currents_extragal = []
    gal_ids = []
    extragal_ids = []

    with graph.as_default():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            galactic_classes_nums = [6, 16, 53, 65, 92]
            extragalactic_classes_nums = [15, 42, 52, 62, 64, 67, 88, 90, 95]

            all_classes_nums = galactic_classes_nums + extragalactic_classes_nums
            all_classes_nums.sort()

            galactic_classes = ['class_' + str(s) for s in galactic_classes_nums ]
            extragalactic_classes = ['class_' + str(s) for s in extragalactic_classes_nums ]

            cols = ['object_id']+['class_'+ str(s) for s in all_classes_nums]+['class_99']

            isfirst = True

            for id, df in get_objects_by_id('./data/test_set.csv', chunksize=1_000_000):

                isgal = False

                df = pd.merge(df, meta_test, on='object_id')

                if(df['hostgal_photoz'][0] == 0.0000):
                    isgal = True
                    gal_ids.append(id)
                else:
                    extragal_ids.append(id)

                del df['ra'], df['decl'], df['gal_l'], df['gal_b'], df['hostgal_specz']

                mean = df.mean(axis=0)
                df.fillna(mean, inplace=True)

                prepared = prepare_for_cnn(df, is_gal=isgal, is_test=True)

                prepared = prepared.reshape(prepared.shape[0], prepared.shape[3], prepared.shape[2], prepared.shape[1])

                if(isgal):
                    currents_gal.append(prepared)
                else:
                    currents_extragal.append(prepared)

                if(isgal):
                    if(len(currents_gal) == chunksize):
                        # predict
                        predict(classifier_gal, currents_gal, galactic_classes, extragalactic_classes, isfirst, gal_ids)
                        currents_gal.clear()
                        gal_ids.clear()
                    else:
                        continue

                else:
                    # extragal
                    if(len(currents_extragal) == chunksize):
                        # predict
                        predict(classifier_extragal, currents_extragal, extragalactic_classes, galactic_classes, isfirst, extragal_ids)
                        currents_extragal.clear()
                        extragal_ids.clear()
                    else:
                        continue

            if(len(currents_extragal) > 0):
                # predict last object
                predict(classifier_extragal, currents_extragal, extragalactic_classes, galactic_classes, isfirst, extragal_ids )

            if(len(currents_gal) > 0):
                # predict last object
                predict(currents_gal, currents_gal, galactic_classes, extragalactic_classes, isfirst, gal_ids)

            z = pd.read_csv('./predictions/predictions_cnn.csv')
            z = z.groupby('object_id').mean()
            z.to_csv('./predictions/single_predictions_cnn.csv', index=True, float_format='%.6f')

# main
if __name__ == '__main__':
    gc.enable()
    try:
        main()
    except Exception:
        raise
