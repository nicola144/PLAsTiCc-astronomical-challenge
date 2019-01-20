
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

import os
from os import environ
import datetime
from collections import deque

import keras
# Any results you write to the current directory are saved as output.
from sklearn.externals import joblib

from utils import *

warnings.simplefilter('ignore', FutureWarning)

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
tf.Session(config=config)


BATCH_SIZE = 100
EPOCHS = 200
LEARNING_RATE = 0.0001 # NOTE: added two zeros
DECAY_RATE = 0.0 #5e-6

DROPOUT = 0.25

BETA_1=0.9
BETA_2=0.999

from progress.bar import Bar
# prog_bar = Bar('Augmenting...', suffix='%(percent).1f%% - %(eta)ds - %(index)d / %(max)d', max=1421705 )

def train_randomforest(train, target, is_gal):

    forest = RandomForestClassifier(n_estimators=5000,max_depth=20,bootstrap=True, class_weight= 'balanced',random_state=1 )

    # param_grid = {
    #              'n_estimators': 10000, # a little expensive 
    #              'max_depth': np.logspace(2, 200, num= 10, base=10.0),
    #              'bootstrap': [True],
    #              'random_state': [1]
    #          }

    if(is_gal):

        custom_metric = get_metric_gal()
    else:
        custom_metric = get_metric_extragal()

    # forest = GridSearchCV(forest, param_grid, scoring=custom_metric, cv=3, n_jobs=4)

    forest.fit(train, target)

    # forest = forest.best_estimator_

    if(is_gal):
        joblib.dump(forest, './models/rf_gal.joblib')
    else:
        joblib.dump(forest, './models/rf_extra_gal.joblib')

    print('saved')

def eval_best_randomforest(train, target, is_gal):
    # get best forest
    if(is_gal):
        best_forest = joblib.load('./models/rf_gal.joblib')
        custom_metric = get_metric_gal()
    else:
        best_forest = joblib.load('./models/rf_extra_gal.joblib')
        custom_metric = get_metric_extragal()

    # print("cross validation score ")
    # print(np.average(cross_val_score(best_forest, train, target,scoring=custom_metric, cv=3)))

    return best_forest

def build_model(input_dim,output_dim, wtable):

    K.clear_session()
    # optimizer
    adam = optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=None, decay=DECAY_RATE, amsgrad=False)

    model = Sequential()

    activation = 'relu'

    initial_neurons = 256

    # kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01) # this seems too much regularization
    model.add(Dense(initial_neurons, input_dim=input_dim, activation=None, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))

    model.add(Dense(start_neurons//2,activation=None, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))

    model.add(Dense(start_neurons//4,activation=None, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))

    model.add(Dense(start_neurons//8,activation=None, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT/2))

    model.add(Dense(output_dim, activation='softmax'))

	# Compile model
    model.compile(loss=loss_wrapper(wtable), optimizer=adam, metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

    return model

def train_nn(train, target, wtable, is_gal):

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(target)
    encoded_target = encoder.transform(target)

    # convert integers to dummy variables (i.e. one hot encoded)
    target = np_utils.to_categorical(encoded_target)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # input dim is number of features, output dim is the number of classes
    input_dim = train.shape[1]

    if(is_gal):
        output_dim = 5
        filepath_to_save_models = "./models/model_gal.hdf5"
    else:
        output_dim = 9
        filepath_to_save_models = "./models/model_extragal.hdf5"

    # this is needed to pass the callback build function for the model to pass parameters to it
    callback = partial(build_model, input_dim, output_dim, wtable)

    # build classifier
    classifier = KerasClassifier(build_fn=callback, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # callbacks
    tensorboard = TensorBoard(log_dir='./log1', histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True, \
    write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, \
    embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
    model_checkpoint = ModelCheckpoint(filepath_to_save_models, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    # train nn
    history = classifier.fit(train,target, callbacks=[tensorboard,model_checkpoint,reduce_lr_on_plateau])

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

    # can also evaluate via scikit learn crossvalidation
    # kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

    return classifier
# load networks
def load_nn(wtable_gal,wtable_extragal):

    classifier_gal = load_model("./models/model_gal.hdf5", custom_objects={'mywloss': loss_wrapper(wtable_gal)})
    classifier_extragal = load_model("./models/model_extragal.hdf5", custom_objects={'mywloss': loss_wrapper(wtable_extragal)})

    global graph
    graph = tf.get_default_graph()

    return classifier_gal, classifier_extragal

def predict_chunk(test_raw_, classifier_gal,classifier_extragal, meta_):
    # merge with meta
    merged_test = preprocess(test_raw_,meta_, is_test=True)

    # need to drop any features that was not used for training
    del merged_test['ra'], merged_test['decl'], merged_test['gal_l'], merged_test['gal_b'], merged_test['hostgal_specz']

    gal_merged_test = merged_test[ merged_test["hostgal_photoz"] == 0.0000]

    extragal_merged_test = merged_test[ merged_test["hostgal_photoz"] != 0.0000]

    obj_ids_gal = gal_merged_test['object_id']
    obj_ids_extragal = extragal_merged_test['object_id']

    galactic_classes_nums = [6, 16, 53, 65, 92]
    extragalactic_classes_nums = [15, 42, 52, 62, 64, 67, 88, 90, 95]

    all_classes_nums = galactic_classes_nums + extragalactic_classes_nums
    all_classes_nums.sort()

    galactic_classes = ['class_' + str(s) for s in galactic_classes_nums ]
    extragalactic_classes = ['class_' + str(s) for s in extragalactic_classes_nums ]

    cols = ['object_id']+['class_'+ str(s) for s in all_classes_nums]+['class_99']

    del gal_merged_test['object_id'], extragal_merged_test['object_id']
    # if there are some galactic objects
    if(gal_merged_test.shape[0] != 0):

        preds_gal = classifier_gal.predict_proba(gal_merged_test)
        preds_extra_gal = classifier_extragal.predict_proba(extragal_merged_test)

        preds_df_gal = pd.DataFrame(data=preds_gal, columns=galactic_classes)
        preds_df_extra_gal = pd.DataFrame(data=preds_extra_gal, columns=extragalactic_classes)

        preds_df_gal['class_99'] = np.ones(preds_gal.shape[0])

        for i in range(preds_gal.shape[1]):
            preds_df_gal['class_99'] *= (1 - preds_gal[:, i])

        for class_ in extragalactic_classes:
            preds_df_gal[class_] = 0.0

        preds_df_extra_gal['class_99'] = np.ones(preds_extra_gal.shape[0])

        for i in range(preds_extra_gal.shape[1]):
            preds_df_extra_gal['class_99'] *= (1 - preds_extra_gal[:, i])

        for class_ in galactic_classes:
            preds_df_extra_gal[class_] = 0.0

        preds_df_gal['object_id'] = obj_ids_gal.values.astype(np.int32)
        preds_df_extra_gal['object_id'] = obj_ids_extragal.values.astype(np.int32)

        preds_df_gal = preds_df_gal.reindex(columns=cols)
        preds_df_extra_gal = preds_df_extra_gal.reindex(columns=cols)

        gal_values = np.matrix(preds_df_gal.values)
        extra_gal_values = np.matrix(preds_df_extra_gal.values)
        preds_df_data = np.vstack([gal_values, extra_gal_values])

        preds_df_ = pd.DataFrame(data=preds_df_data, columns=cols )

        del merged_test, gal_merged_test, extragal_merged_test
        gc.collect()

        return preds_df_

    else:
        # no galactic objects in this chunk, will only predict on extra ones
        preds_extra_gal = classifier_extragal.predict_proba(extragal_merged_test)

        preds_df_extra_gal = pd.DataFrame(data=preds_extra_gal, columns=extragalactic_classes)

        preds_df_extra_gal['class_99'] = np.ones(preds_extra_gal.shape[0])
        for i in range(preds_extra_gal.shape[1]):
            preds_df_extra_gal['class_99'] *= (1 - preds_extra_gal[:, i])

        for class_ in galactic_classes:
            preds_df_extra_gal[class_] = 0.0

        preds_df_extra_gal['object_id'] = obj_ids_extragal.values.astype(np.int32)

        preds_df_extra_gal = preds_df_extra_gal.reindex(columns=cols)

        del merged_test, gal_merged_test, extragal_merged_test
        gc.collect()

        return preds_df_extra_gal

def main():

    train = pd.read_csv('./data/training_set.csv')

    meta_train = pd.read_csv('./data/training_set_metadata.csv')

    # train = add_abs_magn(train, meta_train)

    merged_train = preprocess(train, meta_train, is_test=False)

    # galactic/extragalactic split
    galactic_merged_train = merged_train[merged_train["hostgal_photoz"] == 0.0000]
    extragalactic_merged_train = merged_train[merged_train["hostgal_photoz"] != 0.0000]

    # save target
    galactic_target = galactic_merged_train['target']
    extragalactic_target = extragalactic_merged_train['target']

    # create wtable
    wtable_gal = create_wtable(galactic_target)
    wtable_extragal = create_wtable(extragalactic_target)

    # delete things for training
    del galactic_merged_train['target'], galactic_merged_train['object_id'], galactic_merged_train['ra'], galactic_merged_train['decl'], galactic_merged_train['gal_l'], galactic_merged_train['gal_b'], galactic_merged_train['hostgal_specz']
    del extragalactic_merged_train['target'], extragalactic_merged_train['object_id'], extragalactic_merged_train['ra'], extragalactic_merged_train['decl'], extragalactic_merged_train['gal_l'], extragalactic_merged_train['gal_b'], extragalactic_merged_train['hostgal_specz']
    gc.collect()

    print(galactic_merged_train[np.isinf(galactic_merged_train)].stack().dropna())
    sys.exit()

    # standard scaling
    scaler_gal = StandardScaler()
    scaler_extragal = StandardScaler()
    galactic_merged_train_arr = scaler_gal.fit_transform(galactic_merged_train)
    extragalactic_merged_train_arr = scaler_extragal.fit_transform(extragalactic_merged_train)
    galactic_merged_train = pd.DataFrame(data=galactic_merged_train_arr, columns=galactic_merged_train.columns)
    extragalactic_merged_train = pd.DataFrame(data=extragalactic_merged_train_arr, columns=extragalactic_merged_train.columns)

    # save scalers
    joblib.dump(scaler_gal , 'scaler_gal.pkl')
    joblib.dump(scaler_extragal , 'scaler_extragal.pkl')

    if(int(sys.argv[1]) == 0):
        print("Using random forests")
        # train and evaluate
        if(int(sys.argv[2]) == 1):
            # train
            train_randomforest(galactic_merged_train, galactic_target, is_gal=True)
            print("trained first forest")
            train_randomforest(extragalactic_merged_train, extragalactic_target, is_gal=False)
            print("trained second forest")
        # load
        classifier_gal = eval_best_randomforest(galactic_merged_train, galactic_target, is_gal=True)
        classifier_extragal = eval_best_randomforest(extragalactic_merged_train, extragalactic_target, is_gal=False)

    elif(int(sys.argv[1]) == 1):
        print("Using neural networks")
        # neural network
        if(int(sys.argv[2]) == 1):
            classifier_gal = train_nn(galactic_merged_train, galactic_target, wtable_gal, is_gal=True)
            classifier_extragal = train_nn(extragalactic_merged_train, extragalactic_target, wtable_extragal, is_gal=False)

            print("You need to re run and just load the models after training.")
            sys.exit(0)

        else:
            print('Loading models')
            classifier_gal, classifier_extragal = load_nn(wtable_gal,wtable_extragal)

    else:
        print("must give an arg")
        sys.exit(0)

    # import meta test data
    meta_test = pd.read_csv('./data/test_set_metadata.csv')

    # Number of rows for each chunk
    start = time.time()
    chunks = 5000000
    remain_test_raw = None

    # These three additional lines are needed if we use neural networks

    # with graph.as_default():
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         # iterate through chunks
    for chunk_index, test_raw in enumerate(pd.read_csv('./data/test_set.csv', chunksize=chunks, iterator=True)):

        print("doing chunk: ", chunk_index)
        unique_ids = the_unique(test_raw['object_id'].tolist())
        new_remain_test_raw = test_raw.loc[test_raw['object_id'] == unique_ids[-1]].copy()

        if remain_test_raw is None:
            test_raw = test_raw.loc[test_raw['object_id'].isin(unique_ids[:-1])].copy()
        else:
            test_raw = pd.concat([remain_test_raw, test_raw.loc[test_raw['object_id'].isin(unique_ids[:-1])]], axis=0)

        # Create remaining samples df
        remain_test_raw = new_remain_test_raw

        # start_predict = time.time()

        preds_test_raw = predict_chunk(test_raw_=test_raw, classifier_gal=classifier_gal, classifier_extragal=classifier_extragal, meta_=meta_test)

        # finish_predict = time.time()
        # print("Took ", str(finish_predict - start_predict) , " to predict chunk")

        if chunk_index == 0:
            preds_test_raw.to_csv('./predictions/predictions_nn.csv', header=True, index=False, float_format='%.6f')
        else:
            preds_test_raw.to_csv('./predictions/predictions_nn.csv', header=False, mode='a', index=False, float_format='%.6f')

        del preds_test_raw
        gc.collect()

    # compute last object
    preds_test_raw = predict_chunk(test_raw_=remain_test_raw,  classifier_gal=classifier_gal, classifier_extragal=classifier_extragal, meta_=meta_test)
    # save predictions
    preds_test_raw.to_csv('./predictions/predictions_nn.csv', header=False, mode='a', index=False, float_format='%.6f')
    z = pd.read_csv('./predictions/predictions_nn.csv')
    z = z.groupby('object_id').mean()
    z.to_csv('./predictions/single_predictions_nn.csv', index=True, float_format='%.6f')

# main
if __name__ == '__main__':
    gc.enable()
    try:
        main()
    except Exception:
        raise
