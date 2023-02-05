import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import datetime as dt
import timeit
import itertools as it
import os
# from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Activation, Flatten, Dropout, BatchNormalization


# Xcpetion Transfer learning Grid Search

class Dataset():

    def __init__(self, dataset_dir, batch_size, transformations = None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.load_dataset()

    def load_dataset(self):

        datagen = ImageDataGenerator(
            preprocessing_function = tf.keras.applications.xception.preprocess_input,
            validation_split=0.2,
        )

        self.train_gen = datagen.flow_from_directory(
            self.dataset_dir,
            target_size=(299,299),
            batch_size=self.batch_size,
            subset='training',
            shuffle=True,
        )

        self.validation_gen = datagen.flow_from_directory(
            self.dataset_dir,
            target_size=(299,299),
            batch_size=self.batch_size,
            subset='validation',
            shuffle=True,
        )

        if (transformations):


class Metadata():

    hdf_key="gs_metadata"

    def __init__(self, data_dir, metadata_hdf="gs_metadata.h5"):
        self.data_dir = data_dir
        self.filename =  os.path.join( data_dir, metadata_hdf )
        self.df = self.load_metadata()

    def load_metadata(self):
        if ( os.path.exists(self.filename) ):
            return pd.read_hdf( self.filename, self.hdf_key )
        else: return pd.DataFrame()

    def save_df(self):
        self.df.to_hdf(self.filename, self.hdf_key)

    def add_run(self, params, log_dir, time, scores):

        cols = {
            **params,
            'time': time,
            'log_dir': log_dir,
            'scores.loss': scores[0],
            'scores.accuracy': scores[1],
            'scores.top3': scores[2],
            'scores.top10': scores[3],
        }

        new_run = pd.DataFrame([cols])

        self.df = pd.concat([self.df, new_run])
        self.save_df()

class GridSearch():

    def __init__(self, data_dir="./gs_data"):
        self.data_dir = data_dir
        self.metadata = Metadata(data_dir)

    def set_dataset_dir(self, dataset_dir ):
        self.dataset_dir = dataset_dir

    def set_param_space(self, param_space ):
        self.param_space = param_space

    def get_permutation_count(self):
        if (self.param_space):
            total = 1
            for param, values in self.param_space.items():
                total *= len(values)
            return total
        else: return None

    def run(self):
        params, values = zip(*self.param_space.items())
        experiments = [ dict( zip(params, v)) for v in it.product(*values)]
        for exp_params in experiments:
            print()
            print(exp_params)
            print()
            time, tb_log_dir, scores = self.run_model(**exp_params)
            self.metadata.add_run( exp_params, tb_log_dir, time, scores )

    def run_model(
        self,
        batch_size,
        dropout,
        epochs,
        label_smoothing,
        train_full,
        learning_rate_lim,
    ):

        dataset = Dataset(self.dataset_dir, batch_size)

        model = tf.keras.models.Sequential()

        base_model = tf.keras.applications.Xception(
            include_top=False,
            pooling='avg',
            weights="imagenet"
        )
        if ( not train_full ):
            base_model.trainable = False

        model.add(base_model)

        model.add( tf.keras.layers.Dense(200) )
        model.add( tf.keras.layers.Dropout(dropout) )
        model.add( tf.keras.layers.Activation("softmax", dtype="float32") )

        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate_lim),
            # loss=tf.keras.losses.CategoricalCrossentropy(),
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing = label_smoothing, # forgot to add label smoothing in first run!!
            ),
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="Top3"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=10, name="Top10"),
            ],
        )

        tb_log_dir = os.path.join( self.data_dir, dt.datetime.now().strftime("%Y%m%d-%H%M%S") )

        time = timeit.timeit(
            lambda: model.fit(
                dataset.train_gen,
                validation_data=dataset.validation_gen,
                epochs=epochs,
                callbacks=[
                    tf.keras.callbacks.TensorBoard(
                        log_dir = tb_log_dir,
                        histogram_freq=1,
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        min_delta=0,
                        restore_best_weights=True,
                    ),
                ],
            ),
            number = 1,
        )

        model.save(os.path.join(tb_log_dir, 'final_model' ))

        scores = model.evaluate(dataset.validation_gen)

        return ( time, tb_log_dir, scores )

def run():

    gs = GridSearch()

    gs.set_dataset_dir("/mnt/cub/CUB_200_2011/images")

    '''
    param_space = {
        "learning_rate_lim": [
            0.001,
            0.0001,
        ],
        "batch_size": [
            1, 16, 64, 128,
        ],
        "dropout": [
            *np.linspace(0, 1, num=3)[:-1],
        ],
        "epochs": [
            30,
        ],
        "label_smoothing": [
            *np.linspace(0, 0.4, num = 5)
        ],
        "train_full": [
            # True,
            False,
        ],
    }
    '''
    # narrowed down param space based on results of first runs
    param_space = {
        'learning_rate_lim': [
            # 0.00015,
            0.0001,
            # 0.00005,
        ],
        'batch_size': [
            64, 128,
        ],
        'dropout': [
            *np.linspace(0, 0.5, num=4)[:-1],
        ],
        'epochs': [
            30,
        ],
        'label_smoothing': [
            *np.linspace(0, 0.3, num=3),
        ],
        'train_full': [
            False,
        ],
    }

    gs.set_param_space( param_space )

    gs.run()

if __name__ == '__main__':
    run()

