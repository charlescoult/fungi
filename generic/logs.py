import tensorflow as tf
import json
import os
import pandas as pd

class Log():

    hdf_key = "runs"

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        run_name: str,
        data_dir: str = "./log",
        hdf_filename: str = 'model.h5',
    ):
        self.model_data_dir = os.path.join( data_dir, dataset_name, model_name, run_name )
        self.model_df_filename =  os.path.join( data_dir, dataset_name, model_name, hdf_filename )
        self.df = self.load_metadata()

        if not os.path.exists( self.model_data_dir ):
            os.makedirs( self.model_data_dir )

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir = self.model_data_dir,
            histogram_freq = 1,
        )

        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.model_data_dir, 'best_model' ),
            save_best_only = True,
            monitor = 'val_loss',
            # mode = 'min', # should be chosen correctly based on monitor value
        )

        self.callbacks = [
            self.tensorboard_callback,
            self.model_checkpoint_callback,
        ]

    def save_classes(
        self,
        classes: list[str],
        filename: str = 'classes.json',
    ):
        with open( os.path.join( self.model_data_dir , 'classes.json' ), 'w') as out:
            json.dump(classes, out, indent=2)
        

    def load_metadata(self):
        if ( os.path.exists(self.model_df_filename) ):
            return pd.read_hdf( self.model_df_filename, self.hdf_key )
        else: return pd.DataFrame()

    def save_df(self):
        self.df.to_hdf(self.model_df_filename, self.hdf_key)

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
