import tensorflow as tf
import os
import pandas as pd

class Log():

    hdf_key = "runs"

    def __init__(
        self,
        model_name: str,
        data_dir: str = "./logs",
        hdf_filename: str = "runs.h5",
    ):
        self.data_dir = os.path.join( data_dir, model_name )
        self.filename =  os.path.join( self.data_dir, hdf_filename )
        self.df = self.load_metadata()

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir = self.data_dir,
            histogram_freq = 1,
        )

        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.data_dir, 'best_model' ),
            save_best_only = True,
            monitor = 'val_loss',
            # mode = 'min', # should be chosen correctly based on monitor value
        )

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
