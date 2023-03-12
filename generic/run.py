import tensorflow as tf
import datetime
from datasets import *
from models import Model
from logs import Log
import os

class Run:

    def __init__(
        self,
        dataset_group,
        model: Model,
        log: Log,
        description,
        max_epochs = 20,
        learning_rate = 0.001,
        label_smoothing = 0,
    ):
        self.dataset_group = dataset_group
        self.model = model
        self.log = log
        self.max_epochs = max_epochs
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate

        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = learning_rate
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            # loss=tf.keras.losses.CategoricalCrossentropy(
            #     # from_logits=True,
            #     label_smoothing = self.label_smoothing,
            # ),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.SparseCategoricalCrossentropy(),
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="Top3"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=10, name="Top10"),
            ],
        )

    def start(
        self,
    ):
        history = self.model.fit(
            self.dataset_group.train_ds,
            validation_data = self.dataset_group.val_ds,
            epochs = self.max_epochs,
            callbacks = [
                # 'functional' callbacks
                # self.model.early_stopping_callback,
                *self.model.callbacks,
                # 'logging/static' callbacks
                *self.log.callbacks,
                # self.log.tensorboard_callback,
                # self.log.model_checkpoint_callback,
            ],
        )



if __name__ == '__main__':

    log_data_dir = '/media/data/models'

    max_epochs = 1
    base_model_meta = Model.base_models_meta[2]

    label_col = 'acceptedScientificName'

    # Dataset from hdf file
    data_dir = '/media/data/gbif'
    hdf_filename = 'clean_data.h5'
    hdf_path = os.path.join( data_dir, hdf_filename )
    hdf_key = 'media_merged_filtered-by-species_350pt'

    dataset_group = load_image_datasets_from_hdf(
        hdf_path,
        hdf_key,
        label_col,
        input_dim = base_model_meta[1],
        preprocessing = base_model_meta[2],
    )

    model = Model(
        base_model_meta,
        num_classes = len(dataset_group.classes),
        dropout = 0.33,
        freeze_base_model = False,
        data_augmentation = False,
        log = log,
    )

    log = Log(
        model_name = model.get_model_name(),
        dataset_name = 'gbif',
        run_name = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S'),
        data_dir = log_data_dir,
    )

    log.save_classes( dataset_group.classes )

    run = Run(
        dataset_group,
        model,
        log,
        "",
        max_epochs,
    )

    run.start()

    # dependency issues for saving to tfjs model
    # tfjs.converters.save_keras_model(model, os.path.join(run.log.model_data_dir, 'tfjs'))
