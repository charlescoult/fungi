import tensorflow as tf
import datetime
from datasets import Dataset
from models import Model
from logs import Log

class Run:


    def __init__(
        self,
        dataset: Dataset,
        model: Model,
        log: Log,
        max_epochs = 20,
        learning_rate = 0.001,
        label_smoothing = 0,
    ):
        self.dataset = dataset
        self.model = model
        self.log = log
        self.max_epochs = max_epochs
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate

        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            loss=tf.keras.losses.CategoricalCrossentropy(
                # from_logits=True,
                label_smoothing = self.label_smoothing,
            ),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.CategoricalCrossentropy(),
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="Top3"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=10, name="Top10"),
            ],
        )

    def start(
        self,
    ):
        history = self.model.fit(
            self.dataset.train,
            validation_data = self.dataset.val,
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

    max_epochs = 2
    base_model_meta = Model.base_models_meta[2]
    dataset_meta = Dataset.datasets_meta[1]

    dataset = Dataset(
        dataset_meta,
        image_size = base_model_meta[1],
        batch_size = 64,
        preprocess_input = base_model_meta[2],
        validation_split = 0.05,
        seed = 42,
    )

    print(dataset)

    model = Model(
        base_model_meta,
        num_classes = len(dataset.classes),
        dropout = 0.33,
        thawed_base_model_layers = -1,
        data_augmentation = False,
    )


    log = Log(
        model_name = model.get_model_name(),
        dataset_name = dataset.meta[0],
        run_name = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S_'),
        data_dir = log_data_dir,
    )


    run = Run(
        dataset,
        model,
        log,
        max_epochs,
    )

    run.start()


