from common import *

# TensorBoard logs
class TensorBoard( tf.keras.callbacks.TensorBoard ):

    def __init__(
        self,
        path,
    ):
        super().__init__(
            log_dir = path,
            histogram_freq = 1,
        )
        self.path = path

# Early stopping
class EarlyStopping( tf.keras.callbacks.EarlyStopping ):

    def __init__(
        self,
        monitor,
        patience,
        restore_best_weights,
        start_from_epoch,
    ):
        super().__init__(
            # monitor='val_sparse_categorical_accuracy',
            monitor,
            patience = patience,
            # min_delta = 0.01, # defaults to 0.
            restore_best_weights = restore_best_weights,
            start_from_epoch = start_from_epoch,
            # mode = 'min', # should be chosen correctly based on monitor value
        )
        self._monitor = monitor
        self._patience = patience
        self._restore_best_weights = restore_best_weights
        self._start_from_epoch = start_from_epoch

# Model Checkpoints for saving best model weights
class ModelCheckpoint( tf.keras.callbacks.ModelCheckpoint ):

    def __init__(
        self,
        path,
        save_best_only = True,
        monitor = 'val_loss',
    ):
        super().__init__(
            os.path.join( path, 'best_model' ),
            save_best_only = save_best_only,
            monitor = monitor,
            verbose = 1,
            # mode = 'min', # should be chosen correctly based on monitor value
        )
        self.path = path
        self.save_best_only = save_best_only
        self.monitor = monitor

class TimerCallback( tf.keras.callbacks.Callback ):

    def __init__(
        self,
        metric_name = 'epoch_duration',
    ):
        super().__init__()
        self.__epoch_start = None
        self.__metric_name = metric_name

    def on_epoch_begin(
        self,
        epoch,
        logs = None,
    ):
        self.__epoch_start = datetime.datetime.utcnow()

    def on_epoch_end(
        self,
        epoch,
        logs,
    ):
        logs[ self.__metric_name ] = ( datetime.datetime.utcnow() - self.__epoch_start ) / datetime.timedelta( milliseconds = 1 )

