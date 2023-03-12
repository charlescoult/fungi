import tensorflow as tf
import tensorflow_hub as hub
from logs import Log

class Model(tf.keras.Sequential):

    # Array of tuples describing the models to be tested
    # in the form: (model_handle, input_image_size, preprocessing_function)
    # where the model_handle is a model building function or a url to a tfhub feature model
    base_models_meta = [
        (
            'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4',
            224,
            # https://www.tensorflow.org/hub/common_signatures/images#input
            # The inputs pixel values are scaled between -1 and 1, sample-wise.
            tf.keras.applications.mobilenet_v2.preprocess_input,
        ),
        (
            'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4',
            299,
            # The inputs pixel values are scaled between -1 and 1, sample-wise.
            tf.keras.applications.inception_v3.preprocess_input,
        ),
        (
            'https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5',
            299,
            # The inputs pixel values are scaled between -1 and 1, sample-wise.
            tf.keras.applications.inception_v3.preprocess_input,
        ),
        (
            tf.keras.applications.Xception,
            299,
            # The inputs pixel values are scaled between -1 and 1, sample-wise.
            tf.keras.applications.xception.preprocess_input,
        ),
        (
            tf.keras.applications.resnet.ResNet101,
            224,
            tf.keras.applications.resnet50.preprocess_input,
        ),
        (
            tf.keras.applications.ResNet50,
            224,
            tf.keras.applications.resnet50.preprocess_input,
        ),
        (
            tf.keras.applications.InceptionResNetV2,
            299,
            tf.keras.applications.inception_resnet_v2.preprocess_input,
        ),
        (
            tf.keras.applications.efficientnet_v2.EfficientNetV2B0,
            224,
            # The preprocessing logic has been included in the EfficientNetV2
            # model implementation. Users are no longer required to call this
            # method to normalize the input data. This method does nothing and
            # only kept as a placeholder to align the API surface between old
            # and new version of model.
            tf.keras.applications.efficientnet_v2.preprocess_input,
        )
    ]

    def __init__(
        self,
        base_model_metadata,
        num_classes,
        dropout,
        freeze_base_model,
        log: Log,
        data_augmentation = False,
    ):
        super().__init__(name = "full_model")

        # Get base_model_information
        self.model_handle, self.input_dimension, self.preprocess_input = base_model_metadata
        print(self.input_dimension)

        # TODO: only really needed on training data...?
        #  - should be a function of the dataset, no?
        # if data_augmentation:
        #     self.add(self.build_data_augmentation())

        self.base_model = self.build_base_model(
            self.model_handle,
            input_shape=(self.input_dimension, self.input_dimension) + (3,),
            freeze_base_model = freeze_base_model,
        )
        self.add(self.base_model)

        self.classifier_model = self.build_classifier_model(
            num_classes,
            dropout,
        )
        self.add(self.classifier_model)

        self.early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            # monitor='val_sparse_categorical_accuracy',
            monitor = 'val_loss',
            patience = 5,
            min_delta = 0.01,
        )

        self.log = log

        self.callbacks = [
            self.early_stopping_callback,
            *self.log.callbacks,
        ]

    @staticmethod
    def build_data_augmentation():
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(
                "horizontal",
            ),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ], name = "data_augmentation")

        return data_augmentation

    @staticmethod
    def build_base_model(
        base_model_handle,
        input_shape,
        freeze_base_model = True,
        name="base_model",
    ):
        # If model_handle is a model building function, use that function
        if callable(base_model_handle):
            base_model = model_handle(
                include_top=False,
                input_shape=input_shape,
                weights='imagenet',
                pooling = 'avg',
            )

        # otherwise build a layer from the tfhub url that was passed as a string
        else:
            base_model = hub.KerasLayer(
                base_model_handle,
                input_shape=input_shape,
                name=name,
            )

        # Freeze specified # of layers
        # FullModel.freeze_base_model(base_model, thawed_base_model_layers)
        base_model.trainable = True


        # Print Base model weights
        print("\nBase Model:")
        #  Model.print_weight_counts(base_model)

        return base_model

    # Freeze base model?
    @staticmethod
    def freeze_base_model(
        base_model,
        freeze_base_model = True,
    ):
        print(base_model)
        print(base_model.summary())

        if thawed_base_model_layers == 0:
            base_model.trainable = False
        elif thawed_base_model_layers > 0:
            for layer in base_model.layers[(-1*thawed_base_model_layers):]:
                layer.trainable = False
        else:
            base_model.trainable = True

    @staticmethod
    def build_classifier_model(
        num_classes,
        dropout,
    ):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Dense(
                num_classes,
                # activation = 'softmax',
            )
        )

        model.add(
            tf.keras.layers.Dropout(dropout)
        )

        model.add(
            tf.keras.layers.Activation("softmax", dtype="float32")
        )

        return model

    # Print model weight counts
    @staticmethod
    def print_weight_counts(model):
        print(f'Non-trainable weights: {count_params(model.non_trainable_weights)}')
        print(f'Trainable weights: {count_params(model.trainable_weights)}')

    # returns a unique name that accurately describes the model building function or
    # the tfhub model (by url) that was passed
    def get_model_name( self ):

        if callable(self.model_handle):
            return f'keras.applications/{self.model_handle.__name__}'
        else:
            split = self.model_handle.split('/')
            return f'tfhub/{split[-5]}.{split[-4]}.{split[-3]}'