from common import *

class BaseModel:

    def __init__(
        self,
        source,
        input_dim,
        preprocessor,
    ):
        self.source = source
        self.input_dim = input_dim
        self.preprocessor = preprocessor

base_models = {
    'MobileNet_v2': BaseModel(
        source = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4',
        input_dim = 224,
        # https://www.tensorflow.org/hub/common_signatures/images#input
        # The inputs pixel values are scaled between -1 and 1, sample-wise.
        preprocessor = tf.keras.applications.mobilenet_v2.preprocess_input,
    ),

    'Inception_v3': BaseModel(
        source = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4',
        input_dim = 299,
        # The inputs pixel values are scaled between -1 and 1, sample-wise.
        preprocessor = tf.keras.applications.inception_v3.preprocess_input,
    ),

    'Inception_v3_iNaturalist': BaseModel(
        source = 'https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5',
        input_dim = 299,
        # The inputs pixel values are scaled between -1 and 1, sample-wise.
        preprocessor = tf.keras.applications.inception_v3.preprocess_input,
    ),

    'Xception': BaseModel(
        source = tf.keras.applications.Xception,
        input_dim = 299,
        # The inputs pixel values are scaled between -1 and 1, sample-wise.
        preprocessor = tf.keras.applications.xception.preprocess_input,
    ),

    'ResNet101': BaseModel(
        source = tf.keras.applications.resnet.ResNet101,
        input_dim = 224,
        preprocessor = tf.keras.applications.resnet50.preprocess_input,
    ),

    'ResNet50': BaseModel(
        source = tf.keras.applications.ResNet50,
        input_dim = 224,
        preprocessor = tf.keras.applications.resnet50.preprocess_input,
    ),

    'Inception_ResNet_v2': BaseModel(
        source = tf.keras.applications.InceptionResNetV2,
        input_dim = 299,
        preprocessor = tf.keras.applications.inception_resnet_v2.preprocess_input,
    ),

    'EfficientNet_v2': BaseModel(
        source = tf.keras.applications.efficientnet_v2.EfficientNetV2B0,
        input_dim = 224,
        # The preprocessing logic has been included in the EfficientNetV2
        # model implementation. Users are no longer required to call this
        # method to normalize the input data. This method does nothing and
        # only kept as a placeholder to align the API surface between old
        # and new version of model.
        preprocessor = tf.keras.applications.efficientnet_v2.preprocess_input,
    ),
}

