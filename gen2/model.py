from common import *

# ## Model Building

# return a name that accurately describes the model building function or
# the tfhub model (by url) that was passed
def get_model_name( model_handle ):

    if callable(model_handle):
        return f'keras.applications/{model_handle.__name__}'
    else:
        split = model_handle.split('/')
        return f'tfhub/{split[-5]}.{split[-4]}.{split[-3]}'



# generate base_model layer
def gen_base_model_layer(
    name,
    source,
    input_dim,
    trainable = False,
):
    # If model_handle is a model building function, use that function
    if callable( source ):
        base_model = source(
            include_top = False,
            input_shape = ( input_dim, input_dim ) + (3,),
            weights = 'imagenet',
            # pooling = 'avg',
        )

    # otherwise build a layer from the tfhub url that was passed as a string
    else:
        base_model = hub.KerasLayer(
            source,
            input_shape = ( input_dim, input_dim ) + (3,),
            name = name,
        )

    base_model.trainable = trainable

    return base_model


# generate classifier
def gen_classifier_model_layer(
    num_classes,
    dropout,
    add_softmax = False,
):
    model = tf.keras.Sequential()
    model.add(
        layers.Dense(
            num_classes,
            # activation = 'softmax',
        )
    )

    model.add(
        layers.Dropout(dropout),
    )

    if ( add_softmax ):
        model.add(
            layers.Activation("softmax", dtype="float32"),
        )

    return model

