
from common import *


# ### Input Data Pipeline Generation

def load_image(
    filename,
):
    img_raw = tf.io.read_file( filename )
    img_tensor = tf.image.decode_image(
        img_raw,
        dtype = tf.dtypes.float32,
        channels = 3,
        expand_animations = False,
    )
    return img_tensor


def resize(
    img_tensor,
    input_dim,
):
    return tf.image.resize(
        img_tensor,
        [ input_dim, input_dim ],
    )


def preprocessing(
    img_tensor,
    preprocessor,
):
    return preprocessor( img_tensor )


def my_label_encoder( label, mapping ):
    return label == mapping
    # one_hot = label == mapping
    # label_encoded = tf.argmax( one_hot )
    # return label_encoded


def encode_label(
    label,
    label_encoder = my_label_encoder,
):
    return label_encoder( label )


def data_augmentation(
    img_tensor,
    augmentation_func,
):
    return augmentation_func( img_tensor, training = True )



# Augmentation function selection
augmentation_functions = [
    tf.keras.Sequential( [
        tf.keras.layers.RandomFlip( "horizontal_and_vertical" ),
        tf.keras.layers.RandomRotation( 0.2 ),
    ] )
]

'''
  # label encoding
    # (img_tensor_resized_preprocessed, label_encoded)
    label_encoder = tf.keras.layers.StringLookup(
        vocabulary = ds_classes,
        # sparse = True,
        output_mode = 'one_hot',
        num_oov_indices = 0,
    )
    label_vocab = label_encoder.get_vocabulary()
'''


def make_idp(
    filenames,
    labels,
    input_dim,
    label_encoder = None,
    is_training = False,
    batch_size = 32,
    augmentation_func = None,
    preprocessor = None,
):
    ds = tf.data.Dataset.from_tensor_slices( (
        filenames,
        labels,
    ) )

    # ds_classes = pd.Series(labels).unique().tolist()

    # generate and save the shuffle random seed
    shuffle_seed = tf.random.uniform(
        shape = (),
        dtype = tf.int64,
        minval = tf.int64.min,
        maxval = tf.int64.max,
    ).numpy()

    # make it json serializable...
    shuffle_seed = int( shuffle_seed )

    shuffle_buffer_size = int( len( ds ) )

    # if isTraining, shuffle
    if ( is_training ):
        ds = ds.shuffle(
            buffer_size = shuffle_buffer_size,
            seed = shuffle_seed,
        )

    # image loading
    # (img_tensor, label)
    ds = ds.map(
        lambda filename, label: (
            load_image(filename),
            label,
        ),
        num_parallel_calls = AUTOTUNE,
    )

    # if isTraining and augmentation_func exists, use data augmentation
    if ( is_training and augmentation_func ):
        logging.info("Adding data augmentation.")
        ds = ds.map(
            lambda img_tensor, label: (
                data_augmentation( img_tensor, augmentation_func ),
                label,
            ),
            num_parallel_calls = AUTOTUNE,
        )

    # image resizing
    # (img_tensor_resized, label)
    ds = ds.map(
        lambda img_tensor, label: (
            resize( img_tensor, input_dim ),
            label,
        ),
        num_parallel_calls = AUTOTUNE,
    )

    # image preprocessing
    # (img_tensor_resized_preprocessed, label)
    if ( preprocessor ):
        ds = ds.map(
            lambda img_tensor_resized, label: (
                preprocessing( img_tensor_resized, preprocessor ),
                label,
            ),
            num_parallel_calls = AUTOTUNE,
        )

    # Label encoding
    if ( label_encoder ):
        ds = ds.map(
            lambda img_tensor_resized_preprocessed, label: (
                img_tensor_resized_preprocessed,
                encode_label( label, label_encoder ),
                # encode_label( label, lambda x: my_label_encoder( x, ds_classes ) ),
            ),
            num_parallel_calls = AUTOTUNE,
        )

    # Batch
    ds = ds.batch( batch_size )

    # Prefetch
    ds = ds.prefetch( buffer_size = AUTOTUNE )

    return ds, shuffle_seed


