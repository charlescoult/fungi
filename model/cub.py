import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = '/mnt/cub/CUB_200_2011/images'


def get_cub_dataset(
    batch_size = 64,
    image_size = ( 299, 299 )
):
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        batch_size = batch_size,
        validataion_split = 0.2,
        subset = 'both',
    )



