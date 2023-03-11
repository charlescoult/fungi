import pathlib
import tensorflow as tf
import pandas as pd

flowers_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
flowers_data_dir = tf.keras.utils.get_file('flower_photos', origin=flowers_dataset_url, untar=True)
flowers_data_dir = pathlib.Path(flowers_data_dir)

class Dataset:

    datasets_meta = [
        ('CUB-200-2011', '/media/data/cub/CUB_200_2011/images'),
        ('flowers', flowers_data_dir),
        (
            'GBIF_fungi',
            '/media/data/gbif/media',
        ),
    ]

    def __init__(
        self,
        meta,
        image_size = 244,
        batch_size = 64,
        preprocess_input = None,
        validation_split = 0.05,
        seed = 42,
    ):
        self.meta = meta
        self.image_size = image_size
        self.batch_size = batch_size
        self.preprocess_input = preprocess_input
        self.validation_split = validation_split
        self.seed = seed
        self.build_dataset()

    # build a dataset object from a directory of images 
    def build_dataset(
        self,
    ):
        labels = 'infer'

        self.train, self.val = tf.keras.utils.image_dataset_from_directory(
            self.meta[1],
            batch_size = self.batch_size,
            validation_split = self.validation_split,
            image_size = (self.image_size, self.image_size),
            subset = "both",
            shuffle = True, # default but here for clarity
            seed = self.seed,
            label_mode = "categorical", # enables one-hot encoding (use 'int' for sparse_categorical_crossentropy loss)
            # labels = labels, # need to default to 'infer', not None
        )

        # Retrieve class names
        # (can't do this after converting to PrefetchDataset?)
        self.classes = self.train.class_names
        # Prefetch images
        # train = train.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        # val = val.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.train = self.train.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val = self.val.prefetch(buffer_size=tf.data.AUTOTUNE)

        # apply preprocessing function
        self.train = self.train.map(
            lambda x, y: (self.preprocess_input(x), y),
            num_parallel_calls = 16,
        )
        self.val = self.val.map(
            lambda x, y: (self.preprocess_input(x), y),
            num_parallel_calls = 16,
        )


class CustomDataGen( tf.keras.utils.Sequence ):
    
    def __init__(
        self,
        df,
        X_col,
        y_col,
        batch_size,
        input_size=(224, 224, 3),
        shuffle=True,
        augment=False,
    ):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.augment = augment

        self.n = len(self.df)
    
    def __get_input(self, path, bbox, target_size):
    
        xmin, ymin, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

        return image_arr/255.
    
    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        path_batch = batches[self.X_col['path']]
        bbox_batch = batches[self.X_col['bbox']]
        
        name_batch = batches[self.y_col['name']]
        # type_batch = batches[self.y_col['type']]

        X_batch = np.asarray([self.__get_input(x, y, self.input_size) for x, y in zip(path_batch, bbox_batch)])

        y0_batch = np.asarray([self.__get_output(y, self.n_name) for y in name_batch])
        # y1_batch = np.asarray([self.__get_output(y, self.n_type) for y in type_batch])

        return X_batch, y0_batch
    
    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

import mimetypes as mt
import os
from sklearn.model_selection import train_test_split
from util import timeit

AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_image(
    filename,
    label,
):
    img_raw = tf.io.read_file( filename )
    img_tensor = tf.image.decode_image(
        img_raw,
        channels = 3,
        # Important to include the following
        # https://stackoverflow.com/questions/44942729/tensorflowvalueerror-images-contains-no-shape
        expand_animations = False,
    )
    return ( img_tensor, label )

def preprocessing_default(
    img_tensor,
    label,
):
    img_tensor_preprocessed = tf.image.resize( img_tensor, [ 299, 299 ] )
    img_tensor_preprocessed = img_tensor_preprocessed/255.0
    return ( img_tensor_preprocessed, label )


def load_image_dataset_from_df(
    df,
    batch_size,
    preprocessing,
):
    ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant( df[ 'filename' ].values ),
            tf.constant( df[ label_col ].values )
        ),
    )

    ds = ds.map(
        load_image,
        num_parallel_calls=AUTOTUNE,
    )

    ds = ds.map(
        preprocessing,
        # num_parallel_calls=AUTOTUNE,
    )

    ds = ds.batch( batch_size )

    ds = ds.prefetch( buffer_size = AUTOTUNE )

    # using cache only seems to negatively impact performance
    # ds.cache( filename = './cache.tf-data' )

    return ds

def load_image_datasets_from_hdf(
    hdf_path,
    hdf_key,
    label_col,
    test_size = 0.05,
    val_size = 0.1,
    preprocessing = preprocessing_default,
):

    df = pd.read_hdf( hdf_path, hdf_key )

    train_df, test_df = train_test_split(
        df,
        test_size = test_size,
        stratify = df[[ label_col ]],
        random_state = 42,
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size = val_size,
        stratify= train_df[[ label_col ]],
        random_state = 42,
    )

    assert len(train_df[label_col].unique()) == len(val_df[label_col].unique()) == len(test_df[label_col].unique())

    # properties of the base_model
    input_dim = 299
    scale_coef = 1./255

    # model hyperparameter
    batch_size = 64

    train_ds = load_image_dataset_from_df(
        train_df,
        batch_size,
        preprocessing,
    ) 
    val_ds = load_image_dataset_from_df(
        val_df,
        batch_size,
        preprocessing,
    ) 
    test_ds = load_image_dataset_from_df(
        test_df,
        batch_size,
        preprocessing,
    ) 

    return train_ds, val_ds, test_ds

if __name__ == '__main__':

    data_dir = '/media/data/gbif'
    hdf_filename = 'clean_data.h5'
    hdf_path = os.path.join( data_dir, hdf_filename )
    hdf_key = 'media_merged_filtered-by-species_350pt'

    label_col = 'acceptedScientificName'
    
    train_ds, val_ds, test_ds = load_image_datasets_from_hdf(
        hdf_path,
        hdf_key,
        label_col,
    )