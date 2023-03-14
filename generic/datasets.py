import pathlib
import tensorflow as tf
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from util import timeit
from models import Model
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DatasetGroup:

    def __init__(
        self,
        train_ds,
        val_ds,
        test_ds,
        classes,
    ):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.classes = classes

def load_image(
    filename,
    label,
):
    img_raw = tf.io.read_file( filename )
    img_tensor = tf.image.decode_image(
        img_raw,
        dtype = tf.dtypes.float32,
        channels = 3,
        # Important to include the following
        # https://stackoverflow.com/questions/44942729/tensorflowvalueerror-images-contains-no-shape
        expand_animations = False,
    )
    return ( img_tensor, label )

def preprocessing_default(
    img_tensor,
    input_dim,
):
    img_tensor_preprocessed = img_tensor_preprocessed/255.0
    return img_tensor_preprocessed

def run_preprocessing(
    img_tensor,
    label,
    preprocessing,
    input_dim,
):
    img_tensor_preprocessed = tf.image.resize(
        img_tensor,
        [ input_dim, input_dim ],
    )
    return ( preprocessing(img_tensor_preprocessed), label )

def load_image_dataset_from_df(
    df,
    batch_size,
    input_dim,
    preprocessing,
    label_col,
    classes,
    filename_col = 'filename',
):
    ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant( df[ filename_col ].values ),
            tf.constant( df[ label_col ].values )
        ),
    )

    ds = ds.map(
        load_image,
        num_parallel_calls = AUTOTUNE,
    )

    ds = ds.map(
        lambda img, label: run_preprocessing(
            img,
            label,
            preprocessing,
            input_dim,
        ),
        num_parallel_calls = AUTOTUNE,
    )

    ds = ds.batch( batch_size )

    ds = ds.prefetch( buffer_size = AUTOTUNE )

    # using cache only seems to negatively impact performance
    # ds.cache( filename = './cache.tf-data' )

    return ds

def downsample( df ):

    min = 100

    # print(df['media_count_per_taxonID'].min())
    limited_df = df.groupby(by='taxonID').head( min )
    newCounts = limited_df['taxonID'].map( limited_df['taxonID'].value_counts() )
    # print(newCounts.max())

    return limited_df

def load_image_datasets_from_hdf(
    hdf_path,
    hdf_key,
    label_col,
    test_size = 0.05,
    val_size = 0.1,
    input_dim = 299,
    preprocessing = preprocessing_default,
):

    df = pd.read_hdf( hdf_path, hdf_key )

    df = downsample(df)

    classes = df[ label_col ].unique()

    df[ label_col ] = df[ label_col ].astype('category').cat.codes

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

    # assert that each dataset has the same number of classes
    assert len(classes) == len(train_df[label_col].unique()) == len(val_df[label_col].unique()) == len(test_df[label_col].unique())

    # model hyperparameter
    batch_size = 64

    train_ds = load_image_dataset_from_df(
        train_df,
        batch_size,
        input_dim,
        preprocessing,
        label_col,
        classes,
    ) 
    val_ds = load_image_dataset_from_df(
        val_df,
        batch_size,
        input_dim,
        preprocessing,
        label_col,
        classes,
    ) 
    test_ds = load_image_dataset_from_df(
        test_df,
        batch_size,
        input_dim,
        preprocessing,
        label_col,
        classes,
    ) 

    return DatasetGroup(
        train_ds,
        val_ds,
        test_ds,
        classes,
    )

if __name__ == '__main__':

    data_dir = '/media/data/gbif'
    hdf_filename = 'clean_data.h5'
    hdf_path = os.path.join( data_dir, hdf_filename )
    hdf_key = 'media_merged_filtered-by-species_350pt'

    label_col = 'acceptedScientificName'

    base_model = Model.base_models_meta[0]
    
    dataset_group = load_image_datasets_from_hdf(
        hdf_path,
        hdf_key,
        label_col,
        input_dim = base_model[1],
        preprocessing = base_model[2],
    )

    for img, label in dataset_group.train_ds.take(1):
        print(img.shape)
        print(label.shape)

