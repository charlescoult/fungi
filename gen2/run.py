from common import *
from sklearn.preprocessing import OneHotEncoder

import tensorflow_hub as hub
from keras.utils.layer_utils import count_params

from metadata import RunMeta
from base_model import base_models
from dataset import datasets
import dataset_util
from idp import augmentation_functions, make_idp
from model import get_model_name, gen_base_model_layer, gen_classifier_model_layer
import callbacks

import util

## Set logging to output INFO level to standard output
logging.basicConfig( level = os.environ.get( "LOGLEVEL", "INFO" ) )

## Set tf logging level to WARN
tf.get_logger().setLevel( 'WARN' )

## tf autotune parameters
AUTOTUNE = tf.data.AUTOTUNE

util.limit_memory_growth()

## Multi-GPU strategy
# strategy = tf.distribute.MirroredStrategy( devices = [ "/gpu:0", "/gpu:1" ] )
strategy = tf.distribute.MirroredStrategy()

#
def save_label_mapping(
    label_mapping,
    file_path = './label_mapping.json',
):
    with open( file_path, 'w' ) as f:
        json.dump( label_mapping, f, indent = 3 )

# The `run` RunMeta(dict) will keep track of this run's user-defined hyperparameters
# as well as generated parameters such as random seeds and file paths.
# This information will be saved in the `runs_hdf` specified.
def start_run(
    run = None,
):

    print('Starting Run\n')

    # use a formatted timestamp as the run's ID
    run['id'] = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    run['script_ver'] = version

    print( 'Run ID: %s' % run['id'] )
    print( 'Script version: %s' % run['script_ver'] )

    # the path of the directory where this run's files will be stored (metadata, saved model(s), etc.)
    run['path'] = os.path.join( run.runs_dir, str(run['id']) )
    print( 'Run Path: %s' % run['path'] )

    # TODO: allow loading of model weights from a previous run using its ID
    # load_weights = None

    run.save()

    # Timer
    timer = {}
    timer['start'] = time.perf_counter()

    # Read in source dataframe
    ds_df = pd.read_hdf(
        datasets[ run['dataset']['source'] ].path,
        datasets[ run['dataset']['source'] ].key,
    )

    col_label = datasets[ run['dataset']['source'] ].col_label

    # Creating common label_encoder for all IDPs
    label_encoder = OneHotEncoder( sparse_output = False )
    label_enc = label_encoder.fit_transform( ds_df[[ col_label ]] )

    label_enc_df = pd.DataFrame( label_enc, columns = label_encoder.get_feature_names_out([col_label]))

    # add encoded labels
    # ds_df.drop( col_label, axis = 1 )
    ds_df = ds_df.join( label_enc_df )

    ### Dataset Information

    # save classes
    ds_classes = ds_df[ datasets[ run['dataset']['source'] ].col_label ].unique().tolist()
    print('Label count: %d' % len(ds_classes))
    print('Datapoint count: %d' % len(ds_df))

    run['label_mapping_path'] = os.path.join( run['path'], 'label_mapping.json' )
    save_label_mapping(
        ds_classes,
        file_path = run['label_mapping_path'],
    )

    # get value counts for each class
    ds_df_label_vc = ds_df[ datasets[ run['dataset']['source'] ].col_label ].value_counts()
    ds_df_label_vc = ds_df_label_vc.sort_values( ascending = False )

    ### Dataset Transformation
    # (downsample, upsample, etc.)

    # Downsample to equal number of samples per class if downsample param is set
    if ( run['dataset']['downsample'] ):
        # if downsample param is 'min', downsample all classes to the same number of
        # samples as the class with the least samples
        if ( run['dataset']['downsample'] == 'min' ):
            ds_df_label_vc_min = ds_df_label_vc.min()
            print('Downsampling to least number of samples per class: %d' % ds_df_label_vc_min)
        else:
            # manual override
            if ( run['dataset']['downsample'] > 0 ):
                print( 'Overriding samples per class to: %d' % run['dataset']['downsample'] )
                ds_df_label_vc_min = run['dataset']['downsample']
            else: raise Exception("dataset downsample invalid")

        # downsample to ds_df_label_vc min datapoints per class
        ds_df_trans = ds_df.groupby(
            by = datasets[ run['dataset']['source'] ].col_label,
        ).sample( n = ds_df_label_vc_min )
    else:
        ds_df_trans = ds_df


    # We have transformed the original dataset through downsampling to produce a dataset where all classes have the same number of datapoints as the class with the least amount of datapoints.

    print( 'New datapoint count: %d' % len(ds_df_trans) )

    ### Train, Validation, Test Split

    run['dataset']['split_test'] = 0.05
    run['dataset']['split_val'] = 0.1

    # generate random states for reproducability
    import random

    # [0, 2**32 - 1]
    run['dataset']['seed_split_test'] = random.randint( 0, 2**32 - 1 )
    run['dataset']['seed_split_val'] = random.randint( 0, 2**32 - 1 )

    ds_train, ds_val, ds_test, split_shuffle_seed = dataset_util.train_val_test_split_stratified(
        ds_df_trans,
        col_label = datasets[ run['dataset']['source'] ].col_label,
        val_split = run['dataset']['split_val'],
        test_split = run['dataset']['split_test'],
    )

    # set augmentation_func to None if no augmentation is desired
    # augmentation_func = augmentation_functions[0]
    augmentation_func = augmentation_functions[0] if run['dataset']['data_augmentation'] else None

    # Determines if data augmentation should be done in the IDP or in the model
    # Data augmentation will
    data_augmentation_in_ds = True

    # Determines if preprocessing should be done in the IDP or in the model
    preprocessing_in_ds = True

    ds_train = ds_train.drop( col_label, axis = 1 )
    ds_val = ds_val.drop( col_label, axis = 1 )
    ds_test = ds_test.drop( col_label, axis = 1 )

    # IDP creation
    ds_idp_train, run['dataset']['seed_shuffle'] = make_idp(
        ds_train[ datasets[ run['dataset']['source'] ].col_filename ].values,
        ds_train.filter( regex = ( datasets[ run['dataset']['source'] ].col_label + '+' ) ).values,
        # ds_train[ datasets[ run['dataset']['source'] ].col_label ].values,
        input_dim = base_models[ run['model']['base'] ].input_dim,
        is_training = True,
        batch_size = run['batch_size'],
        augmentation_func = augmentation_func if data_augmentation_in_ds == True else None,
        preprocessor = base_models[ run['model']['base'] ].preprocessor if preprocessing_in_ds else None,
        # label_encoder = label_encoder,
    )

    ds_idp_val, _ = make_idp(
        ds_val[ datasets[ run['dataset']['source'] ].col_filename ].values,
        ds_val.filter( regex = ( datasets[ run['dataset']['source'] ].col_label + '+' ) ).values,
        # ds_val[ datasets[ run['dataset']['source'] ].col_label ].values,
        input_dim = base_models[ run['model']['base'] ].input_dim,
        is_training = False,
        batch_size = run['batch_size'],
        # turned off by is_training = False anyway...
        augmentation_func = None,
        preprocessor = base_models[ run['model']['base'] ].preprocessor if preprocessing_in_ds else None,
        # label_encoder = label_encoder,
    )

    ds_idp_test, _ = make_idp(
        ds_test[ datasets[ run['dataset']['source'] ].col_filename ].values,
        ds_test.filter( regex = ( datasets[ run['dataset']['source'] ].col_label + '+' ) ).values,
        # ds_test[ datasets[ run['dataset']['source'] ].col_label ].values,
        input_dim = base_models[ run['model']['base'] ].input_dim,
        is_training = False,
        batch_size = run['batch_size'],
        # turned off by is_training = False anyway...
        augmentation_func = None,
        preprocessor = base_models[ run['model']['base'] ].preprocessor if preprocessing_in_ds else None,
        # label_encoder = label_encoder,
    )

    ## Model Building

    # Initialize full model
    with strategy.scope():
        full_model = tf.keras.Sequential( name = "full_model" )

        # if preprocessing_in_ds, then input is assumed to be preprocessed correctly from input dataset pipeline (idp)
        # else, add preprocessing layer to model
        if ( not preprocessing_in_ds ):
            raise Exception('not yet implemented')

        # Add base model to full_model
        full_model.add( gen_base_model_layer(
            name = get_model_name( base_models[ run['model']['base'] ].source ),
            source = base_models[ run['model']['base'] ].source,
            input_dim = base_models[ run['model']['base'] ].input_dim,
            trainable = True,
        ) )

        # Add classifier model to full_model
        # TODO allow selection between different classification models
        full_model.add( gen_classifier_model_layer(
            num_classes = len( ds_classes ),
            dropout = run['model']['classifier']['dropout'],
            add_softmax = run['model']['classifier']['output_normalize'],
        ) )

    # TODO: allow loading of model weights from previous run
    load_weights = None

    # Compile model
    # Sparse vs non-sparse CCE https://www.kaggle.com/general/197993
    with strategy.scope():
        full_model.compile(
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = run['model']['learning_rate']
            ),
            # loss = tf.keras.losses.SparseCategoricalCrossentropy(
            #     from_logits = True,
            # ),
            loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits = not run['model']['classifier']['output_normalize'],
                label_smoothing = run['model']['label_smoothing'],
            ),
            metrics = [
                'accuracy',
                tf.keras.metrics.AUC(),
                # tf.keras.metrics.SparseCategoricalCrossentropy(),
                # tf.keras.metrics.SparseTopKCategoricalAccuracy(
                #     k = 3,
                #     name = "Top3",
                # ),
                # tf.keras.metrics.SparseTopKCategoricalAccuracy(
                #     k = 10,
                #     name="Top10",
                # ),
                # tf.keras.metrics.CategoricalCrossentropy(),            
                # tf.keras.metrics.TopKCategoricalAccuracy( k=3, name="Top3" ),
                # tf.keras.metrics.TopKCategoricalAccuracy( k=10, name="Top10" ),
            ],
        )

    callbacks_list = [
        # Tensorboard logs
        callbacks.TensorBoard(
            path = run['path'],
        ),
        # Early stopping
        callbacks.EarlyStopping(
            run['callbacks']['early_stopping']['monitor'],
            run['callbacks']['early_stopping']['patience'],
            run['callbacks']['early_stopping']['restore_best_weights'],
            run['callbacks']['early_stopping']['start_from_epoch'],
        ),
        # Model Checkpoints for saving best model weights
        callbacks.ModelCheckpoint(
            run['path'],
            save_best_only = True,
            monitor = 'val_loss',
            # mode = 'min', # should be chosen correctly based on monitor value
        ),
    ]

    run.save()

    # Train
    timer['train_start'] = time.perf_counter()

    try:
        with strategy.scope():
            history = full_model.fit(
                ds_idp_train,
                validation_data = ds_idp_val,
                epochs = run['max_epochs'],
                callbacks = callbacks_list,
                # validation_freq=2,
            )
        pass
    except KeyboardInterrupt:
        print('\n\nInterrupted...')
        # run['interrupted'] = True
    else:
        print('Completed.')
        # run['interrupted'] = False

    timer['train_end'] = time.perf_counter()


    run['time'] = timer['train_end'] - timer['train_start']
    print(run['time'])


    run.save()


    print( len( history.epoch ) )


    # ## Testing
    test_labels = np.concatenate([y for x, y in ds_idp_test], axis = 0)

    with strategy.scope():
        predictions = full_model.predict(
            ds_idp_test,
        )

    cm = tf.math.confusion_matrix(
        np.argmax( test_labels, axis=1),
        np.argmax( predictions, axis=1),
    )

    f1 = sklearn.metrics.f1_score(
        np.argmax( test_labels, axis = 1 ),
        np.argmax( predictions, axis = 1 ),
        average = 'micro',
    )

    run['scores'] = {
        'f1': f1,
    }

    run.save()

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate your vaccination QR code.")
    parser.add_argument('runs_dir')
    parser.add_argument('runs_hdf')
    parser.add_argument('runs_hdf_key')
    args = parser.parse_args()

    print( args )


if __name__ == '__main__':

    runs_dir = '/media/data/runs'
    runs_hdf = 'runs.h5'
    runs_hdf_key = 'runs'

    runMeta = RunMeta(
        {
            'batch_size': 32,
            'max_epochs': 20,
            'model': {
                'base': 'Inception_v3_iNaturalist',
                'classifier': {
                    # dropout % of dense layer(s) of classifier
                    'dropout': 0.2,
                    # normalize output with a softmax?
                    'output_normalize': False,
                },
                'learning_rate': 0.01, # Adam Optimizer
                # label smoothing
                'label_smoothing': 0.1,
            },
            'dataset': {
                'data_augmentation': False,
                # 'downsample': 'min' or a number indicating the max number of samples per class to allow
                'downsample': None,
                # 'downsample': 20,
                # the key of the source described in `datasets`
                # 'source': 'gbif',
                'source': 'flowers',
                # test split
                'split_test': 0.05,
                # val split
                'split_val': 0.2,

            },
            'callbacks': {
                'early_stopping': {
                    'monitor': 'val_loss',
                    'patience': 10,
                    'restore_best_weights': True,
                    'start_from_epoch': 5,
                }
            }
        },
        runs_dir = runs_dir,
        runs_hdf = runs_hdf,
        runs_hdf_key = runs_hdf_key,
    )
    start_run( runMeta )
