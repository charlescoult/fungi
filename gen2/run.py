from common import *
from sklearn.preprocessing import OneHotEncoder

import tensorflow_hub as hub
from keras.utils.layer_utils import count_params

from metadata import RunMeta
from base_model import base_models
from dataset import datasets
import dataset_util
from idp import augmentation_functions, make_idp
from model import get_model_name, gen_base_model_layer, gen_classifier_model_layer, print_weight_counts
import callbacks
from scoring import score
from transformation import transform_dataset_df

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

    _data_count = len( ds_df )

    col_filename = datasets[ run['dataset']['source'] ].col_filename
    col_label = datasets[ run['dataset']['source'] ].col_label
    assert col_label in ds_df.columns
    assert col_filename in ds_df.columns

    _label_count = len( ds_df[ col_label ].unique().tolist() )

    # drop unnecessary columns
    ds_df = ds_df[[
        col_filename,
        col_label,
    ]]

    assert len( ds_df ) == _data_count
    assert ds_df.isna().all().all() == False

    # Creating common label_encoder for all IDPs
    label_encoder = OneHotEncoder( sparse_output = False )
    label_enc = label_encoder.fit_transform( ds_df[[ col_label ]] )

    assert len( label_encoder.categories_[0] ) == _label_count
    assert len( label_enc ) == _data_count

    label_enc_df = pd.DataFrame( label_enc, columns = label_encoder.get_feature_names_out([col_label]) )

    assert len( label_enc_df ) == _data_count
    assert label_enc_df.isna().all().all() == False

    # source dataframe can have a non-sequential index
    # THIS BUG TOOK ME WAY TOO LONG TO FIGURE OUT.
    ds_df = ds_df.reset_index()

    assert len( ds_df ) == _data_count
    assert ds_df.isna().all().all() == False

    # add the one-hot columns to the DataFrame
    ds_df = pd.concat( [ ds_df, label_enc_df ], axis = 1 )

    assert len( ds_df ) == _data_count
    assert ds_df.isna().all().all() == False

    ### Dataset Information

    # save classes
    ds_classes = ds_df[ col_label ].unique().tolist()
    print('Label count: %d' % len(ds_classes))
    print('Datapoint count: %d' % len(ds_df))

    assert len( ds_classes ) == _label_count

    run['label_mapping_path'] = os.path.join( run['path'], 'label_mapping.json' )
    save_label_mapping(
        ds_classes,
        file_path = run['label_mapping_path'],
    )

    ### Dataset Transformation
    # (downsample, upsample, etc.)

    # this should really be in the input data pipeline (idp)
    ds_df_trans = transform_dataset_df(
        ds_df,
        col_label,
        _label_count,
        downsample = run['dataset']['downsample'],
    )

    # Assert no NaN values
    assert ds_df_trans.isna().all().all() == False

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
        col_label = col_label,
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

    print("Creating TRAIN")

    # IDP creation
    ds_idp_train, run['dataset']['seed_shuffle'] = make_idp(
        ds_train[ col_filename ].values,
        ds_train.filter( regex = ( col_label + '+' ) ).values,
        # ds_train[ col_label ].values,
        input_dim = base_models[ run['model']['base'] ].input_dim,
        is_training = True,
        batch_size = run['batch_size'],
        augmentation_func = augmentation_func if data_augmentation_in_ds == True else None,
        preprocessor = base_models[ run['model']['base'] ].preprocessor if preprocessing_in_ds else None,
        # label_encoder = label_encoder,
    )
    print("Creating VAL")

    ds_idp_val, _ = make_idp(
        ds_val[ col_filename ].values,
        ds_val.filter( regex = ( col_label + '+' ) ).values,
        # ds_val[ col_label ].values,
        input_dim = base_models[ run['model']['base'] ].input_dim,
        is_training = False,
        batch_size = run['batch_size'],
        # turned off by is_training = False anyway...
        augmentation_func = None,
        preprocessor = base_models[ run['model']['base'] ].preprocessor if preprocessing_in_ds else None,
        # label_encoder = label_encoder,
    )

    print("Creating TEST")

    ds_idp_test, _ = make_idp(
        ds_test[ col_filename ].values,
        ds_test.filter( regex = ( col_label + '+' ) ).values,
        # ds_test[ col_label ].values,
        input_dim = base_models[ run['model']['base'] ].input_dim,
        is_training = False,
        batch_size = run['batch_size'],
        # turned off by is_training = False anyway...
        augmentation_func = None,
        preprocessor = base_models[ run['model']['base'] ].preprocessor if preprocessing_in_ds else None,
        # label_encoder = label_encoder,
    )

    print("IDPs created")

    '''
    # Peek at a batch to ensure compliance with expected values
    for el in next( iter( ds_idp_train.as_numpy_iterator() ) ):
        print(el[0].shape)

        # image resized correctly
        assert el[0].shape[0] == el[0].shape[1] == base_models[ run['model']['base'] ].input_dim
        # 3 channels
        assert el[0].shape[2] == 3

        print('Min and max will depend on what the base_model is expecting from preprocessor function.')
        print(el[0].min())
        print(el[0].max())
    '''


    ## Model Building

    print("Building model")

    # Initialize full model
    full_model = tf.keras.Sequential( name = "full_model" )

    # if preprocessing_in_ds, then input is assumed to be preprocessed correctly from input dataset pipeline (idp)
    # else, add preprocessing layer to model
    if ( not preprocessing_in_ds ):
        raise Exception('not yet implemented')

    print( "Adding base")

    # Add base model to full_model
    full_model.add( gen_base_model_layer(
        name = get_model_name( base_models[ run['model']['base'] ].source ),
        source = base_models[ run['model']['base'] ].source,
        input_dim = base_models[ run['model']['base'] ].input_dim,
        trainable = True,
    ) )

    print( "Adding classifier")

    # Add classifier model to full_model
    # TODO allow selection between different classification models
    full_model.add( gen_classifier_model_layer(
        num_classes = len( ds_classes ),
        dropout = run['model']['classifier']['dropout'],
        add_softmax = run['model']['classifier']['output_normalize'],
    ) )

    # TODO: allow loading of model weights from previous run
    load_weights = None

    print_weight_counts(full_model)
    print( full_model.summary( expand_nested = True, ) )

    # Compile model
    # Sparse vs non-sparse CCE https://www.kaggle.com/general/197993
    full_model.compile(
        optimizer = tf.keras.optimizers.Adam(
            learning_rate = run['model']['learning_rate']
        ),
        # loss = tf.keras.losses.SparseCategoricalCrossentropy(
        #     from_logits = True,
        # ),
        loss = tf.keras.losses.CategoricalCrossentropy(
            # from_logits = True,
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
            tf.keras.metrics.CategoricalCrossentropy(),
            tf.keras.metrics.TopKCategoricalAccuracy( k = 3, name = "Top3" ),
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
        history = full_model.fit(
            ds_idp_train,
            validation_data = ds_idp_val,
            epochs = run['max_epochs'],
            callbacks = callbacks_list,
            # validation_freq=2,
        )
    except KeyboardInterrupt:
        print('\n\nInterrupted...')
        # run['interrupted'] = True
    else:
        print('Completed.')
        # run['interrupted'] = False

    timer['train_end'] = time.perf_counter()

    ## End-of-run metrics

    run['time'] = timer['train_end'] - timer['train_start']
    print( 'Run took: %d' % run['time'] )

    run['epoch_count'] = len( history.epoch )
    print( '%d epochs run' % run['epoch_count'] )

    ## Test data scoring

    # get labels
    test_labels = np.concatenate([y for x, y in ds_idp_test], axis = 0)

    # get predictions
    predictions = full_model.predict(
        ds_idp_test,
    )

    # score results
    run['scores'] = score(
        test_labels,
        predictions,
        not run['model']['classifier']['output_normalize'],
    )

    run.save()

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

    with strategy.scope():
        start_run( runMeta )
