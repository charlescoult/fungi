from common import *

# TODO: this should really be in the input data pipeline (idp)

def downsample(
    ds_df,
    col_label,
    label_count,
    downsample,
):

    # if downsample param is 'min', downsample all classes to the same number of
    # samples as the class with the least samples
    if ( downsample == 'min' ):
        # get value counts for each class
        ds_df_label_vc_min = ds_df[ col_label ].value_counts().min()
        print('Downsampling to least number of samples per class: %d' % ds_df_label_vc_min)
        _data_count = ds_df_label_vc_min * label_count
    else:
        # manual override
        if ( downsample > 0 ):
            print( 'Overriding samples per class to: %d' % run['dataset']['downsample'] )
            ds_df_label_vc_min = run['dataset']['downsample']
            _data_count = downsample * label_count
        else: raise Exception("dataset downsample invalid")

    # downsample to ds_df_label_vc min datapoints per class
    ds_df_trans = ds_df.groupby(
        by = col_label,
    ).sample( n = ds_df_label_vc_min )

    # Assert the number of datapoints is what we expect
    assert len( ds_df ) == _data_count


def transform_dataset_df(
    ds_df,
    col_label,
    label_count,
    downsample = None,
):

    original_count = len( ds_df )

    # Downsample to equal number of samples per class if downsample param is set
    if ( downsample ):
        ds_df = downsample_df(
            ds_df,
            col_label,
            label_count,
            downsample,
        )

    # Assert no NaN values
    assert ds_df.isna().all().all() == False

    # We have transformed the original dataset through downsampling to produce a dataset where all classes have the same number of datapoints as the class with the least amount of datapoints.

    print( 'New datapoint count: %d' % len(ds_df) )

    return ds_df

