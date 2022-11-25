import sys
import os
import io
import pandas as pd
import requests as rq
from tqdm import tqdm

import schema

# must pass a valid user-agent in header for GET request otherwise error 403 returned
request_header = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'
        }

def get_current_data_dfs():

    current_dfs = {}

    # todo: parallelize downlaods
    for key, url in schema.data_urls.items():
        print( f'Downloading: {key}' )

        retrieved = pd.Timestamp.now()

        # using tqdm
        current_dfs[ key ] = pd.concat([ chunk for chunk in tqdm(
            pd.read_csv(
                url,
                sep = '\t',
                lineterminator = '\n',
                storage_options = request_header,
                chunksize=1000,
                )
            , desc='Loading data' ) ])

        # using requests as intermediary
        # csv_file = io.StringIO( rq.get( url ).content.decode('utf-8') )
        # current_dfs[ key ] = pd.read_csv( csv_file )

        # set index according to definitions in schema.index_columns
        current_dfs[ key ].set_index( schema.index_columns[ key ] )

        # Add a timestamp for when the data was collected
        # current_dfs[ key ].assign( **{ '_retrieved': retrieved } )
        current_dfs[ key ] = current_dfs[ key ].assign( **{ '_retrieved': retrieved } )

        print( current_dfs[ key ].head() )
        print()

    return current_dfs

def get_old_data_dfs( data_hdf = schema.data_hdf):

    old_dfs = {}

    if ( os.path.exists( data_hdf ) ):
        for key in schema.data_urls.keys():
            old_dfs[ key ] = pd.read_hdf( data_hdf, key )
    else:
        print( f'File "{data_hdf}" does not exist.' )
        # populate with empty DataFrames
        for key in schema.data_urls.keys():
            old_dfs[ key ] = pd.DataFrame()

    return old_dfs

def save_data_dfs( data_dfs, data_hdf = schema.data_hdf ):
    for key, df in data_dfs.items():
        df.to_hdf( data_hdf, key )

def main():

    old_data_dfs = get_old_data_dfs()
    current_data_dfs = get_current_data_dfs()

    new_data_dfs = {}
    for key in schema.data_urls.keys():
        # concat new and old dfs, dropping those with duplicate indicies, keeping the values of the old_data
        # in order to retain the older timestamps

        # select rows in current data where the index (id) is not in the old data
        only_new_data_df = current_data_dfs[ key ][ ~ current_data_dfs[ key ].index.isin( old_data_dfs[ key ] ) ]
        # add the new rows to the old data
        new_data_dfs[ key ] = pd.concat( [ old_data_dfs[ key ], only_new_data_df ] )

    save_data_dfs( new_data_dfs )

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nProgram, interrupted.')
        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)

