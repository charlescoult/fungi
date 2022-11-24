import sys
import os
import io
import pandas as pd
import requests as rq
from tqdm import tqdm

# CSV files updated nightly
data_urls = {
        "observations": 'https://mushroomobserver.org/observations.csv',
        "images_observations": 'https://mushroomobserver.org/images_observations.csv',
        "names": 'https://mushroomobserver.org/names.csv',
        "locations": 'https://mushroomobserver.org/locations.csv',
        "name_descriptions": 'https://mushroomobserver.org/name_descriptions.csv',
        }

# unique row ids for each df
index_columns = {
        "observations": 'id',
        "images_observations": 'image_id',
        "names": 'id',
        "locations": 'id',
        "name_descriptions": 'id',
        }

# local file for storing data
data_hdf = 'data.h5'

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
    for key, url in data_urls.items():
        print( f'Downloading: {key}' )

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

        print( current_dfs[ key ].head() )
        print()

    return current_dfs

def get_old_data_dfs():

    old_dfs = {}

    if ( os.path.exists( data_hdf ) ):
        for key in data_urls.keys():
            old_dfs[ key ] = pd.read_hdf( data_hdf, key )
    else:
        print( f'File "{data_hdf}" does not exist.' )
        # populate with empty DataFrames
        for key in data_urls.keys():
            old_dfs[ key ] = pd.DataFrame()

    return old_dfs

def save_data_dfs( data_dfs, data_hdf = data_hdf ):
    for key, df in data_dfs.items():
        df.to_hdf( data_hdf, key )

# 'runs.h5' contains a list of files that need to be taken in by the next
# process in the data pipeline 
runs_hdf = 'runs.h5'
runs_df_key = 'runs'

def get_runs():
    if( os.path.exists( runs_hdf ) ):
        return pd.read_hdf( runs_hdf, runs_df_key )
    else:
        return 

def save_runs():
    pass

# returns only the rows in df2 whose 'id's are not in df1
# would this be easier if we set the id's to be the indexes? (df1.set_index('id'), etc.)
# if so, how would this be done?
def get_new_data( df1, df2 ):

    # JOIN RIGHT on 'id' column (how='right' not really necessary but saves computation)
    df = df1.merge( df2, on='id', suffixes = ('_DROP', ''), indicator=True, how='right')
    # Only keep 'id's from df2
    df = df.loc[ df['_merge'] == 'right_only' ]
    # get rid of the columns with the '_DROP' suffix (column values from df1)
    df = df.loc[ :, ~ df.columns.str.endswith('_DROP') ]
    # drop the indicator column from the merge
    df = df.drop( '_merge', axis = 1 )

    # one-liner
    # return df1.merge(df2, on='id', suffixes= ('_DROP',''), indicator=True, how='right').loc[lambda x : x['_merge'] == 'right_only' ].loc[ :, lambda x : ~x.columns.str.endswith('_DROP') ].drop( '_merge', axis=1)

    return df

def main():

    old_data_dfs = get_old_data_dfs()
    current_data_dfs = get_current_data_dfs()

    new_data_dfs = {}
    for key in data_urls.keys():
        new_data_df = get_new_data( old_data_dfs[ key ], current_data_dfs[ key ] )

    runs_df = get_runs()
    
    # create a new run






if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nProgram, interrupted.')
        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)

