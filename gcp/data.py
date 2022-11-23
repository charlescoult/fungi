import pandas as pd
import os


default_md_fn = 'md.parquet'
data_folder = './data'
images_folder = os.path.join(data_folder, 'images')

# load metadata dataframe
def load( md_fn = os.path.join( data_folder, default_md_fn ):
    # may want to address potential read error if another process is loading/saving md file at the same time
    try:
        return pd.read_parquet( md_fn )
    except FileNotFoundError:
        # load columns into empty dataframe?
        return pd.DataFrame()

def save( df, md_fn = default_md_fn ):
    df.to_parquet( md_fn )

# should really be a class object?
# class MetaData():
# 
#     def __init__( self ):
#         self.df = load()
# 
#     def append(
# 
#     def save():
#         save( self.df )

