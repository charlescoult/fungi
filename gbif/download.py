


import pandas as pd
from parallel_download import download_parallel
from parallel_download import download_sequential


media = pd.read_hdf('clean_data.h5', 'media_merged')

download_parallel( 
# download_sequential( 
                  zip ( 
                       media['identifier_media'].to_numpy(), 
                       media.index.to_numpy()
                       )
                  )
