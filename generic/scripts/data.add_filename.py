import os
import mimetypes as mt
import pandas as pd

# Adds 'filename' column to clean_data.py['media_merged_filtered-by-species_350pt'] dataframe
## 'media_merged_filtered-by-species_350pt' - df of media where each class ('acceptedScientificName')
## has at least 350 datapoints (pt = 'per taxon')


data_dir = '/media/data/gbif'
hdf_filename = 'clean_data.h5'
hdf_file = os.path.join( data_dir, hdf_filename )
hdf_key = 'media_merged_filtered-by-species_350pt'

label_name = 'acceptedScientificName'

df = pd.read_hdf( hdf_file, hdf_key )

# get extension
mt.add_type('image/pjpeg', '.jpg')
df['extension'] = df['format'].map(lambda x: mt.guess_extension(x, strict=False))

# print(df.columns)

# df['filename'] = df.apply( lambda x: os.path.join( '/media/data/gbif/media/', x[label_name], str(x.index) ) + x['extension'], axis = 1 )
for index, row in df.iterrows():
    ext = row['extension']
    label = row['acceptedScientificName']
    filename = os.path.join( data_dir, 'media', label, f'{index}{ext}' )
    df.at[index, 'filename'] = filename

df.head()['filename'].map( lambda x: print(x))

df['file_exists'] = df['filename'].map( lambda x: os.path.isfile(x))
assert df['file_exists'].all()
df = df.drop( 'file_exists', axis = 1 )

print(df.columns)

df.to_hdf( hdf_file, hdf_key )