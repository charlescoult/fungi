




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

dtypes = {
        'observations': {
            'id': 'int64',
            'name_id': 'int64',
            'when': 'int64',
            'location_id': 'int64',
            'lat': 'int64',
            'long': 'int64',
            'alt': 'int64',
            'vote_cache': 'int64',
            'is_collection_location': 'int64',
            'thumb_image_id': 'int64',
            },
        'images_observations': {
            },
        'names': {
            },
        'locations': {
            },
        'name_descriptions': {
            },
        }


