from common import *

class DatasetHDFSource:

    def __init__(
        self,
        path,
        key,
        col_filename = 'filename',
        col_label = 'label',
    ):
        self.path = path
        self.key = key
        self.col_filename = col_filename
        self.col_label = col_label

## Enumerate Available Datasets
datasets = {
    'gbif': DatasetHDFSource(
        '/media/data/gbif/clean_data.h5',
        'media_merged_filtered-by-species_350pt',
        col_label = 'acceptedScientificName',
    ),
    'cub': DatasetHDFSource(
        '/media/data/cub/cub.h5',
        'cub',
        col_filename = 'file_path',
        col_label = 'class_name',
    ),
    'flowers': DatasetHDFSource(
        '/media/data/flowers/flowers.h5',
        'flowers',
        # col_filename = 'filename',
        col_label = 'class',
    ),
}




