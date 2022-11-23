import json
import os
import sys
import pandas as pd
from google_images_search import GoogleImagesSearch

import data as md

def get_api_info( fn ):
    with open(fn) as f:
        return json.load(f)

api_info_fn = '/home/charlescoult/.api/gcp/fungi.json'
api_info = get_api_info( api_info_fn )
gis = GoogleImagesSearch( api_info['API_KEY'], api_info['CX'] )

def query(
        q_str,
        num_images = 10,
        filetype = 'jpg|png',
        img_type = 'photo',
        ):

    search_params = {
            'q': q_str,
            'num': num_images,
            # 'fileType': 'jpg|gif|png',
            'fileType': filetype,
            # 'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived',
            'safe': 'active', ##
            # 'safe': 'active|high|medium|off|safeUndefined', ##
            'imgType': img_type, ##
            # 'imgType': 'clipart|face|lineart|stock|photo|animated|imgTypeUndefined', ##
            # 'imgSize': 'huge|icon|large|medium|small|xlarge|xxlarge|imgSizeUndefined', ##
            # 'imgDominantColor': 'black|blue|brown|gray|green|orange|pink|purple|red|teal|white|yellow|imgDominantColorUndefined', ##
            # 'imgColorType': 'color|gray|mono|trans|imgColorTypeUndefined' ##
            }

    # folder setup
    # files are downloaded to download_folder, resized, then moved to images_folder with
    # a new name, 
    download_folder = os.path.join(md.data_folder, 'download')
    images_folder = os.path.join(md.data_folder, 'images')

    # retrieve existing log, if it exists, otherwise start anew
    md_log = md.load()

    # search first, then download and resize afterwards:
    query_timestamp = pd.Timestamp.now()
    gis.search( search_params = search_params )
    for i, image in enumerate(gis.results()):
        image.url  # image direct url
        image.referrer_url  # image referrer url (source)

        # may not need to download if we work only in memory...
        # for the future...
        image.download( download_folder )  # download image
        image.resize(500, 500)  # resize downloaded image
        print(type(image))
        os.rename( image.path, os.path.join( images_folder, f'{i}' ) )

        image.path  # downloaded local file path

        image_md = {
            'query.str': q_str,
            'query.index': i,
            'query.time': query_timestamp,
            'url': image.url,
            'referrer_url': image.referrer_url,
            'local_path': image.path,
        }

        md_df.append(image_md, ignore_index=True)

    print(md_df.head())
    md.save( md_df )

        # label_image( image.path )

def main():
    q_str = sys.argv[1:]
    query( q_str )

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nProgram, interrupted.')
        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)
